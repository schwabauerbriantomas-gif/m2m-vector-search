"""
M2MPersistence - Capa de persistencia en capas para M2M.

Arquitectura:
    storage/
      data/
        vectors/    # Arrays numpy (shard_NNN.npy)
        metadata/   # SQLite database
      index/
        hrm2.idx    # Índice jerárquico serializado
      wal/
        wal.log     # Write-Ahead Log
        checkpoint/ # Checkpoints periódicos
      energy/
        landscape.npz  # Estado del paisaje energético
"""

import hashlib
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .wal import WriteAheadLog

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest"  # -> <storage_path>/index/manifest.idx


def _confined_path(root: Path, name: str, *parents: str) -> Path:
    """Resuelve ``root/parents.../name`` exigiendo que quede DENTRO de ``root``.

    Confinamiento de rutas (audit r3a-17 resto): rechaza con ValueError ``..``,
    rutas absolutas y symlinks que escapen del directorio de storage. El ``..``
    literal ya se bloqueaba en storage_path/backup_path, pero un symlink en
    ``index/`` o en un shard eludía esa protección.
    """
    candidate = Path(str(name))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.name:
        raise ValueError(
            f"Path traversal detected in {name!r}: must be a file name inside the storage directory"
        )
    resolved = (root.joinpath(*parents) / candidate).resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents:
        raise ValueError(
            f"Path escapes the storage directory: {name!r} resolves to {resolved}, "
            f"outside {root_resolved}"
        )
    return resolved


class IndexTamperedError(RuntimeError, ValueError):
    """Un storage firmado no verifica (HMAC o contenido alterado).

    Hereda de ValueError para conservar el contrato previo de load_index()
    (audit r3a-17) y de RuntimeError para el flujo save()/load().
    """


class _RestrictedUnpickler(pickle.Unpickler):
    """
    Defense-in-depth: even with HMAC verification, restrict unpickling
    to safe types only. Blocks os.system, subprocess, eval, etc.
    """

    _SAFE_MODULES = {
        "numpy": {"ndarray", "dtype", "float32", "float64", "int32", "int64", "uint32", "uint64"},
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
        "numpy._core.multiarray": {"_reconstruct", "scalar"},
        "numpy.core.numeric": {"_frombuffer"},
        "numpy._core.numeric": {"_frombuffer"},
        "builtins": {"dict", "list", "tuple", "float", "int", "str", "bytes", "bool", "set", "frozenset", "NoneType", "complex"},
        "collections": {"OrderedDict", "defaultdict"},
        # m2m's own dataclasses are safe to reconstruct
        "m2m.dataset_transformer": {"GaussianSplat", "HRM2Node", "TransformConfig"},
    }

    def find_class(self, module: str, name: str):
        allowed = self._SAFE_MODULES.get(module, set())
        if name in allowed:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked unsafe class during unpickling: {module}.{name}"
        )


class M2MPersistence:
    """
    Gestión de persistencia para M2M Vector DB.

    Maneja:
    - Vectores (numpy memmap/shard files)
    - Metadata (SQLite)
    - Índice HRM2 (pickle serializado)
    - WAL para durabilidad
    - Backups y restore
    """

    def __init__(self, storage_path: str, enable_wal: bool = True):
        """
        Args:
            storage_path: Directorio raíz de almacenamiento
            enable_wal: Si True, habilita Write-Ahead Log
        """
        # -- H-01 FIX: Path traversal prevention -----------------------
        resolved = Path(storage_path).resolve()
        if ".." in Path(storage_path).parts:
            raise ValueError(f"Path traversal detected in storage_path: {storage_path}")

        self.storage_path = resolved
        self.enable_wal = enable_wal
        # RLock: _get_conn() flushea la metadata pendiente tomando este mismo
        # lock desde adentro de una sección crítica (update_vector).
        self._lock = threading.RLock()
        # Metadata entregada a save_metadata() que aún no se commiteó, y filas
        # que faltan entregar del lote abierto por el último save_vectors()
        # (audit r3a-17 resto: UNA transacción por add(), no una por documento).
        self._pending_metadata: List[Tuple] = []
        self._batch_expected = 0

        # Crear estructura de directorios
        self._create_directories()

        # Inicializar SQLite metadata
        self._init_metadata_db()

        # Inicializar WAL
        if enable_wal:
            wal_path = str(self.storage_path / "wal" / "wal.log")
            self.wal = WriteAheadLog(wal_path, sync_interval=50)
        else:
            self.wal = None

    def _create_directories(self):
        """Crea la estructura de directorios de storage."""
        dirs = [
            "data/vectors",
            "data/metadata",
            "index",
            "wal/checkpoint",
            "energy",
        ]
        for d in dirs:
            (self.storage_path / d).mkdir(parents=True, exist_ok=True)

    def _init_metadata_db(self):
        """Inicializa la base de datos SQLite de metadata."""
        db_path = str(self.storage_path / "data" / "metadata" / "metadata.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                shard_idx INTEGER,
                vector_idx INTEGER,
                metadata TEXT,
                document TEXT,
                deleted INTEGER DEFAULT 0,
                created_at REAL,
                updated_at REAL
            )
        """)
        # Bases viejas sin la columna `deleted`: agregarla (aditivo) para que el
        # índice y las lecturas toleren el esquema anterior (audit r1-01).
        if "deleted" not in self._document_columns(conn):
            conn.execute("ALTER TABLE documents ADD COLUMN deleted INTEGER DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deleted ON documents(deleted)")
        conn.commit()
        conn.close()
        self._db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        """Obtiene conexión SQLite thread-safe (flusheando antes la metadata pendiente)."""
        self._flush_pending_metadata()
        return sqlite3.connect(self._db_path, check_same_thread=False)

    # -------------------------------------------------------------------------
    # Vector Storage (numpy shards)
    # -------------------------------------------------------------------------

    def save_vectors(self, vectors: np.ndarray, ids: List[str], shard_name: Optional[str] = None):
        """Guarda vectores en un archivo numpy shard.

        Con shard_name=None (recomendado) se asigna automáticamente el próximo
        shard libre (shard_001, shard_002, ...): cada add() escribe su propio
        archivo en vez de SOBRESCRIBIR shard_001 y destruir los lotes previos
        (data loss reproducido en la auditoría: 20 docs irrecuperables tras
        add(20)+add(5)+restart).
        """
        # Un shard nuevo abre un lote de metadata nuevo: lo pendiente de un lote
        # anterior (incompleto) queda durable ANTES de escribir este shard.
        with self._lock:
            self._flush_pending_metadata()
        if shard_name is None:
            existing = self.list_shards()
            n = len(existing) + 1
            shard_name = f"shard_{n:03d}"
        shard_path = _confined_path(self.storage_path, f"{shard_name}.npy", "data", "vectors")
        np.save(str(shard_path), vectors.astype(np.float32))

        # shard_idx (1-based) es lo que restore usa para mapear doc -> shard
        try:
            shard_idx = int(shard_name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            shard_idx = len(self.list_shards())

        if self.wal:
            self.wal.log_operation(
                "add_vectors",
                {
                    "ids": ids,
                    "shard": shard_name,
                    "count": len(vectors),
                    "dim": vectors.shape[1] if vectors.ndim > 1 else vectors.shape[0],
                },
            )
        self._last_shard_name = shard_name
        self._last_shard_idx = shard_idx
        # Filas de metadata que el caller entregará para este shard: el lote se
        # commitea en UNA transacción al recibir la última.
        with self._lock:
            self._batch_expected = 1 if vectors.ndim == 1 else int(len(vectors))

    def load_vectors(self, shard_name: str = "shard_001") -> Optional[np.ndarray]:
        """Carga vectores de un shard."""
        shard_path = _confined_path(self.storage_path, f"{shard_name}.npy", "data", "vectors")
        if shard_path.exists():
            return np.load(str(shard_path))
        return None

    def list_shards(self) -> List[str]:
        """Lista todos los shards disponibles."""
        vectors_dir = self.storage_path / "data" / "vectors"
        return [f.stem for f in vectors_dir.glob("*.npy")]

    # -------------------------------------------------------------------------
    # Metadata (SQLite)
    # -------------------------------------------------------------------------

    def save_metadata(
        self,
        doc_id: str,
        shard_idx: int,
        vector_idx: int,
        metadata: Optional[Dict] = None,
        document: Optional[str] = None,
    ):
        """Guarda metadata de un documento.

        Escritura por lotes (audit r3a-17 resto): antes cada llamada abría una
        conexión y hacía un commit (fsync) propio, así un add() de N documentos
        costaba N transacciones y eso dominaba el tiempo de escritura. Ahora la
        fila se acumula y TODO el lote del shard actual —las filas entregadas
        desde el último save_vectors()— se escribe en UNA transacción atómica
        dentro de la última save_metadata() del lote: al volver de add() el lote
        entero es durable y visible para cualquier otra conexión o instancia.
        Una llamada suelta (sin save_vectors() previo, o con el lote ya
        cerrado) commitea inmediatamente, como siempre. Si un lote queda
        incompleto, lo pendiente se flushea en la próxima operación de storage
        (_get_conn(), save_vectors(), checkpoint(), close(), backup()).
        """
        now = time.time()
        row = (doc_id, shard_idx, vector_idx, json.dumps(metadata or {}), document, now, now)
        with self._lock:
            self._pending_metadata.append(row)
            if self._batch_expected > 1:
                self._batch_expected -= 1
                return
            self._batch_expected = 0
            self._flush_pending_metadata()

    def save_metadata_many(self, rows: Iterable[Dict[str, Any]]) -> int:
        """Guarda muchas filas de metadata en UNA transacción (API explícita).

        Args:
            rows: dicts con ``id`` (o ``doc_id``), ``shard_idx``, ``vector_idx``
                y opcionalmente ``metadata`` / ``document``.

        Returns:
            Cantidad de filas escritas. Si la transacción falla no se escribe
            ninguna (atómico) y la excepción se propaga.
        """
        prepared = [self._metadata_row(r) for r in rows]
        with self._lock:
            self._flush_pending_metadata()  # lo entregado antes, primero
            self._write_metadata_rows(prepared)
        return len(prepared)

    @staticmethod
    def _metadata_row(r: Dict[str, Any]) -> Tuple:
        now = time.time()
        return (
            r.get("id", r.get("doc_id")),
            r.get("shard_idx"),
            r.get("vector_idx"),
            json.dumps(r.get("metadata") or {}),
            r.get("document"),
            now,
            now,
        )

    def _flush_pending_metadata(self) -> None:
        """Escribe en UNA transacción las filas de metadata pendientes."""
        with self._lock:
            if not self._pending_metadata:
                return
            # Si falla, las filas quedan pendientes (no se pierden) y la
            # excepción se propaga.
            self._write_metadata_rows(self._pending_metadata)
            self._pending_metadata = []
            self._batch_expected = 0

    def _write_metadata_rows(self, rows: List[Tuple]) -> None:
        """INSERT OR REPLACE de varias filas en una sola transacción SQLite."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO documents
                    (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """,
                rows,
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def update_metadata(self, doc_id: str, metadata: Dict = None, document: str = None):
        """Actualiza metadata de un documento existente."""
        conn = self._get_conn()
        try:
            if metadata is not None:
                conn.execute(
                    "UPDATE documents SET metadata=?, updated_at=? WHERE id=? AND deleted=0",
                    (json.dumps(metadata), time.time(), doc_id),
                )
            if document is not None:
                conn.execute(
                    "UPDATE documents SET document=?, updated_at=? WHERE id=? AND deleted=0",
                    (document, time.time(), doc_id),
                )
            conn.commit()
        finally:
            conn.close()

        if self.wal:
            self.wal.log_operation(
                "update_metadata",
                {"id": doc_id, "metadata": metadata, "document": document},
            )

    def update_vector(self, doc_id: str, vector: np.ndarray) -> bool:
        """Reescribe el vector de un documento ya persistido (audit r1-09).

        update_metadata() sólo toca metadata/document: sin esto, tras reload el doc
        volvía con la metadata NUEVA y el vector VIEJO.

        Estrategia:
          1. reescritura in-place de la fila (shard_idx, vector_idx) del shard
             original — el layout de shards no cambia;
          2. si esa fila no se puede ubicar (shard inexistente, índice fuera de
             rango), append a un shard nuevo y repuntar shard_idx/vector_idx.

        Returns:
            True si el vector quedó escrito; False si el doc no tiene fila en
            metadata.db (en ese caso no se escribe nada).
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        vectors_dir = self.storage_path / "data" / "vectors"

        # read-modify-write del shard: serializar entre updates_vector
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT shard_idx, vector_idx FROM documents WHERE id=?", (doc_id,)
                ).fetchone()
            finally:
                conn.close()
            if row is None:
                return False
            shard_idx, vector_idx = row

            shard_name = (
                f"shard_{max(1, int(shard_idx)):03d}" if shard_idx is not None else None
            )
            shard = self.load_vectors(shard_name) if shard_name else None
            in_place = (
                shard is not None
                and shard.ndim == 2
                and vector_idx is not None
                and 0 <= int(vector_idx) < len(shard)
            )
            if in_place:
                shard[int(vector_idx)] = vec
            else:
                # Append a un shard nuevo (nunca pisar uno existente) y repuntar la fila
                max_idx = 0
                for stem in self.list_shards():
                    try:
                        max_idx = max(max_idx, int(stem.rsplit("_", 1)[1]))
                    except (IndexError, ValueError):
                        continue
                shard_name = f"shard_{max_idx + 1:03d}"
                shard = vec[np.newaxis, :]
                shard_idx, vector_idx = max_idx + 1, 0

            np.save(str(vectors_dir / f"{shard_name}.npy"), np.ascontiguousarray(shard, dtype=np.float32))

            if not in_place:
                conn = self._get_conn()
                try:
                    conn.execute(
                        "UPDATE documents SET shard_idx=?, vector_idx=?, updated_at=? WHERE id=?",
                        (shard_idx, vector_idx, time.time(), doc_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

        if self.wal:
            self.wal.log_operation(
                "update_vector",
                {"id": doc_id, "shard": shard_name, "vector_idx": int(vector_idx)},
            )
        return True

    def get_metadata(self, doc_id: str) -> Optional[Dict]:
        """Obtiene metadata de un documento."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT metadata, document, shard_idx, vector_idx FROM documents WHERE id=? AND deleted=0",
                (doc_id,),
            ).fetchone()
            if row:
                return {
                    "metadata": json.loads(row[0]) if row[0] else {},
                    "document": row[1],
                    "shard_idx": row[2],
                    "vector_idx": row[3],
                }
        finally:
            conn.close()
        return None

    def _document_columns(self, conn: sqlite3.Connection) -> set:
        """Columnas de la tabla documents (esquema viejo sin `deleted` incluido)."""
        return {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}

    def get_all_documents(self, include_deleted: bool = True) -> List[Dict]:
        """Todas las filas de metadata.db en orden ESTABLE (shard_idx, vector_idx).

        Una sola conexión (get_metadata() abre una por doc) y un orden
        reproducible: es el orden de inserción de add() —cada lote escribe su
        propio shard y vector_idx es la posición dentro del lote—, que
        SimpleVectorDB._restore_from_storage usa para alinear _splat_id_order
        con las posiciones de splat (audit r1-01). En bases viejas sin la
        columna `deleted` se reporta como False.

        Returns:
            Lista de dicts {id, shard_idx, vector_idx, metadata, document, deleted}.
        """
        conn = self._get_conn()
        try:
            has_deleted = "deleted" in self._document_columns(conn)
            sql = "SELECT id, shard_idx, vector_idx, metadata, document"
            sql += ", deleted" if has_deleted else ""
            sql += " FROM documents"
            if not include_deleted and has_deleted:
                sql += " WHERE deleted=0"
            sql += " ORDER BY shard_idx, vector_idx, rowid"
            rows = conn.execute(sql).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": row[0],
                "shard_idx": row[1],
                "vector_idx": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "document": row[4],
                "deleted": bool(row[5]) if has_deleted else False,
            }
            for row in rows
        ]

    def _storage_digest(self) -> Dict[str, Any]:
        """Digest del contenido del storage: sha256 por shard + digest lógico de
        metadata.db (filas ordenadas por id, independiente del layout físico de
        SQLite). Base del manifiesto firmado (audit r3a-17)."""
        files = {}
        for shard_path in sorted((self.storage_path / "data" / "vectors").glob("*.npy")):
            h = hashlib.sha256()
            with open(str(shard_path), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            files[f"data/vectors/{shard_path.name}"] = h.hexdigest()
        conn = self._get_conn()
        try:
            has_deleted = "deleted" in self._document_columns(conn)
            sql = "SELECT id, shard_idx, vector_idx, metadata, document"
            sql += ", deleted" if has_deleted else ""
            rows = conn.execute(sql + " FROM documents ORDER BY id").fetchall()
        finally:
            conn.close()
        h = hashlib.sha256()
        for row in rows:
            canon = json.dumps(
                [None if v is None else str(v) for v in row],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            h.update(canon.encode("utf-8"))
            h.update(b"\n")
        return {"version": 1, "files": files, "metadata": h.hexdigest()}

    def write_manifest(self) -> Optional[str]:
        """Firma el storage (shards + metadata) con HMAC-SHA256.

        Escribe index/manifest.idx via save_index(); requiere M2M_HMAC_SECRET.
        Sin secreto no escribe nada ni levanta (firma opt-in, audit r3a-17).

        Returns:
            Ruta del manifiesto escrito, o None si no hay secreto.
        """
        if not os.environ.get("M2M_HMAC_SECRET"):
            return None
        self.save_index(self._storage_digest(), name=MANIFEST_NAME)
        return str(self.storage_path / "index" / f"{MANIFEST_NAME}.idx")

    def verify_manifest(self) -> bool:
        """Verifica la firma y el contenido del storage contra index/manifest.idx.

        - Sin M2M_HMAC_SECRET: no-op silencioso (devuelve True).
        - Con secreto y manifiesto: recomputa el digest y compara; un shard o
          metadata alterado, o el manifiesto mismo, levanta IndexTamperedError
          (RuntimeError).
        - Con secreto pero SIN manifiesto (storage escrito sin firma): warning y
          devuelve False; no bloquea la carga de bases preexistentes.
        """
        if not os.environ.get("M2M_HMAC_SECRET"):
            return True
        index_path = self.storage_path / "index" / f"{MANIFEST_NAME}.idx"
        if not index_path.exists():
            logger.warning(
                "M2M_HMAC_SECRET is set but %s has no signed manifest "
                "(index/%s.idx): skipping integrity verification",
                self.storage_path,
                MANIFEST_NAME,
            )
            return False
        try:
            expected = self.load_index(MANIFEST_NAME)
        except ValueError as e:  # HMAC del manifiesto (load_index)
            raise IndexTamperedError(str(e)) from e
        if (
            not isinstance(expected, dict)
            or not isinstance(expected.get("files"), dict)
            or "metadata" not in expected
        ):
            raise IndexTamperedError(f"Invalid signed manifest for {self.storage_path}")
        actual = self._storage_digest()
        exp_files, act_files = expected["files"], actual["files"]
        problems = []
        for rel in sorted(set(exp_files) | set(act_files)):
            if rel not in act_files:
                problems.append(f"{rel} (missing)")
            elif rel not in exp_files:
                problems.append(f"{rel} (not in manifest)")
            elif exp_files[rel] != act_files[rel]:
                problems.append(f"{rel} (content modified)")
        if expected["metadata"] != actual["metadata"]:
            problems.append("data/metadata/metadata.db (content modified)")
        if problems:
            raise IndexTamperedError(
                f"Integrity verification failed for storage {self.storage_path}: "
                f"possible tampering in {', '.join(problems)}"
            )
        return True

    def get_all_ids(self, include_deleted: bool = False) -> List[str]:
        """Obtiene todos los IDs almacenados."""
        conn = self._get_conn()
        try:
            if include_deleted:
                rows = conn.execute("SELECT id FROM documents").fetchall()
            else:
                rows = conn.execute("SELECT id FROM documents WHERE deleted=0").fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def soft_delete(self, doc_id: str) -> bool:
        """Marca un documento como eliminado (soft delete)."""
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE documents SET deleted=1, updated_at=? WHERE id=?",
                (time.time(), doc_id),
            )
            conn.commit()
            affected = cur.rowcount > 0
        finally:
            conn.close()

        if self.wal and affected:
            self.wal.log_operation("delete", {"id": doc_id, "hard": False})

        return affected

    def hard_delete(self, doc_id: str) -> bool:
        """Elimina permanentemente un documento."""
        conn = self._get_conn()
        try:
            cur = conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            conn.commit()
            affected = cur.rowcount > 0
        finally:
            conn.close()

        if self.wal and affected:
            self.wal.log_operation("delete", {"id": doc_id, "hard": True})

        return affected

    def filter_by_metadata(self, filter_dict: Dict) -> List[str]:
        """
        Filtra documentos por metadata.

        Soporta operadores: $eq, $gt, $gte, $lt, $lte, $ne
        """
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT id, metadata FROM documents WHERE deleted=0").fetchall()
        finally:
            conn.close()

        results = []
        for doc_id, meta_str in rows:
            meta = json.loads(meta_str) if meta_str else {}
            if self._matches_filter(meta, filter_dict):
                results.append(doc_id)
        return results

    def _matches_filter(self, meta: dict, filter_dict) -> bool:
        """Evalua si un documento cumple el filtro.

        Delega en ``m2m.match_metadata_filter``: MISMA semantica de operadores que
        ``SimpleVectorDB._match_filter`` ($eq,$ne,$gt,$gte,$lt,$lte,$in,$nin o
        callable metadata -> bool); operador desconocido / condicion ``{}`` /
        tipo invalido -> ValueError (fail closed) [audit r1-14]. Import diferido
        porque ``m2m/__init__`` importa este modulo al arrancar.
        """
        from .. import match_metadata_filter

        return match_metadata_filter(meta, filter_dict)

    # -------------------------------------------------------------------------
    # Index Persistence
    # -------------------------------------------------------------------------

    def save_index(self, index_data: Any, name: str = "hrm2"):
        """Guarda el índice serializado con firma HMAC (H-05 fix)."""
        index_path = _confined_path(self.storage_path, f"{name}.idx", "index")
        import hashlib
        import hmac
        import os

        secret = os.environ.get("M2M_HMAC_SECRET")
        if not secret:
            raise RuntimeError(
                "M2M_HMAC_SECRET environment variable required for secure deserialization"
            )
        data = pickle.dumps(index_data, protocol=pickle.HIGHEST_PROTOCOL)
        sig = hmac.new(secret.encode(), data, hashlib.sha256).digest()
        with open(str(index_path), "wb") as f:
            f.write(sig + data)

    def load_index(self, name: str = "hrm2") -> Optional[Any]:
        """Carga el índice desde disco con verificación HMAC (H-05 fix)."""
        import hashlib
        import hmac
        import os

        secret = os.environ.get("M2M_HMAC_SECRET")
        if not secret:
            raise RuntimeError(
                "M2M_HMAC_SECRET environment variable required for secure deserialization"
            )
        index_path = _confined_path(self.storage_path, f"{name}.idx", "index")
        if index_path.exists():
            with open(str(index_path), "rb") as f:
                raw = f.read()
            sig, data = raw[:32], raw[32:]
            expected_sig = hmac.new(secret.encode(), data, hashlib.sha256).digest()
            if not hmac.compare_digest(sig, expected_sig):
                raise IndexTamperedError(
                    f"HMAC verification failed for index {name}. Possible tampering."
                )
            import io
            return _RestrictedUnpickler(io.BytesIO(data)).load()
        return None

    # -------------------------------------------------------------------------
    # Energy State Persistence
    # -------------------------------------------------------------------------

    def save_energy_state(self, state: Dict):
        """Guarda el estado energético del sistema."""
        energy_path = self.storage_path / "energy" / "landscape.npz"
        saveable = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                saveable[k] = v
            else:
                saveable[k] = np.array([v])
        np.savez(str(energy_path), **saveable)

    def load_energy_state(self) -> Optional[Dict]:
        """Carga el estado energético."""
        energy_path = self.storage_path / "energy" / "landscape.npz"
        if energy_path.exists():
            data = np.load(str(energy_path))
            return dict(data)
        return None

    # -------------------------------------------------------------------------
    # Backup / Restore
    # -------------------------------------------------------------------------

    def backup(self, backup_path: str) -> str:
        """Crea un backup completo del storage."""
        import shutil

        self._flush_pending_metadata()  # el backup copia metadata.db: debe estar al día
        # -- P-04 FIX: Path traversal prevention -----------------------
        backup_dir = Path(backup_path).resolve()
        if ".." in Path(backup_path).parts:
            raise ValueError(f"Path traversal detected in backup_path: {backup_path}")

        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        dest = backup_dir / f"m2m_backup_{ts}"
        shutil.copytree(str(self.storage_path), str(dest))
        return str(dest)

    def restore(self, backup_path: str):
        """Restaura el storage desde un backup escrito por ``backup()``.

        La fuente se valida ANTES de tocar el storage (audit P15): antes
        ``rmtree(storage_path)`` corría primero, así que una fuente inexistente
        o que no es un directorio dejaba el storage BORRADO, y una fuente dentro
        del propio storage borraba además su propia copia. Se rechaza con:

        * ``ValueError``: ``..`` en la ruta (igual que ``backup()``/``load()``), o
          una fuente que al resolver queda igual, dentro o por encima del
          directorio de storage (incluye symlinks que apuntan adentro);
        * ``FileNotFoundError`` / ``NotADirectoryError``: la fuente no existe o
          no es un directorio.

        Las rutas absolutas y los symlinks a un backup EXTERNO al storage se
        aceptan (se resuelven con ``Path.resolve()``), igual que ``load()``.
        """
        import shutil

        src = Path(backup_path)
        if ".." in src.parts:
            raise ValueError(f"Path traversal detected in backup_path: {backup_path}")
        if not src.exists():
            raise FileNotFoundError(f"restore(): backup path does not exist: {src}")
        if not src.is_dir():
            raise NotADirectoryError(f"restore(): {src} is not a directory")
        src_resolved = src.resolve()
        dst_resolved = Path(self.storage_path).resolve()
        if (
            src_resolved == dst_resolved
            or dst_resolved in src_resolved.parents
            or src_resolved in dst_resolved.parents
        ):
            raise ValueError(
                "restore(): backup_path must not be the storage directory, inside it "
                f"or contain it (backup={src_resolved}, storage={dst_resolved})"
            )

        # El storage se reemplaza entero: las filas pendientes de un lote
        # incompleto corresponden al estado previo y se descartan.
        with self._lock:
            self._pending_metadata = []
            self._batch_expected = 0
        shutil.rmtree(str(self.storage_path), ignore_errors=True)
        shutil.copytree(backup_path, str(self.storage_path))
        self._init_metadata_db()

    # -------------------------------------------------------------------------
    # WAL Checkpoint
    # -------------------------------------------------------------------------

    def checkpoint(self):
        """Crea checkpoint del WAL."""
        self._flush_pending_metadata()  # checkpoint = todo durable
        if self.wal:
            self.wal.checkpoint()

    def recover_from_wal(self) -> List[Dict]:
        """
        Recupera operaciones del WAL después de un crash.

        Returns:
            Lista de operaciones que deben re-ejecutarse.
        """
        if not self.wal:
            return []
        entries = self.wal.recover()
        # Solo devolver operaciones (no checkpoints)
        return [
            {"operation": e.operation, "data": e.data, "lsn": e.lsn}
            for e in entries
            if e.operation != "checkpoint"
        ]

    def get_stats(self) -> Dict:
        """Estadísticas del storage."""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM documents WHERE deleted=0").fetchone()[0]
            deleted = conn.execute("SELECT COUNT(*) FROM documents WHERE deleted=1").fetchone()[0]
        finally:
            conn.close()

        shards = self.list_shards()
        return {
            "total_documents": total,
            "soft_deleted": deleted,
            "shards": len(shards),
            "storage_path": str(self.storage_path),
            "wal_enabled": self.wal is not None,
        }

    def close(self):
        """Cierra el storage flusheando el WAL."""
        self._flush_pending_metadata()
        if self.wal:
            self.wal.checkpoint()
            self.wal.close()
