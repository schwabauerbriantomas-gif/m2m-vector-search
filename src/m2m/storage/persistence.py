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

import json
import os
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .wal import WriteAheadLog


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
        self.storage_path = Path(storage_path)
        self.enable_wal = enable_wal
        self._lock = threading.Lock()

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
        conn.execute(
            """
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
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_deleted ON documents(deleted)"
        )
        conn.commit()
        conn.close()
        self._db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        """Obtiene conexión SQLite thread-safe."""
        return sqlite3.connect(self._db_path, check_same_thread=False)

    # -------------------------------------------------------------------------
    # Vector Storage (numpy shards)
    # -------------------------------------------------------------------------

    def save_vectors(
        self, vectors: np.ndarray, ids: List[str], shard_name: str = "shard_001"
    ):
        """Guarda vectores en un archivo numpy shard."""
        shard_path = self.storage_path / "data" / "vectors" / f"{shard_name}.npy"
        np.save(str(shard_path), vectors.astype(np.float32))

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

    def load_vectors(self, shard_name: str = "shard_001") -> Optional[np.ndarray]:
        """Carga vectores de un shard."""
        shard_path = self.storage_path / "data" / "vectors" / f"{shard_name}.npy"
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
        """Guarda metadata de un documento."""
        now = time.time()
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """,
                (
                    doc_id,
                    shard_idx,
                    vector_idx,
                    json.dumps(metadata or {}),
                    document,
                    now,
                    now,
                ),
            )
            conn.commit()
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

    def get_all_ids(self, include_deleted: bool = False) -> List[str]:
        """Obtiene todos los IDs almacenados."""
        conn = self._get_conn()
        try:
            if include_deleted:
                rows = conn.execute("SELECT id FROM documents").fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM documents WHERE deleted=0"
                ).fetchall()
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
            rows = conn.execute(
                "SELECT id, metadata FROM documents WHERE deleted=0"
            ).fetchall()
        finally:
            conn.close()

        results = []
        for doc_id, meta_str in rows:
            meta = json.loads(meta_str) if meta_str else {}
            if self._matches_filter(meta, filter_dict):
                results.append(doc_id)
        return results

    def _matches_filter(self, meta: dict, filter_dict: dict) -> bool:
        """Evalúa si un documento cumple el filtro."""
        for key, condition in filter_dict.items():
            if key not in meta:
                return False
            val = meta[key]
            if isinstance(condition, dict):
                for op, comp in condition.items():
                    if op == "$eq" and val != comp:
                        return False
                    elif op == "$ne" and val == comp:
                        return False
                    elif op == "$gt" and not (val > comp):
                        return False
                    elif op == "$gte" and not (val >= comp):
                        return False
                    elif op == "$lt" and not (val < comp):
                        return False
                    elif op == "$lte" and not (val <= comp):
                        return False
            else:
                if val != condition:
                    return False
        return True

    # -------------------------------------------------------------------------
    # Index Persistence
    # -------------------------------------------------------------------------

    def save_index(self, index_data: Any, name: str = "hrm2"):
        """Guarda el índice serializado."""
        index_path = self.storage_path / "index" / f"{name}.idx"
        with open(str(index_path), "wb") as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, name: str = "hrm2") -> Optional[Any]:
        """Carga el índice desde disco."""
        index_path = self.storage_path / "index" / f"{name}.idx"
        if index_path.exists():
            with open(str(index_path), "rb") as f:
                return pickle.load(f)
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

        backup_dir = Path(backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        dest = backup_dir / f"m2m_backup_{ts}"
        shutil.copytree(str(self.storage_path), str(dest))
        return str(dest)

    def restore(self, backup_path: str):
        """Restaura desde un backup."""
        import shutil

        shutil.rmtree(str(self.storage_path), ignore_errors=True)
        shutil.copytree(backup_path, str(self.storage_path))
        self._init_metadata_db()

    # -------------------------------------------------------------------------
    # WAL Checkpoint
    # -------------------------------------------------------------------------

    def checkpoint(self):
        """Crea checkpoint del WAL."""
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
            total = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE deleted=0"
            ).fetchone()[0]
            deleted = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE deleted=1"
            ).fetchone()[0]
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
        if self.wal:
            self.wal.checkpoint()
            self.wal.close()
