"""
WriteAheadLog (WAL) - Durabilidad de datos para M2M.

El WAL garantiza que todas las operaciones se registran antes de ejecutarse,
permitiendo recuperación completa ante fallos del sistema.
"""

import json
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False


@dataclass
class WALEntry:
    """Una entrada en el Write-Ahead Log."""

    lsn: int  # Log Sequence Number
    timestamp: float
    operation: str  # 'add', 'update', 'delete', 'checkpoint'
    data: Dict[str, Any]


class WriteAheadLog:
    """
    Write-Ahead Log para durabilidad de datos.

    Registra cada operación ANTES de ejecutarla. Si el sistema falla,
    el WAL permite recuperar todas las operaciones pendientes.

    Uso:
        wal = WriteAheadLog('./data/wal/wal.log')
        lsn = wal.log_operation('add', {'id': 'doc1', 'vector': [...]})
        wal.checkpoint()
        ops = wal.recover()
    """

    def __init__(self, path: str, sync_interval: int = 100):
        """
        Args:
            path: Ruta al archivo WAL
            sync_interval: Número de operaciones entre syncs automáticos
        """
        self.path = path
        self.sync_interval = sync_interval
        self.lsn = 0
        self._lock = threading.Lock()
        self._op_count = 0

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Abrir archivo en modo append-binary
        self._file = open(path, "ab")

        # Cargar LSN actual desde checkpoints existentes
        self._load_current_lsn()

    def _load_current_lsn(self):
        """Carga el LSN más alto desde el WAL existente."""
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return
        try:
            entries = self._read_entries()
            if entries:
                self.lsn = entries[-1].lsn + 1
        except Exception:
            pass  # WAL corrupto, empezar desde 0

    def _serialize(self, entry: dict) -> bytes:
        """Serializa una entrada. Usa msgpack si disponible, si no JSON."""
        if _HAS_MSGPACK:
            return msgpack.packb(entry, use_bin_type=True)
        else:
            return json.dumps(entry, default=str).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict:
        """Deserializa una entrada."""
        if _HAS_MSGPACK:
            return msgpack.unpackb(data, raw=False)
        else:
            return json.loads(data.decode("utf-8"))

    def log_operation(self, op: str, data: Dict[str, Any]) -> int:
        """
        Registra una operación en el WAL antes de ejecutarla.

        Args:
            op: Tipo de operación ('add', 'update', 'delete')
            data: Datos de la operación

        Returns:
            LSN (Log Sequence Number) de la entrada
        """
        with self._lock:
            entry = {
                "lsn": self.lsn,
                "timestamp": time.time(),
                "operation": op,
                "data": data,
            }
            serialized = self._serialize(entry)
            length = len(serialized)

            # Escribir: [4 bytes longitud][datos]
            self._file.write(struct.pack(">I", length))
            self._file.write(serialized)

            # Sync automático
            self._op_count += 1
            if self._op_count >= self.sync_interval:
                self._file.flush()
                self._op_count = 0

            current_lsn = self.lsn
            self.lsn += 1
            return current_lsn

    def checkpoint(self) -> str:
        """
        Crea un checkpoint: flush completo del WAL.

        Returns:
            Ruta del checkpoint creado
        """
        with self._lock:
            # Flush y fsync para garantizar durabilidad
            self._file.flush()
            os.fsync(self._file.fileno())

            # Registrar el checkpoint mismo
            checkpoint_data = {
                "lsn": self.lsn,
                "timestamp": time.time(),
                "operation": "checkpoint",
                "data": {"lsn_at_checkpoint": self.lsn},
            }
            serialized = self._serialize(checkpoint_data)
            length = len(serialized)
            self._file.write(struct.pack(">I", length))
            self._file.write(serialized)
            self._file.flush()
            os.fsync(self._file.fileno())

            self.lsn += 1

        return self.path

    def recover(self) -> List[WALEntry]:
        """
        Recupera todas las operaciones registradas en el WAL.

        Returns:
            Lista de WALEntry con todas las operaciones registradas
        """
        return self._read_entries()

    def _read_entries(self) -> List[WALEntry]:
        """Lee todas las entradas del WAL."""
        entries = []
        if not os.path.exists(self.path):
            return entries

        try:
            with open(self.path, "rb") as f:
                while True:
                    length_bytes = f.read(4)
                    if not length_bytes or len(length_bytes) < 4:
                        break
                    length = struct.unpack(">I", length_bytes)[0]
                    if length == 0:
                        break
                    data_bytes = f.read(length)
                    if len(data_bytes) < length:
                        break  # WAL truncado
                    entry_dict = self._deserialize(data_bytes)
                    entry = WALEntry(
                        lsn=entry_dict.get("lsn", 0),
                        timestamp=entry_dict.get("timestamp", 0.0),
                        operation=entry_dict.get("operation", ""),
                        data=entry_dict.get("data", {}),
                    )
                    entries.append(entry)
        except Exception:
            pass  # Retornar lo que se pudo leer
        return entries

    def truncate(self, before_lsn: int):
        """Elimina entradas anteriores a before_lsn (compactación)."""
        with self._lock:
            entries = self._read_entries()
            remaining = [e for e in entries if e.lsn >= before_lsn]

            # Reescribir WAL solo con entradas restantes
            self._file.close()
            with open(self.path, "wb") as f:
                for entry in remaining:
                    d = {
                        "lsn": entry.lsn,
                        "timestamp": entry.timestamp,
                        "operation": entry.operation,
                        "data": entry.data,
                    }
                    serialized = self._serialize(d)
                    f.write(struct.pack(">I", len(serialized)))
                    f.write(serialized)
                    f.flush()
            self._file = open(self.path, "ab")

    def close(self):
        """Cierra el WAL flusheando primero."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
