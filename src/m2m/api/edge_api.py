"""
M2M REST API - Servidor completo con colecciones, CRUD, búsqueda y EBM features.

Endpoints:
  GET  /v1/health
  GET  /v1/stats
  GET  /v1/collections
  POST /v1/collections
  GET  /v1/collections/{name}
  DELETE /v1/collections/{name}
  POST /v1/collections/{name}/vectors
  GET  /v1/collections/{name}/vectors/{id}
  PUT  /v1/collections/{name}/vectors/{id}
  DELETE /v1/collections/{name}/vectors/{id}
  POST /v1/collections/{name}/search
  POST /v1/collections/{name}/query
  POST /v1/collections/{name}/energy
  POST /v1/collections/{name}/explore
  GET  /v1/collections/{name}/suggest
  GET  /v1/collections/{name}/stats
  POST /v1/admin/checkpoint
  POST /v1/admin/backup
"""

import hmac
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator

from .. import AdvancedVectorDB, SimpleVectorDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int
    mode: str = "standard"  # 'edge', 'standard', 'ebm'
    enable_ebm: bool = False
    storage_path: Optional[str] = None
    metadata_schema: Optional[Dict[str, str]] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not _COLLECTION_NAME_RE.match(v):
            raise ValueError("Invalid collection name")
        return v

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        if v < 1 or v > 65536:
            raise ValueError("dimension must be between 1 and 65536")
        return v


class InsertVectorsRequest(BaseModel):
    vectors: List[List[float]]
    ids: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    documents: Optional[List[str]] = None

    @field_validator("vectors")
    @classmethod
    def validate_vectors(cls, v: list) -> list:
        if len(v) > _RATE_LIMIT_MAX_VECTORS_PER_INSERT:
            raise ValueError(f"Max {_RATE_LIMIT_MAX_VECTORS_PER_INSERT} vectors per request")
        return v


class UpdateVectorRequest(BaseModel):
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    document: Optional[str] = None
    upsert: bool = False


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    include_metadata: bool = True
    include_documents: bool = False
    include_energy: bool = False
    filter: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError("k must be between 1 and 10000")
        return v

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: list) -> list:
        if not v:
            raise ValueError("vector must not be empty")
        if any(not isinstance(x, (int, float)) or x != x for x in v):  # NaN check
            raise ValueError("vector contains NaN or non-numeric values")
        if any(abs(x) > 1e38 for x in v):  # Near-overflow check (P-01 fix)
            raise ValueError("vector contains values too large (possible overflow)")
        return v


class EnergyRequest(BaseModel):
    vector: Optional[List[float]] = None
    center: Optional[List[float]] = None
    radius: float = 1.0
    resolution: int = 20

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v < 1 or v > 100:  # P-16 fix: cap resolution at 100
            raise ValueError("resolution must be between 1 and 100")
        return v

    @field_validator("radius")
    @classmethod
    def validate_radius(cls, v: float) -> float:
        if v <= 0 or v > 1e6:
            raise ValueError("radius must be between 0 and 1e6")
        return v


class ExploreRequest(BaseModel):
    topic_vector: Optional[List[float]] = None
    n_suggestions: int = 3
    min_energy: float = 0.7

    @field_validator("n_suggestions")
    @classmethod
    def validate_n_suggestions(cls, v: int) -> int:
        if v < 1 or v > 50:  # P-18 fix: cap at 50
            raise ValueError("n_suggestions must be between 1 and 50")
        return v


class BackupRequest(BaseModel):
    backup_path: str

    @field_validator("backup_path")
    @classmethod
    def validate_backup_path(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        return v


# ---------------------------------------------------------------------------
# Collection Manager (in-memory)
# ---------------------------------------------------------------------------


class CollectionManager:
    """Gestiona colecciones en memoria."""

    def __init__(self):
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._dbs: Dict[str, SimpleVectorDB] = {}

    def create(self, req: CreateCollectionRequest) -> Dict:
        """Crea una nueva colección."""
        if req.name in self._collections:
            raise ValueError(f"Collection '{req.name}' already exists")

        if req.enable_ebm or req.mode == "ebm":
            db = AdvancedVectorDB(
                latent_dim=req.dimension,
                storage_path=req.storage_path,
                enable_energy_features=True,
                enable_soc=True,
            )
        else:
            db = SimpleVectorDB(
                latent_dim=req.dimension,
                storage_path=req.storage_path,
                enable_ebm=False,
                mode=req.mode,
            )

        self._dbs[req.name] = db
        self._collections[req.name] = {
            "name": req.name,
            "dimension": req.dimension,
            "mode": req.mode,
            "enable_ebm": req.enable_ebm,
            "created_at": time.time(),
            "vector_count": 0,
        }
        return self._collections[req.name]

    def get(self, name: str) -> SimpleVectorDB:
        if name not in self._dbs:
            raise KeyError(f"Collection '{name}' not found")
        return self._dbs[name]

    def info(self, name: str) -> Dict:
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' not found")
        return self._collections[name]

    def delete(self, name: str):
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' not found")
        del self._collections[name]
        del self._dbs[name]

    def list_names(self) -> List[str]:
        return list(self._collections.keys())

    def update_count(self, name: str, delta: int):
        if name in self._collections:
            self._collections[name]["vector_count"] += delta


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# -- SECURITY: Request body size limit (audit r3a-17 resto) ---------------
# Un body mayor a M2M_MAX_BODY_BYTES (default 32 MiB) se rechaza con HTTP 413
# ANTES de parsearse: con Content-Length, sin leer ni un byte; sin
# Content-Length (chunked) la lectura se corta al superar el tope. Lo implementa
# el max_body_size nativo de FastAPI/Starlette; se lee al construir la app.
_DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024


def _max_body_bytes() -> int:
    raw = os.environ.get("M2M_MAX_BODY_BYTES")
    if raw is None:
        return _DEFAULT_MAX_BODY_BYTES
    try:
        limit = int(raw)
    except ValueError:
        limit = -1
    if limit < 0:
        logger.warning(
            "M2M_MAX_BODY_BYTES=%r is not a non-negative integer; using %d",
            raw,
            _DEFAULT_MAX_BODY_BYTES,
        )
        return _DEFAULT_MAX_BODY_BYTES
    return limit


def _rate_limit_per_min() -> int:
    """Tope de pedidos por minuto e IP: ``M2M_RATE_LIMIT_PER_MIN``.

    Opt-in REAL (audit P15): ausente, vacío, 0 o negativo = límite APAGADO (el
    default). Un valor que no es entero apaga el límite y lo avisa (no adivina
    un tope). Se lee en cada pedido, igual que ``M2M_API_KEY``.
    """
    raw = os.environ.get("M2M_RATE_LIMIT_PER_MIN")
    if raw is None or not raw.strip():
        return 0
    try:
        limit = int(raw.strip())
    except ValueError:
        logger.warning(
            "M2M_RATE_LIMIT_PER_MIN=%r is not an integer; rate limiting is OFF", raw
        )
        return 0
    return max(0, limit)


app = FastAPI(
    title="M2M EBM Vector Database API",
    description="REST API para M2M Vector Search con features EBM",
    version="2.0.0",
    max_body_size=_max_body_bytes(),
)

# -- SECURITY: CORS configuration (M-01 fix) ----------------------------
# allow_origins=["*"] + allow_credentials=True is invalid per CORS spec.
# Default: permissive for development. Override via env for production.
_cors_origins = os.environ.get("M2M_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],  # False when wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- SECURITY: Rate limiting (H-03, P-07 fix) --------------------------
# Opt-in REAL (audit P15): el tope es ``M2M_RATE_LIMIT_PER_MIN`` = pedidos por
# minuto e IP. Ausente, vacío, 0 o negativo (default) = límite APAGADO. Antes el
# chequeo comparaba contra una constante de 100 sin registrar nunca los pedidos,
# así que jamás devolvía 429.
_rate_limit_store: Dict[str, List[float]] = {}
_RATE_LIMIT_WINDOW = 60.0  # seconds
# Valor de REFERENCIA (el recomendado para M2M_RATE_LIMIT_PER_MIN); ya no se consulta.
_RATE_LIMIT_MAX_REQUESTS = 100
_RATE_LIMIT_MAX_VECTORS_PER_INSERT = 100_000
# Memoria acotada (audit r3a-17 resto): a lo sumo _RATE_LIMIT_MAX_CLIENTS IPs en
# memoria. Las IPs sin pedidos dentro de la ventana se purgan en un barrido
# amortizado (a lo sumo uno cada _RATE_LIMIT_SWEEP_INTERVAL s, además del prune
# por-IP); si aun con el barrido se supera el tope, se evicta la IP con la
# actividad más vieja. La ventana (60 s) no cambia; el tope de pedidos por IP lo
# fija ``M2M_RATE_LIMIT_PER_MIN`` (ver ``_rate_limit_per_min``).
_RATE_LIMIT_MAX_CLIENTS = 10_000
_RATE_LIMIT_SWEEP_INTERVAL = 5.0
_rate_limit_last_sweep = 0.0
_rate_limit_lock = threading.Lock()


def _trim_rate_limit_store(now: float) -> None:
    """Recorta la tabla a ``_RATE_LIMIT_MAX_CLIENTS`` (evicta la IP con actividad más vieja)."""
    store = _rate_limit_store
    overflow = len(store) - _RATE_LIMIT_MAX_CLIENTS
    if overflow > 0:
        # Una entrada sin timestamps registrados cuenta como recién tocada (ahora).
        oldest_first = sorted(store, key=lambda ip: store[ip][-1] if store[ip] else now)
        for ip in oldest_first[:overflow]:
            store.pop(ip, None)


def _sweep_rate_limit_store(now: float) -> None:
    """Purga las IPs sin pedidos dentro de la ventana y recorta la tabla al tope."""
    store = _rate_limit_store
    for ip in [ip for ip, ts in store.items() if not ts or now - ts[-1] >= _RATE_LIMIT_WINDOW]:
        store.pop(ip, None)
    _trim_rate_limit_store(now)


def _check_rate_limit(client_ip: str, now: Optional[float] = None) -> None:
    """Rate limit por IP, opt-in vía ``M2M_RATE_LIMIT_PER_MIN`` (audit P15).

    Registra el pedido de ``client_ip`` en la ventana corrediza y levanta
    ``HTTPException(429)`` cuando la IP ya hizo ``limit`` pedidos dentro de la
    ventana. Antes comparaba contra ``_RATE_LIMIT_MAX_REQUESTS`` sin registrar
    nunca los pedidos: el tope jamás se disparaba.

    Con el límite APAGADO (default) no cuenta ni rechaza; sólo mantiene la tabla
    (purga por ventana + tope de IPs), así la memoria queda acotada aunque
    ``M2M_RATE_LIMIT_PER_MIN`` se active o desactive en caliente. Los pedidos
    rechazados no se registran (no alargan el bloqueo). Lo cuentan los endpoints
    que lo llaman: escrituras, búsqueda y los legacy; ``/health``, ``/v1/stats``
    y los GET no.
    """
    global _rate_limit_last_sweep

    if now is None:
        now = time.time()
    limit = _rate_limit_per_min()
    with _rate_limit_lock:
        if now - _rate_limit_last_sweep >= _RATE_LIMIT_SWEEP_INTERVAL:
            _sweep_rate_limit_store(now)
            _rate_limit_last_sweep = now
        if client_ip not in _rate_limit_store:
            _rate_limit_store[client_ip] = []
            _trim_rate_limit_store(now)
        # Prune old entries
        _rate_limit_store[client_ip] = [
            t for t in _rate_limit_store[client_ip] if now - t < _RATE_LIMIT_WINDOW
        ]
        if limit <= 0:
            return  # límite apagado
        if len(_rate_limit_store[client_ip]) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        _rate_limit_store[client_ip].append(now)


# -- SECURITY: API Key validation (C-01, P-06 fix) ---------------------
# Modelo de autenticación (audit r3a-17 resto):
#   * SIN M2M_API_KEY (default): la API queda ABIERTA en los endpoints normales
#     (desarrollo local / Quick Start). Decisión conservada a propósito.
#   * CON M2M_API_KEY: los endpoints que validan exigen ``Authorization: Bearer
#     <key>`` o ``X-API-Key: <key>``, comparados en tiempo constante.
#   * ``/v1/admin/*`` (checkpoint, backup) EXIGEN key: sin key configurada
#     responden 503 (fail closed) en vez de quedar abiertos.
#   * GET/PUT/DELETE de un vector también validan (audit P15): antes, con clave
#     configurada, se leía, sobreescribía y borraba sin ella. Siguen SIN validar
#     (sólo lectura de nombres/contadores): ``GET /v1/collections``,
#     ``GET /v1/collections/{name}`` y ``GET .../stats``.
_API_KEY = None  # Set via M2M_API_KEY env var


def _configured_api_key() -> Optional[str]:
    """Key efectiva: la global ``_API_KEY`` si se seteó a mano; si no, ``M2M_API_KEY``."""
    return _API_KEY if _API_KEY is not None else os.environ.get("M2M_API_KEY")


def _extract_token(request: Request) -> str:
    """Token del pedido: ``Authorization: Bearer <tok>`` (scheme case-insensitive,
    RFC 7235) o el header ``X-API-Key``. Otro scheme de Authorization no es una key."""
    auth = request.headers.get("Authorization", "")
    scheme, _, credential = auth.partition(" ")
    if scheme.strip().lower() == "bearer":
        return credential.strip()
    return request.headers.get("X-API-Key", "").strip()


def _validate_api_key(request: Request, admin: bool = False) -> None:
    """Valida la API key si está configurada (M2M_API_KEY o ``edge_api._API_KEY``).

    La comparación es en tiempo constante (``hmac.compare_digest``). Con
    ``admin=True`` el endpoint es de administración y falla cerrado: sin key
    configurada responde 503 en vez de quedar abierto.
    """
    expected = _configured_api_key()
    if expected is None:
        if admin:
            raise HTTPException(
                status_code=503,
                detail="Admin endpoints require an API key: M2M_API_KEY is not configured",
            )
        return
    token = _extract_token(request)
    if not token or not hmac.compare_digest(
        token.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# -- SECURITY: Collection name sanitization (H-01, P-03 fix) -----------
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _validate_collection_name(name: str) -> str:
    """Validates and sanitizes collection name to prevent path traversal."""
    if not _COLLECTION_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Invalid collection name. Use only alphanumeric, underscore, hyphen. Max 128 chars.",
        )
    return name


# -- SECURITY: Backup path sanitization (P-04 fix) --------------------
def _validate_backup_path(path: str) -> str:
    """Prevents path traversal in backup operations."""
    from pathlib import Path

    resolved = Path(path).resolve()
    # Allow relative paths but block traversal above CWD-like patterns
    if ".." in str(path):
        raise HTTPException(status_code=400, detail="Path traversal not allowed in backup_path")
    return str(resolved)


# -- SECURITY: Global error handler (H-04 fix) ------------------------
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    """Prevents internal error details from leaking to clients."""
    import traceback

    traceback.print_exc()  # Log internally
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


_manager = CollectionManager()

# ---------------------------------------------------------------------------
# Health & Stats
# ---------------------------------------------------------------------------


@app.get("/v1/health")
def health(request: Request):
    """Health check del servidor."""
    return {"status": "ok", "version": "2.0.0", "timestamp": time.time()}


@app.get("/v1/stats")
def stats(request: Request):
    """Estadísticas generales del servidor."""
    _validate_api_key(request)
    collections = []
    for name in _manager.list_names():
        try:
            db = _manager.get(name)
            s = db.get_stats()
            collections.append({"name": name, **s})
        except Exception as e:
            # Antes: `pass` → la colección desaparecía del informe sin rastro.
            collections.append({"name": name, "error": str(e)})
    return {
        "collections_count": len(_manager.list_names()),
        "collections": collections,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Collections CRUD
# ---------------------------------------------------------------------------


@app.get("/v1/collections")
def list_collections():
    """Lista todas las colecciones."""
    return {"collections": _manager.list_names()}


@app.post("/v1/collections", status_code=201)
def create_collection(req: CreateCollectionRequest, request: Request):
    """Crea una nueva colección."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    try:
        info = _manager.create(req)
        return {"message": "Collection created", "collection": info}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/v1/collections/{name}")
def get_collection(name: str):
    """Información de una colección."""
    try:
        return _manager.info(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/v1/collections/{name}")
def delete_collection(name: str, request: Request):
    """Elimina una colección."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    _validate_collection_name(name)
    try:
        _manager.delete(name)
        return {"message": f"Collection '{name}' deleted"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Vector CRUD
# ---------------------------------------------------------------------------


async def _parse_insert_request(request: Request) -> InsertVectorsRequest:
    """Body del insert en dos pasos: JSON → CONTEO de vectores → validación.

    El tope de vectores por request se chequea por conteo sobre el JSON ya
    decodificado y ANTES de que pydantic convierta/valide los arrays (audit
    r3a-17 resto): antes el validador del tope corría último, así 100.001
    vectores quemaban segundos de CPU en validación para terminar en 422.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        # Mismo 422 ("json_invalid") que produce FastAPI para JSON inválido.
        raise RequestValidationError(
            [
                {
                    "type": "json_invalid",
                    "loc": ("body", e.pos),
                    "msg": "JSON decode error",
                    "input": {},
                    "ctx": {"error": e.msg},
                }
            ],
            body=e.doc,
        ) from e
    vectors = body.get("vectors") if isinstance(body, dict) else None
    if isinstance(vectors, list) and len(vectors) > _RATE_LIMIT_MAX_VECTORS_PER_INSERT:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "vectors"],
                    "msg": f"Value error, Max {_RATE_LIMIT_MAX_VECTORS_PER_INSERT} vectors per request",
                    "type": "value_error",
                    "input": {"count": len(vectors)},
                }
            ],
        )
    try:
        return InsertVectorsRequest.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=jsonable_encoder(e.errors())) from e


@app.post("/v1/collections/{name}/vectors")
def insert_vectors(
    name: str, request: Request, req: InsertVectorsRequest = Depends(_parse_insert_request)
):
    """Inserta vectores en una colección."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    _validate_collection_name(name)
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    vectors = np.array(req.vectors, dtype=np.float32)
    try:
        added = db.add(
            ids=req.ids,
            vectors=vectors,
            metadata=req.metadata,
            documents=req.documents,
        )
    except ValueError as e:
        # Payload inválido (NaN/Inf, norma ~0, ids inválidos) → 400, no 500.
        raise HTTPException(status_code=400, detail=str(e))
    _manager.update_count(name, added)
    return {"added": added, "collection": name}


@app.get("/v1/collections/{name}/vectors/{vector_id}")
def get_vector(name: str, vector_id: str, request: Request):
    """Obtiene un vector por ID (valida la API key si está configurada [audit P15])."""
    _validate_api_key(request)  # antes del lookup: sin clave no se filtra 401 vs 404
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    vec = db._vectors.get(vector_id)
    if vec is None or vector_id in db._deleted:
        raise HTTPException(status_code=404, detail=f"Vector '{vector_id}' not found")

    return {
        "id": vector_id,
        "vector": vec.tolist(),
        "metadata": db._metadata.get(vector_id, {}),
        "document": db._documents.get(vector_id),
    }


@app.put("/v1/collections/{name}/vectors/{vector_id}")
def update_vector(
    name: str, vector_id: str, req: UpdateVectorRequest, request: Request
):
    """Actualiza un vector existente (valida la API key si está configurada [audit P15])."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    vector = np.array(req.vector, dtype=np.float32) if req.vector else None
    try:
        result = db.update(
            id=vector_id,
            vector=vector,
            metadata=req.metadata,
            document=req.document,
            upsert=req.upsert,
        )
    except ValueError as e:
        # Vector inválido (NaN/Inf, norma ~0, upsert sin vector) → 400, no 500.
        raise HTTPException(status_code=400, detail=str(e))

    if not result.success:
        raise HTTPException(status_code=404, detail=result.message)

    return {
        "success": result.success,
        "energy_delta": result.energy_delta,
        "message": result.message,
    }


@app.delete("/v1/collections/{name}/vectors/{vector_id}")
def delete_vector(name: str, vector_id: str, request: Request, hard: bool = False):
    """Elimina un vector (valida la API key si está configurada [audit P15])."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = db.delete(id=vector_id, hard=hard)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if result.deleted == 0:
        raise HTTPException(status_code=404, detail=f"Vector '{vector_id}' not found")

    _manager.update_count(name, -result.deleted)
    return {"deleted": result.deleted, "energy_freed": result.energy_freed}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@app.post("/v1/collections/{name}/search")
def search(name: str, req: SearchRequest, request: Request):
    """Búsqueda de similitud en la colección."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    _validate_collection_name(name)
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    query = np.array(req.vector, dtype=np.float32)
    start = time.time()

    results_raw = db.search(
        query,
        k=req.k,
        filter=req.filter,
        include_energy=req.include_energy,
        include_metadata=req.include_metadata,
    )

    elapsed_ms = (time.time() - start) * 1000

    if isinstance(results_raw, tuple):
        # Legacy format: (mu, alpha, kappa) o (mu, alpha, kappa, splat_indices)
        # T2: el engine HRM2 devuelve 4-tuple; unpack tolerante a ambos.
        if len(results_raw) == 4:
            mu, alpha, kappa, _ = results_raw
        else:
            mu, alpha, kappa = results_raw
        # T2: el engine HRM2 devuelve mu/alpha con batch dim (1, k); el path
        # LSH (k,). Aplanar el batch dim para el scoring legacy.
        alpha = np.asarray(alpha).reshape(-1)[: len(mu) if mu.ndim < 2 else mu.shape[-2]] if hasattr(alpha, "reshape") else alpha
        results = [{"id": f"idx_{i}", "score": float(alpha[i])} for i in range(min(req.k, len(alpha)))]
    else:
        results = []
        for r in results_raw:
            item: Dict[str, Any] = {"id": r.id, "score": r.score}
            if req.include_metadata:
                item["metadata"] = r.metadata
            if req.include_documents and r.document:
                item["document"] = r.document
            if req.include_energy and r.energy is not None:
                item["energy"] = r.energy
                item["confidence"] = r.confidence
            results.append(item)

    return {
        "results": results,
        "search_time_ms": elapsed_ms,
        "count": len(results),
    }


@app.post("/v1/collections/{name}/query")
def query_advanced(name: str, req: SearchRequest, request: Request):
    """Query avanzada con soporte de energía y exploración."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    _validate_collection_name(name)
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    query = np.array(req.vector, dtype=np.float32)

    if db.ebm_enabled and req.include_energy:
        sr = db.search_with_energy(query, k=req.k)
        results = []
        for r in sr.results:
            item: Dict[str, Any] = {
                "id": r.id,
                "score": r.score,
                "energy": r.energy,
                "confidence": r.confidence,
            }
            if req.include_metadata:
                item["metadata"] = r.metadata
            if req.include_documents and r.document:
                item["document"] = r.document
            results.append(item)
        return {
            "results": results,
            "query_energy": sr.query_energy,
            "total_confidence": sr.total_confidence,
            "search_time_ms": sr.search_time_ms,
            "uncertainty_regions": len(sr.uncertainty_regions),
        }
    else:
        return search(name, req)


# ---------------------------------------------------------------------------
# EBM Features
# ---------------------------------------------------------------------------


@app.post("/v1/collections/{name}/energy")
def compute_energy(name: str, req: EnergyRequest, request: Request):
    """Calcula la energía de un vector en el paisaje energético."""
    _validate_api_key(request)
    _validate_collection_name(name)
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not db.ebm_enabled:
        raise HTTPException(
            status_code=400,
            detail="EBM not enabled for this collection. Create with enable_ebm=True.",
        )

    if req.vector is not None:
        vec = np.array(req.vector, dtype=np.float32)
        energy_val = db.get_energy(vec)
        confidence = 1.0 / (1.0 + energy_val)
        return {
            "energy": energy_val,
            "confidence": confidence,
            "zone": db._ebm_energy.classify_energy(energy_val),
        }
    elif req.center is not None:
        center = np.array(req.center, dtype=np.float32)
        X, Y, energy_map = db._ebm_energy.local_energy_map(
            center, radius=req.radius, resolution=req.resolution
        )
        return {
            "energy_map": energy_map.tolist(),
            "x_grid": X.tolist(),
            "y_grid": Y.tolist(),
            "radius": req.radius,
        }
    else:
        raise HTTPException(status_code=400, detail="Provide 'vector' or 'center'")


@app.post("/v1/collections/{name}/explore")
def explore(name: str, req: ExploreRequest, request: Request):
    """Explora regiones de alta incertidumbre."""
    _validate_api_key(request)
    _validate_collection_name(name)
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not db.ebm_enabled:
        raise HTTPException(status_code=400, detail="EBM not enabled for this collection.")

    suggestions = db.suggest_exploration(n=req.n_suggestions)

    return {
        "suggestions": [
            {
                "description": s.description,
                "potential_value": s.potential_value,
                "suggested_queries": s.suggested_queries,
                "region_energy": s.region.energy,
            }
            for s in suggestions
        ],
        "count": len(suggestions),
    }


@app.get("/v1/collections/{name}/suggest")
def suggest_exploration(name: str, request: Request, n: int = 3):
    """Sugerencias de exploración (GET)."""
    req = ExploreRequest()
    req.n_suggestions = n
    # El Request viaja a explore(): con key configurada validaba ``request=None``
    # y devolvía 500 (AttributeError) en vez de 401/200.
    return explore(name, req, request)


# ---------------------------------------------------------------------------
# Collection Stats
# ---------------------------------------------------------------------------


@app.get("/v1/collections/{name}/stats")
def collection_stats(name: str):
    """Estadísticas de una colección."""
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return db.get_stats()


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


@app.post("/v1/admin/checkpoint")
def checkpoint(request: Request, name: Optional[str] = None):
    """Crea checkpoint (flush + fsync del WAL) de una colección o de todas.

    No silencia errores (audit r3a-17 resto). ``name`` inexistente → 404. Una
    colección sin ``storage_path`` pedida por nombre → 400 (no hay WAL que
    flushear); en el barrido de todas se informa como ``skipped``. Un fallo real
    de ``db.save()`` → 500 con el mensaje y el resultado por colección (antes:
    ``except Exception: pass`` y un "Checkpoint created" falso).
    """
    _validate_api_key(request, admin=True)
    if name is not None and name not in _manager._collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    names = [name] if name else _manager.list_names()
    results: Dict[str, Dict[str, str]] = {}
    failed = []
    for n in names:
        db = _manager.get(n)
        if not db.storage:
            if name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection '{n}' has no storage_path: nothing to checkpoint",
                )
            results[n] = {"status": "skipped", "reason": "no storage configured"}
            continue
        try:
            path = db.save()
        except Exception as e:
            results[n] = {"status": "error", "reason": str(e)}
            failed.append(f"{n}: {e}")
        else:
            results[n] = {"status": "ok", "path": str(path)}
    if failed:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Checkpoint failed for {len(failed)} collection(s): " + "; ".join(failed),
                "collections": names,
                "results": results,
            },
        )
    return {"message": "Checkpoint created", "collections": names, "results": results}


@app.post("/v1/admin/backup")
def backup(req: BackupRequest, request: Request):
    """Crea backup de todas las colecciones con storage.

    Sin ``storage_path`` → ``skipped``. ``ValueError`` (entrada inválida, p.ej.
    path traversal) → 400. Cualquier otro fallo → 500 con el motivo por
    colección (antes: HTTP 200 con el error escondido adentro del body).
    """
    _validate_api_key(request, admin=True)
    results = {}
    failed = []
    for name in _manager.list_names():
        try:
            db = _manager.get(name)
            if db.storage:
                path = db.storage.backup(req.backup_path)
                results[name] = {"status": "ok", "path": path}
            else:
                results[name] = {"status": "skipped", "reason": "no storage configured"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{name}: {e}")
        except Exception as e:
            results[name] = {"status": "error", "reason": str(e)}
            failed.append(name)
    if failed:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Backup failed for {len(failed)} collection(s): " + ", ".join(failed),
                "backup_path": req.backup_path,
                "results": results,
            },
        )
    return {"backup_path": req.backup_path, "results": results}


# ---------------------------------------------------------------------------
# Legacy endpoints (backward compatibility)
# ---------------------------------------------------------------------------


@app.get("/health")
def legacy_health():
    """Legacy health endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Legacy /ingest and /search endpoints (backward compat with existing tests)
# ---------------------------------------------------------------------------

# Ensure default collection exists for legacy endpoints
_LEGACY_COLLECTION = "legacy"


def _ensure_legacy_collection():
    """Creates the legacy collection if it doesn't exist."""
    if _LEGACY_COLLECTION not in _manager._collections:
        _manager.create(CreateCollectionRequest(name=_LEGACY_COLLECTION, dimension=128))


@app.post("/ingest")
async def legacy_ingest(request: Request):
    """Legacy ingest endpoint - maps to default 'legacy' collection."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    body = await request.json()
    _ensure_legacy_collection()
    vectors = body.get("vectors", [])
    if not isinstance(vectors, list) or len(vectors) == 0:
        raise HTTPException(status_code=400, detail="vectors must be a non-empty list")
    if len(vectors) > _RATE_LIMIT_MAX_VECTORS_PER_INSERT:
        raise HTTPException(
            status_code=400, detail=f"Max {_RATE_LIMIT_MAX_VECTORS_PER_INSERT} vectors per request"
        )
    doc_ids = body.get("doc_ids", None)
    vecs = np.array(vectors, dtype=np.float32)
    if np.any(np.isnan(vecs)):
        raise HTTPException(status_code=400, detail="vectors contain NaN values")
    db = _manager.get(_LEGACY_COLLECTION)
    try:
        added = db.add(ids=doc_ids, vectors=vecs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _manager.update_count(_LEGACY_COLLECTION, added)
    return {"added": added}


@app.post("/search")
async def legacy_search(request: Request):
    """Legacy search endpoint - maps to default 'legacy' collection."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    body = await request.json()
    query_vec = body.get("query", [])
    if not isinstance(query_vec, list) or len(query_vec) == 0:
        raise HTTPException(status_code=400, detail="query must be a non-empty list")
    k = min(max(body.get("k", 10), 1), 10000)
    _ensure_legacy_collection()
    query = np.array(query_vec, dtype=np.float32)
    if np.any(np.isnan(query)):
        raise HTTPException(status_code=400, detail="query contains NaN values")
    db = _manager.get(_LEGACY_COLLECTION)
    raw = db.search(query, k=k, include_metadata=True)
    if isinstance(raw, list):
        results = [{"doc_id": r.id, "distance": r.score} for r in raw]
    else:
        mu, alpha, kappa = raw
        results = [
            {"doc_id": f"idx_{i}", "distance": float(np.asarray(alpha[i]).flat[0])}
            for i in range(min(k, len(mu)))
        ]
    return {"results": results[:k]}
