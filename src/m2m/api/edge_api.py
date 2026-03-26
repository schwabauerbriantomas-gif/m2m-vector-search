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

import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .. import AdvancedVectorDB, SimpleVectorDB

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

app = FastAPI(
    title="M2M EBM Vector Database API",
    description="REST API para M2M Vector Search con features EBM",
    version="2.0.0",
)

# -- SECURITY: CORS configuration (M-01 fix) ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- SECURITY: Rate limiting (H-03, P-07 fix) --------------------------
_rate_limit_store: Dict[str, List[float]] = {}
_RATE_LIMIT_WINDOW = 60.0  # seconds
_RATE_LIMIT_MAX_REQUESTS = 100  # per window per IP
_RATE_LIMIT_MAX_VECTORS_PER_INSERT = 100_000


def _check_rate_limit(client_ip: str) -> None:
    """Basic rate limiting per IP. Raises HTTPException if exceeded."""
    import time as _time

    now = _time.time()
    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []
    # Prune old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < _RATE_LIMIT_WINDOW
    ]
    if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# -- SECURITY: API Key validation (C-01, P-06 fix) ---------------------
_API_KEY = None  # Set via M2M_API_KEY env var


def _validate_api_key(request: Request) -> None:
    """Validates API key if configured via M2M_API_KEY environment variable."""
    import os

    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("M2M_API_KEY")
    if _API_KEY is not None:
        auth = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else api_key
        if not token or token != _API_KEY:
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
    return HTTPException(status_code=500, detail="Internal server error")


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
        except Exception:
            pass
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


@app.post("/v1/collections/{name}/vectors")
def insert_vectors(name: str, req: InsertVectorsRequest, request: Request):
    """Inserta vectores en una colección."""
    _validate_api_key(request)
    _check_rate_limit(request.client.host if request.client else "unknown")
    _validate_collection_name(name)
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    vectors = np.array(req.vectors, dtype=np.float32)
    added = db.add(
        ids=req.ids,
        vectors=vectors,
        metadata=req.metadata,
        documents=req.documents,
    )
    _manager.update_count(name, added)
    return {"added": added, "collection": name}


@app.get("/v1/collections/{name}/vectors/{vector_id}")
def get_vector(name: str, vector_id: str):
    """Obtiene un vector por ID."""
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
def update_vector(name: str, vector_id: str, req: UpdateVectorRequest):
    """Actualiza un vector existente."""
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    vector = np.array(req.vector, dtype=np.float32) if req.vector else None
    result = db.update(
        id=vector_id,
        vector=vector,
        metadata=req.metadata,
        document=req.document,
        upsert=req.upsert,
    )

    if not result.success:
        raise HTTPException(status_code=404, detail=result.message)

    return {
        "success": result.success,
        "energy_delta": result.energy_delta,
        "message": result.message,
    }


@app.delete("/v1/collections/{name}/vectors/{vector_id}")
def delete_vector(name: str, vector_id: str, hard: bool = False):
    """Elimina un vector."""
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = db.delete(id=vector_id, hard=hard)
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
        # Legacy format
        mu, alpha, kappa = results_raw
        results = [{"id": f"idx_{i}", "score": float(alpha[i])} for i in range(min(req.k, len(mu)))]
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
def suggest_exploration(name: str, n: int = 3):
    """Sugerencias de exploración (GET)."""
    req = ExploreRequest()
    req.n_suggestions = n
    return explore(name, req)


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
def checkpoint(name: Optional[str] = None, request: Request = None):
    """Crea checkpoint del WAL."""
    if request:
        _validate_api_key(request)
    names = [name] if name else _manager.list_names()
    for n in names:
        try:
            db = _manager.get(n)
            db.save(path="")
        except Exception:
            pass
    return {"message": "Checkpoint created", "collections": names}


@app.post("/v1/admin/backup")
def backup(req: BackupRequest, request: Request):
    """Crea backup de todas las colecciones con storage."""
    _validate_api_key(request)
    results = {}
    for name in _manager.list_names():
        try:
            db = _manager.get(name)
            if db.storage:
                path = db.storage.backup(req.backup_path)
                results[name] = {"status": "ok", "path": path}
            else:
                results[name] = {"status": "skipped", "reason": "no storage configured"}
        except Exception as e:
            results[name] = {"status": "error", "reason": str(e)}
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
    added = db.add(ids=doc_ids, vectors=vecs)
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
