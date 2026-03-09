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

import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


class InsertVectorsRequest(BaseModel):
    vectors: List[List[float]]
    ids: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    documents: Optional[List[str]] = None


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


class EnergyRequest(BaseModel):
    vector: Optional[List[float]] = None
    center: Optional[List[float]] = None
    radius: float = 1.0
    resolution: int = 20


class ExploreRequest(BaseModel):
    topic_vector: Optional[List[float]] = None
    n_suggestions: int = 3
    min_energy: float = 0.7


class BackupRequest(BaseModel):
    backup_path: str


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

_manager = CollectionManager()

# ---------------------------------------------------------------------------
# Health & Stats
# ---------------------------------------------------------------------------


@app.get("/v1/health")
def health():
    """Health check del servidor."""
    return {"status": "ok", "version": "2.0.0", "timestamp": time.time()}


@app.get("/v1/stats")
def stats():
    """Estadísticas generales del servidor."""
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
def create_collection(req: CreateCollectionRequest):
    """Crea una nueva colección."""
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
def delete_collection(name: str):
    """Elimina una colección."""
    try:
        _manager.delete(name)
        return {"message": f"Collection '{name}' deleted"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Vector CRUD
# ---------------------------------------------------------------------------


@app.post("/v1/collections/{name}/vectors")
def insert_vectors(name: str, req: InsertVectorsRequest):
    """Inserta vectores en una colección."""
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
def search(name: str, req: SearchRequest):
    """Búsqueda de similitud en la colección."""
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
        results = [
            {"id": f"idx_{i}", "score": float(alpha[i])}
            for i in range(min(req.k, len(mu)))
        ]
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
def query_advanced(name: str, req: SearchRequest):
    """Query avanzada con soporte de energía y exploración."""
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
def compute_energy(name: str, req: EnergyRequest):
    """Calcula la energía de un vector en el paisaje energético."""
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
def explore(name: str, req: ExploreRequest):
    """Explora regiones de alta incertidumbre."""
    try:
        db = _manager.get(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not db.ebm_enabled:
        raise HTTPException(status_code=400, detail="EBM not enabled for this collection.")

    topic = np.array(req.topic_vector, dtype=np.float32) if req.topic_vector else None
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
def checkpoint(name: Optional[str] = None):
    """Crea checkpoint del WAL."""
    names = [name] if name else _manager.list_names()
    for n in names:
        try:
            db = _manager.get(n)
            db.save(path="")
        except Exception:
            pass
    return {"message": "Checkpoint created", "collections": names}


@app.post("/v1/admin/backup")
def backup(req: BackupRequest):
    """Crea backup de todas las colecciones con storage."""
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
def legacy_ingest(request: dict):
    """Legacy ingest endpoint - maps to default 'legacy' collection."""
    _ensure_legacy_collection()
    vectors = request.get("vectors", [])
    doc_ids = request.get("doc_ids", None)
    vecs = np.array(vectors, dtype=np.float32)
    db = _manager.get(_LEGACY_COLLECTION)
    added = db.add(ids=doc_ids, vectors=vecs)
    _manager.update_count(_LEGACY_COLLECTION, added)
    return {"added": added}


@app.post("/search")
def legacy_search(request: dict):
    """Legacy search endpoint - maps to default 'legacy' collection."""
    _ensure_legacy_collection()
    query = np.array(request.get("query", []), dtype=np.float32)
    k = request.get("k", 10)
    db = _manager.get(_LEGACY_COLLECTION)
    raw = db.search(query, k=k, include_metadata=True)
    if isinstance(raw, list):
        results = [{"doc_id": r.id, "distance": r.score} for r in raw]
    else:
        mu, alpha, kappa = raw
        results = [{"doc_id": f"idx_{i}", "distance": float(np.asarray(alpha[i]).flat[0])} for i in range(min(k, len(mu)))]
    return {"results": results[:k]}

