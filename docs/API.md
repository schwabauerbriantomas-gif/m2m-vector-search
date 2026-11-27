# API reference

M2M has two surfaces: the Python API (`import m2m`) and the HTTP API
(`m2m.api.edge_api`, `m2m.api.coordinator_api`). The signatures below were read
with `inspect.signature` from the 2.4.0 source; the docstrings in `src/m2m/` are
the authority. There are no separate type stubs: the package ships `py.typed` and
the annotations live in the source.

## Python API

### `SimpleVectorDB`

```python
import numpy as np
from m2m import SimpleVectorDB

db = SimpleVectorDB(latent_dim=128, storage_path="./data")   # storage_path optional
vectors = np.random.default_rng(0).standard_normal((3, 128)).astype(np.float32)
n = db.add(ids=["a", "b", "c"], vectors=vectors)
print(n, [r.id for r in db.search(vectors[0], k=2, include_metadata=True)])
```

Constructor parameters are documented in [`CONFIGURATION.md`](CONFIGURATION.md).

#### Writing

| Method | Behaviour |
|---|---|
| `add(ids=None, vectors=None, metadata=None, documents=None) -> int` | Adds `[N, D]` vectors. Ids are auto-generated when omitted. Raises `ValueError` before mutating anything for `NaN`/`Inf`, norm `< 1e-6`, duplicate or unhashable ids in the batch. Re-adding an active id rewrites its splat; re-adding a soft-deleted id revives it. Returns the number of vectors added. Vectors beyond `max_splats` are skipped with a warning. With `storage_path` every call writes to disk: one document costs about 85.8 ms against 0.176 ms in memory, so add in batches. |
| `update(id, vector=None, metadata=None, document=None, upsert=False) -> UpdateResult` | Updates a document; `upsert=True` creates it (or revives it) — a vector is then required. |
| `delete(id=None, ids=None, filter=None, hard=False) -> DeleteResult` | Soft-deletes by default. `hard=True` also removes the splat from the index (compact + rebuild, `O(n)` per call) and frees capacity. `filter` uses the metadata-filter syntax below. `DeleteResult.deleted` counts only active-to-deleted transitions. |
| `get_stats() -> dict` | Counters of the database, plus `backend` (the backend in use) and `requested_device` (what was asked for). `db.engine.get_backend_info()` has the details [P15; `tests/test_audit_closure.py`]. |

#### Searching

| Method | Returns |
|---|---|
| `search(query, k=10, filter=None, include_energy=False, include_metadata=False)` | With `include_metadata=True`, `include_energy=True` or `filter`: `list[DocResult]`, at most `k` active documents that pass the filter. Otherwise the legacy tuple `(mu, alpha, kappa, splat_indices)` — always four values, also when empty or in LSH mode. With a filter in IVF mode the top-k is **exact** (pruning bound on the cluster radius [P17]; `tests/test_p17_exact_filter.py`). |
| `search_batch(queries, k=10, include_metadata=False)` | `include_metadata=False`: `np.ndarray` `[B, k]` of positions (`-1` when missing); `True`: `list[list[DocResult]]`. `(B, 0)` for an empty database. Each query is scored against the rows of **its own** probed clusters, never the union of the batch. Without a filter the probe is **not widened** to replace soft-deleted documents: in HRM2 mode (`n > 3000`) rows can hold fewer than `k` positions (`-1`) when most documents of the probed clusters are deleted; a filter widens the probe and is exact (known limitation, see [PERFORMANCE.md](PERFORMANCE.md); `tests/test_audit_batch.py`). |
| `search_batch_filtered(queries, k=10, filter_fn=None, filter=None)` | `np.ndarray` `[B, k]` of positions. The filter runs **before** the top-k; `filter_fn` is `callable(position) -> bool`, `filter` a dict or `callable(metadata) -> bool` (same syntax as `search`). Both together: `ValueError`. Exact top-k in IVF mode [P17]. |
| `search_with_energy(query, k=10) -> SearchResult` | `SearchResult.results` (list of `DocResult` with `energy`/`confidence`), `total_confidence`, `query_energy`. Needs the energy features (`enable_ebm=True`, `mode="ebm"` or `AdvancedVectorDB`). |
| `get_energy(vector) -> float`, `suggest_exploration(n=3)`, `find_knowledge_gaps(n=5)` | Energy features; they raise `RuntimeError` without the energy features (`tests/test_crud.py`). |

#### Persistence

| Method | Behaviour |
|---|---|
| `save(path=None) -> str` | `save()`: WAL checkpoint of `storage_path`; `RuntimeError` without a `storage_path`. `save(path)`: self-contained snapshot; `FileExistsError` if `path` exists and is not empty; `path == storage_path` means `save()`. |
| `load(path) -> SimpleVectorDB` | Replaces the in-memory state with the storage at `path` (splats, index, soft-deletes rebuilt) and keeps persisting there. `FileNotFoundError`, `NotADirectoryError` (a file, e.g. a `.bin` dump), `RuntimeError`/`IndexTamperedError` on a bad HMAC signature, `ValueError` on path traversal. |
| — | Reopening: `SimpleVectorDB(latent_dim=D, storage_path="./data")` restores the database, including its index. |

Signing is opt-in through `M2M_HMAC_SECRET` (see [`SECURITY.md`](../SECURITY.md)).

### `AdvancedVectorDB`

Extends `SimpleVectorDB` with `enable_soc=True` and `enable_energy_features=True`
and adds:

| Method | Returns |
|---|---|
| `consolidate(threshold=None) -> int` | Runs SOC consolidation and returns the number of splats pruned. It is invoked by the caller; nothing runs in the background. |
| `feedback(query, relevant_ids=None, irrelevant_ids=None) -> FeedbackResult` | Also on `SimpleVectorDB`. Hebbian update by **document id**: `α`, `κ` up and `μ` towards the query for relevant ids; `α`, `κ` down for irrelevant ones; `μ` stays on the sphere. Unknown or soft-deleted ids go to `FeedbackResult.unknown_ids`. The HRM2 index is kept in sync, so `search()` sees the drift [audit r3a-08, r2-14; `tests/test_audit_learning_soc.py`]. |
| `maybe_consolidate(threshold=None) -> ConsolidationDecision` | Consolidates **only** when the criticality index is SUPERCRITICAL; idempotent (a second call returns `triggered=False`). `auto_consolidate=True` in the constructor (default `False`) calls it after every change of the store [audit r3a-07; `tests/test_audit_learning_soc.py`]. |
| `check_criticality() -> CriticalityReport` | Diagnostic with `state`, `index`, `energy_variance`, `size_variance`. |
| `relax(iterations=10) -> RelaxationResult` | Relaxation step of the SOC engine. |
| `trigger_avalanche(seed_point=None) -> AvalancheResult` | Triggers an SOC avalanche. |
| `generate(query, n_steps=10) -> np.ndarray` | Langevin exploration: returns **one** sample of dimension `D`. `n_steps` is accepted and passed on, but `m2m.M2MEngine.generate()` ignores it. |

### `DocResult`

| Field | Meaning |
|---|---|
| `id` | Document id. |
| `score` | **Cosine similarity** in `[-1, 1]` between the normalized query and the stored vector. Same semantics on every path. Higher is closer. |
| `alpha` | Weight `alpha_i` of the splat. `None` in LSH mode. |
| `distance` | L2 distance between the normalized query and the stored vector; `distance**2 == 2 - 2*score` on the sphere. |
| `energy` | With energy features only: `E_i(q) = -ln(alpha_i) + kappa_i*||q - mu_i||**2`, capped at `OOD_ENERGY = 10`. Lower is more confident. |
| `confidence` | With energy features only: `sigmoid(-energy)`. `0.5` at the exact centre of a splat with `alpha = 1`; the cap maps to `4.5e-5`. |
| `metadata`, `document` | As passed to `add()` (`{}` / `None` when absent). |
| `vector` | Stored vector in `search()`; `None` in `search_batch(include_metadata=True)`. |

**Reading the energy.** `energy == 10.0` and `confidence == 0.0000` on every result
mean the query is out of distribution: no splat covers it. With `np.random.randn`
data in 128 dimensions that is the expected output (the nearest neighbour has a
cosine of about `0.25`), not a bug.

### Metadata filters

```python
import numpy as np
from m2m import SimpleVectorDB

rng = np.random.default_rng(0)
db = SimpleVectorDB(latent_dim=16)
db.add(ids=["a", "b", "c", "d"], vectors=rng.standard_normal((4, 16)).astype(np.float32),
       metadata=[{"category": "science", "year": 2021}, {"category": "art", "year": 2019},
                 {"category": "science", "year": 2019}, {"tag": "x"}])
q = rng.standard_normal(16).astype(np.float32)
Q = rng.standard_normal((2, 16)).astype(np.float32)

db.search(q, k=10, filter={"category": "science"})                 # equality
db.search(q, k=10, filter={"year": {"$gte": 2020, "$lt": 2025}})   # operators
db.search(q, k=10, filter={"tag": {"$in": ["a", "b"]}})
db.search(q, k=10, filter=lambda meta: meta.get("public") is True)  # callable
db.delete(filter={"category": {"$eq": "spam"}})
db.search_batch_filtered(Q, k=10, filter={"category": "science"})
```

Operators: `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, a bare value
(as `$eq`) or a callable `metadata -> bool`. Filters are **fail closed**: an
unknown operator, an invalid value or an empty condition `{}` raise `ValueError`,
and `delete(filter=...)` refuses to run on such a filter. `filter=None` means no
filter. Filter masks are cached per metadata version; mutating a metadata dict
in place after `add()` is not seen by the cache — call `update()` instead.

### Lower-level building blocks

- `m2m.engine.M2MEngine` selects the hardware backend (`get_backend_info()`); the
  backend in use is `db.engine.backend`. `m2m.M2MEngine` wraps the splat memory and is
  what `create_m2m(M2MConfig(...))` returns; it adds `search`, `add_splats`, `generate`,
  `load_optimized`, `get_statistics` and `get_backend_info()`. Its `search(queries, k)`
  takes a `[B, D]` batch and returns four `[B, k, ...]` arrays
  `(mu, alpha, kappa, splat_indices)`.
- `m2m.hrm2_engine.HRM2Engine` / `HRM2Config`: hierarchical routing with
  `query(q, k, lod=2)` and `query_batch(Q, k, lod=2)`, which agree row by row.
- `m2m.gate_modulus.gate_select_k(X, target_docs_per_cluster=25, verdict_cache=None,
  seed=42, stability_seeds=0)` chooses the number of coarse clusters and returns a
  `GateKSelection` whose verdict reports the discarded and omitted scales.
- `m2m.sq8_disk.SQ8DiskStore(dim, dump_dir=None, spectral_r=None)`: SQ8 codes in RAM,
  `float32` on disk for exact rescoring (`add`, `build_from_float_npy`, `flush`,
  `load`, `search_batch`, `ram_bytes`). The spectral phase is opt-in.
- `m2m.geometry`: `exp_map`, `log_map`, `geodesic_distance`, `project_to_tangent`,
  `normalize_sphere` (also re-exported by the package).
- `m2m.integrations.langchain.M2MVectorStore`: a LangChain `VectorStore` with
  `add_texts`, `similarity_search`, `update`, `delete`, `filter_by_metadata`;
  a retriever is `as_retriever()`. Requires `pip install "m2m-vector-search[langchain]"`.
- `m2m.semantic_memory.SemanticMemoryDB`: hybrid BM25 + vector search with RRF. It needs an
  `encoder`; without one `store()` raises `RuntimeError` (`No encoder set`) — use
  `store_with_vector()` for pre-computed vectors (`tests/test_semantic_memory.py`).
- There is **no** LlamaIndex integration. `examples/llamaindex_rag.py` builds a toy
  retriever on `M2MEngine` without importing `llama-index`.

## HTTP API

```bash
pip install "m2m-vector-search[api]"
python -m m2m                                   # 127.0.0.1:8080
uvicorn m2m.api.coordinator_api:app --port 8000 # cluster coordinator
```

### Edge API (`m2m.api.edge_api`)

| Method and path | Purpose |
|---|---|
| `GET /health`, `GET /v1/health` | Liveness, no key. |
| `GET /v1/stats` | Global statistics. |
| `GET /v1/collections`, `POST /v1/collections`, `GET /v1/collections/{name}`, `DELETE /v1/collections/{name}` | Collection CRUD. |
| `POST /v1/collections/{name}/vectors` | Insert vectors (ids, vectors, metadata, documents). |
| `GET`, `PUT`, `DELETE /v1/collections/{name}/vectors/{vector_id}` | Read, upsert or delete one vector. |
| `GET /v1/collections/{name}/stats` | Per-collection statistics. |
| `POST /v1/collections/{name}/search`, `/query` | k-NN search (`query`, `k`, optional `filter`). |
| `POST /v1/collections/{name}/energy`, `/explore`, `GET .../suggest` | Energy features. |
| `POST /v1/admin/checkpoint`, `POST /v1/admin/backup` | Checkpoint every collection with a `storage_path`; write a backup. Fail closed without a key. |
| `POST /ingest`, `POST /search` | Legacy endpoints. |

With `M2M_API_KEY` set, every endpoint of the table requires the key (`401` otherwise)
except the two `/health` paths [P15; `tests/test_audit_closure.py`].
The OpenAPI schema is served at `/docs` and `/openapi.json` by the running server.

### Status codes

| Code | When |
|---|---|
| `400` | A validation `ValueError` from `add`/`update`/`delete` (bad vector, bad filter, duplicate ids). |
| `401` | Missing or wrong API key on an endpoint that requires it. |
| `404` | Unknown collection or vector. |
| `413` | Body larger than `M2M_MAX_BODY_BYTES`; rejected before being read (`text/plain`). |
| `422` | Request body does not match the pydantic model, or more than `100_000` vectors in one insert. |
| `429` | More than `M2M_RATE_LIMIT_PER_MIN` requests from one client IP in 60 s. Only when that variable is a positive number (opt-in, off by default); the response carries a `Retry-After` header [P15; `tests/test_audit_closure.py`]. |
| `500` | Real failure of checkpoint/backup (returned with the error, not swallowed). |
| `503` | `/v1/admin/*` with no API key configured. |

The three gaps listed by the audit are closed [P15]: vector `GET`/`PUT`/`DELETE` validate the
key (they answer `401` before the collection lookup), the rate limit is real and opt-in, and
the coordinator has `GET /health`. `/health` and `/v1/health` are the only edge endpoints that
never ask for a key.

### Coordinator API (`m2m.api.coordinator_api`)

`POST /register`, `POST /heartbeat`, `POST /route`, `POST /search`. Edges register
themselves, the router picks edges per query and the aggregator merges the
results. The four `POST` endpoints are authenticated with `M2M_COORDINATOR_API_KEY` when
set; `GET /health` needs no key. On a fan-out the coordinator sends `M2M_EDGE_API_KEY` to each
edge (the coordinator's own key is never forwarded) [P15; `tests/test_audit_closure.py`]. See
[`cluster.md`](cluster.md).
