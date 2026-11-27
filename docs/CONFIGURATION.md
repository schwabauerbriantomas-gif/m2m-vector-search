# Configuration

Everything M2M reads from the environment, with the default and the place in
the source where it is read (lines as of 2.4.0). The list was produced by
grepping `os.environ` in `src/`; there is no other configuration file and no
`.env` loader — set the variables in the shell, in the service unit, or in the
`environment:` block of the container.

## Environment variables

<!-- P10b: verificar P15 -->
| Variable | Read in | Default | Effect |
|---|---|---|---|
| `M2M_HOST` | `src/m2m/__main__.py:25` | `127.0.0.1` | Host `python -m m2m` binds to. The Dockerfiles set it to `0.0.0.0`. A warning is logged if the host is not loopback and `M2M_API_KEY` is unset (`src/m2m/__main__.py:51`). Only affects `python -m m2m` / the `m2m` script; `uvicorn m2m.api.edge_api:app` takes `--host` from the command line. |
| `M2M_PORT` | `src/m2m/__main__.py:26` | `8080` | Port of `python -m m2m`. Must parse as an integer. |
| `M2M_API_KEY` | `src/m2m/api/edge_api.py:353` | unset = no authentication | When set, the endpoints that validate authentication require it as `Authorization: Bearer <key>` or `X-API-Key: <key>` (constant-time compare). `/v1/admin/*` answer `503` when it is unset. Read per request. |
| `M2M_COORDINATOR_API_KEY` | `src/m2m/api/coordinator_api.py:26` | unset = no authentication | Key of the coordinator's `/register`, `/heartbeat`, `/route` and `/search`. **Read once at import time**: set it before importing the module. |
| `M2M_CORS_ORIGINS` | `src/m2m/api/edge_api.py:275` | `*` (open) | Comma-separated list of allowed origins for the edge API. With the wildcard, credentials are not allowed. Read once at import time. The coordinator has no CORS middleware. |
| `M2M_MAX_BODY_BYTES` | `src/m2m/api/edge_api.py:248` | `33554432` (32 MiB) | Requests with a larger body get `413` before the body is read. A value that is not a non-negative integer logs a warning and falls back to the default. Read when the app object is built. |
| `M2M_HMAC_SECRET` | `src/m2m/storage/persistence.py:539, 554, 682, 698` | unset = no signing | When set, `save()`/checkpoint writes an HMAC-SHA256 manifest of the storage that is verified on reopen and by `load()` (mismatch: `IndexTamperedError`); a storage without a manifest loads with a warning. `M2MPersistence.save_index()`/`load_index()` require it (`RuntimeError` otherwise). Not encryption: it detects modification, it does not hide content. |
| `M2M_RATE_LIMIT_PER_MIN` | not read yet | absent / `0` = off | <!-- P10b: verificar P15 --> Real per-client rate limit, opt-in. Not implemented in 2.4.0: the existing limiter never records requests, so its `100` requests/minute cap is never enforced. |

No other `M2M_*` variable is read by `src/`. `EDGE_ID` and `COORDINATOR_URL`
used to be listed by the deployment files and are gone; the edge node takes its
coordinator URL and id as constructor arguments.

## Running the server

```bash
pip install "m2m-vector-search[api]"

# localhost only, no authentication (development)
python -m m2m

# reachable from the network, with a key
M2M_HOST=0.0.0.0 M2M_PORT=8000 M2M_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" python -m m2m
```

The interactive OpenAPI docs are then at `http://127.0.0.1:8080/docs`.

## Constructor parameters

### `SimpleVectorDB`

`src/m2m/__init__.py`. Signature:

```python
from m2m import SimpleVectorDB

# Every parameter at its default (2.4.0)
db = SimpleVectorDB(
    device=None,
    latent_dim=640,
    enable_lsh_fallback=False,
    lsh_threshold=0.15,
    storage_path=None,
    enable_wal=True,
    enable_ebm=False,
    mode='standard',
    max_splats=None,
    n_probe=None,
    gate_enabled=None,
    k_selection=None,
    gate_target_docs_per_cluster=None,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `None` | Requested backend: `None`/`"cpu"`, `"cuda"`, `"vulkan"`. The backend actually in use is `db.engine.backend` (see `docs/API.md`). |
| `latent_dim` | `640` | Dimension `D` of the vectors. |
| `enable_lsh_fallback` | `False` | Enable the Cross-Polytope LSH index. Experimental; off by default. |
| `lsh_threshold` | `0.15` | Silhouette below which LSH is used when the fallback is enabled. |
| `storage_path` | `None` | Directory of the persistent storage (shards + SQLite metadata + WAL). Without it the database lives in memory only and `save()` raises. |
| `enable_wal` | `True` | Write-ahead log for crash recovery. |
| `enable_ebm` | `False` | Energy-based model features: `include_energy=True` in `search()`, `search_with_energy()`, `get_energy()`, `suggest_exploration()`, `find_knowledge_gaps()`. |
| `mode` | `"standard"` | Preset: `"edge"` (no WAL), `"standard"`, `"ebm"`. |
| `max_splats` | `None` | Capacity of the splat store; vectors beyond it are rejected with a warning (not stored). `None` uses the `M2MConfig` default. |
| `n_probe` | `None` | Clusters probed per query by the HRM2 index. `None` uses the `HRM2Config` default. |
| `gate_enabled` | `None` | Enable the gate of modulus for choosing the number of coarse clusters. |
| `k_selection` | `None` | How the coarse `k` is chosen (`"auto"` uses `gate_select_k`). |
| `gate_target_docs_per_cluster` | `None` | Operational `k` for hierarchical corpora (`k = N // target`). |

`AdvancedVectorDB` adds `enable_soc=True` and `enable_energy_features=True` and
exposes `consolidate()`, `check_criticality()`, `relax()`,
`trigger_avalanche()` and `generate()` on top of the `SimpleVectorDB` API.

### `HRM2Config` / `HRM2Engine`

`HRM2Config` is a dataclass (`src/m2m/hrm2_engine.py`) exported from the
package. Fields: `embedding_dim`, `n_coarse`, `n_fine`, `n_probe`, `metric`,
`batch_size`, `random_state`, `auto_k`, `k_selection`,
`gate_target_docs_per_cluster`, `adaptive_probing`, `adaptive_threshold`,
`adaptive_min_probe`. Defaults: `n_coarse=100`, `n_fine=1000`, `embedding_dim=640`, `n_probe=5`, `batch_size=1000`, `random_state=42`, `adaptive_probing=False`, `adaptive_threshold=1.5`, `metric='euclidean'`, `auto_k=False`, `k_selection='fixed'`, `gate_target_docs_per_cluster=25`, `adaptive_min_probe=1`.

`HRM2Engine(n_coarse=None, n_fine=None, embedding_dim=None, n_probe=None,
batch_size=None, random_state=None, config=None, metric=None, auto_k=None,
k_selection=None, gate_target_docs_per_cluster=None)` takes either individual
values or a `config=HRM2Config(...)`.

Levels of detail of `HRM2Engine.query(q, k, lod=...)`:

| `lod` | What is scanned | Use |
|---|---|---|
| `0` | The probed coarse clusters; returns the nearest splats among them. | Fast, approximate. |
| `1` | Only the nearest fine clusters inside the probed coarse clusters. | Approximation; recall is lower than `lod=2` because the neighbours of a query spread over several fine clusters. |
| `2` (default) | Coarse routing plus fine re-ranking. | Default. |

Adaptive probing (`adaptive_probing=True` in `HRM2Config`, or
`engine.adaptive_probing = True`) adds probes for a query when the first ones
yield too few candidates.

### `SQ8DiskStore`

`SQ8DiskStore(dim, dump_dir=None, spectral_r=None)` (`src/m2m/sq8_disk.py`):
SQ8 codes in RAM, `float32` on disk (mmap) for exact rescoring. `spectral_r`
enables the opt-in spectral phase: codes are projected to a rank-`r` subspace
(`"auto"` picks the rank from the spectrum and falls back to `r = dim` when
there is no clear knee). It reduces the size of the codes at the cost of
extra rescoring per query; it is not a speed-up. Dumps written by versions
before 2.4.0 with the old knee detection must be rebuilt.

### `SemanticMemoryDB`

`SemanticMemoryDB(encoder=None, latent_dim=384, storage_path=None,
hybrid_weight=0.6, fusion_method="rrf", bm25_k1=1.5, bm25_b=0.75,
temporal_decay=False, temporal_half_life_days=30.0, auto_categorize=False,
device="cpu", mode="standard")` (`src/m2m/semantic_memory.py`). `encoder` is
any callable `text -> np.ndarray` of dimension `latent_dim`; without one the constructor
accepts the call but `store()` raises `RuntimeError` (`No encoder set`) unless you use
`store_with_vector()` with pre-computed vectors. `fusion_method` is one of `"rrf"`,
`"weighted"`, `"vector_only"`, `"bm25_only"`.

### `M2MConfig`

`M2MConfig` (`src/m2m/config.py`) is the dataclass behind `M2MEngine`
(`create_m2m(config)`): `device`, `dtype`, `latent_dim`, `n_splats_init`,
`max_splats`, `knn_k`, `init_alpha`, `init_kappa`, `min_kappa`, `max_kappa`,
`splat_temperature`, the energy weights, `soc_threshold`, the Vulkan and CUDA
flags, and more. Note that `M2MConfig.device` is what was **requested**; for the
backend in use ask the engine (`db.engine.backend`).
