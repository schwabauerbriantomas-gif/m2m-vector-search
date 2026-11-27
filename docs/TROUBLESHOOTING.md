# Troubleshooting

## Every result shows `energy=10.0000 confidence=0.0000`

The query is out of distribution: no splat covers it, so the energy hits the OOD
cap (`OOD_ENERGY = 10`, `confidence = sigmoid(-10) = 4.5e-5`). This is the normal
output for `np.random.randn` data in 128+ dimensions, where the nearest neighbour
has a cosine of about `0.25`. Use data with clusters (or real embeddings) and a
query close to one of them to see distinct energies.

## `ValueError: vectors contain NaN or Inf values` / `vector norm ... < 1e-6`

`add()` and `update()` reject such vectors before touching the store. Over HTTP
this is a `400` with the same message. Normalize or clean the embeddings; a
zero vector is not a valid direction.

## `ValueError` from a metadata filter / `delete(filter=...)` refuses to run

Filters are fail closed. Check the operator name (`$eq`, `$ne`, `$in`, `$nin`,
`$gt`, `$gte`, `$lt`, `$lte`), the value type, and that the condition is not `{}`.
`filter=None` means "no filter".

## `TypeError: k must be an integer`

`k` has to be an `int` (or an integer NumPy scalar). `k=10.0` is rejected instead
of being truncated.

## `ValueError: too many values to unpack (expected 3)`

The legacy return of `search()` is **always** a 4-tuple
`(mu, alpha, kappa, splat_indices)`. Unpack four values, or use
`include_metadata=True` and work with `DocResult` objects.

## `RuntimeError` from `save()`

`save()` without arguments needs a `storage_path`. Create the database with
`SimpleVectorDB(storage_path="./data")` or write a snapshot with `save("./snap")`.

## `FileExistsError` from `save(path)`

`path` exists and is not empty. Snapshots never overwrite: choose another
directory or remove the old one.

## `FileNotFoundError`, `NotADirectoryError` or `IndexTamperedError` from `load(path)`

`load()` validates its argument. A `.bin` dump of `M2MDatasetTransformer` is not a
storage directory (it carries no ids); use `engine.load_optimized(path)` for that
format. `IndexTamperedError` means the HMAC signature of the storage does not
match `M2M_HMAC_SECRET`: the files changed, or you are using a different secret.

## `RuntimeError: M2M_HMAC_SECRET environment variable required ...`

`M2MPersistence.save_index()` / `load_index()` only work with a secret. Set it, or
use `save()`/`load()` which sign only when the variable is set.

## `UserWarning` with `device="cuda"`, or "Vulkan init failed ... Falling back to CPU"

The requested backend is not available (`torch` missing for CUDA; the `vulkan`
extra or the `glslc` compiler missing for Vulkan). The engine runs on CPU:
`db.engine.backend` tells you. Install `"m2m-vector-search[embeddings]"` for CUDA
or the Vulkan SDK plus `"m2m-vector-search[vulkan]"` for Vulkan.

## Results are missing after a restart

Check that you reopen with the same `storage_path` and `latent_dim`. With
`storage_path` the splats, the index and the soft-deleted documents are restored;
without it the database lives in memory only. Metadata written through the API
needs a checkpoint (`save()` or `POST /v1/admin/checkpoint`) to be durable.

## `ModuleNotFoundError: No module named 'langchain_core'`

The LangChain integration is an extra: `pip install "m2m-vector-search[langchain]"`.

## `ModuleNotFoundError: No module named 'm2m'` when running from a checkout

The package lives under `src/`. Install it in editable mode
(`pip install -e ".[dev]"`) or run scripts with `PYTHONPATH=src python ...`.

## `pytest` hangs or errors on a fresh checkout

A bare `pytest` also selects the tests marked `gpu`, `integration` and
`embeddings`, which need `torch`, CUDA or model downloads. Use the CI selection:

```bash
pytest tests/ -m "not slow and not integration and not gpu and not embeddings"
```

## Low recall on random (uniform) vectors

HRM2 routes queries to `n_probe` coarse clusters; on data without clusters the neighbours
of a query are spread over all of them. Recall@10 with the default `n_probe` is 0.149 at
10k and 0.093 at 50k on uniform data. There is no `exact=True` switch: set
`SimpleVectorDB(n_probe=...)` to the number of coarse clusters, pass a filter (the exact
path of P17), or scan with NumPy — on such data the scan is faster than both exact
options (`docs/PERFORMANCE.md`).

## `RuntimeError: No encoder set ...` from `SemanticMemoryDB.store()`

The constructor accepts `encoder=None`, but every text operation needs one. Pass a callable
`text -> np.ndarray` of dimension `latent_dim`, or store pre-computed vectors with
`store_with_vector()` (`tests/test_semantic_memory.py`).

## `search_batch()` returns `-1`, or `search()` returns fewer than `k` results

Without a filter the probe is not widened to replace soft-deleted documents. In HRM2 mode
(`n > 3000`) a query whose probed clusters are mostly deleted comes back with fewer than
`k` results (`-1` padding in the batch). A filter widens the probe and is exact — even one
that accepts everything (`filter=lambda meta: True`). This is a known limitation of the
unfiltered path (`tests/test_audit_batch.py` covers the exclusion of deleted documents).

## `429 Too Many Requests` from the edge API

`M2M_RATE_LIMIT_PER_MIN` is set. Wait the number of seconds in the `Retry-After` header, or
unset the variable (`0` or negative also switches the limit off; off is the default).

## `401 Unauthorized` from the edge API with a key configured

Every endpoint except `/health` and `/v1/health` asks for `M2M_API_KEY` when it is set,
including the read-only `GET` endpoints of vectors and collections. Send the key, or unset
the variable for a local development setup (`SECURITY.md`).

## The coordinator's `POST /search` returns empty results / the edges answer `401`

The edges have `M2M_API_KEY` set but the coordinator does not forward one. Set
`M2M_EDGE_API_KEY` on the coordinator to the edges' key; the coordinator's own
`M2M_COORDINATOR_API_KEY` is never forwarded (`docs/cluster.md`).

## `FileNotFoundError` from `SQ8DiskStore.load()`

A dump built with `build_from_float_npy()` keeps only the SQ8 codes; the `float32` source
is needed again to rescore. Its path is recorded in `float_source.json` when the dump is
built; if that file is gone, call `load(dump_dir, float_path=...)` with the source. The
error names the missing file and the remedy (`tests/test_sq8_disk_load.py`).

## `add()` is slow with `storage_path`

Each call writes to disk: one document costs about 85.8 ms (p95 270) against
0.176 ms in memory. Add documents in batches: 1000 in one call take 346 ms
with storage and 110 ms without (`docs/PERFORMANCE.md`).
