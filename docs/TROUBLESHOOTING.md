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
