# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Tags such as `[audit r1-13]` refer to the findings of the 2026-11 audit of the
code base; each one has a regression test under `tests/test_audit_*.py`.

## [Unreleased]

<!-- P10b: verificar P15 -->
The items below are contracted work that is **not implemented yet**. They will be
listed under a released version only once they land.

- Edge API: `PUT`/`DELETE`/`GET /v1/collections/{name}/vectors/{id}` must validate
  `M2M_API_KEY` when one is configured (today they stay open).
- Edge API: a real, opt-in rate limit via `M2M_RATE_LIMIT_PER_MIN`
  (absent or `0` = off). The current limiter never records requests.
- Coordinator: forward the configured key to the edges on fan-out and expose
  `GET /health`.
- Root `docker-compose.yml`: do not publish port 8080 without an API key.
- `SimpleVectorDB.feedback(query, relevant_ids=..., irrelevant_ids=...)` and
  `AdvancedVectorDB.maybe_consolidate()`.
- `restore()` must reject `..` in the backup path.

## [2.4.0] - Unreleased

Version `2.4.0` is set in `pyproject.toml` and `m2m.__version__` [audit r3a-04].
It is **not published on PyPI yet**: the `publish` workflow runs on
`release: published`, so nothing is uploaded until a GitHub Release is created.

### Note on the entries below

This release is the result of a full audit of the code, its claims and its
benchmarks (2026-11). Roughly 95 commits fix the behaviour of the engine, the
persistence layer and the HTTP API; the documentation was rewritten to describe
what the code does. Every behavioural change listed here cites the audit id of
the finding it closes, and each has a regression test.

**No benchmark figures are quoted in this entry.** The figures that used to live
in the README were measured before these fixes and with a methodology the audit
found flawed (for example `search_batch` scoring the union of the batch instead
of each query's own candidates). They are being re-measured and will be added to
`docs/PERFORMANCE.md` and the README in a follow-up change.

<!-- P10b: cifras -->

### Changed (breaking or contract-level)

**Search results**

- `DocResult.score` is the **cosine similarity** in `[-1, 1]` between the
  normalized query and the stored vector, on every path: `search()` with or
  without metadata, filter or energy, LSH mode and
  `search_batch(include_metadata=True)`. It used to mean different things per
  path (the splat weight `alpha` without a filter, `-distance` with a filter,
  `-sqrt(d2)` against the raw query in batch). The splat weight is now the new
  field `DocResult.alpha` and the L2 distance the new field `DocResult.distance`
  (`distance**2 == 2 - 2 * score` on the sphere) [audit r1-13].
- Per-result `energy` is the canonical energy of the **normalized query** under
  the splat of that result,
  `E_i(q) = -ln(alpha_i) + kappa_i * ||q - mu_i||**2`, capped at
  `m2m.ebm.energy_api.OOD_ENERGY` (`10`). It used to be evaluated at the
  position of the stored document itself, which gave the same value for every
  result. `confidence = sigmoid(-E)`; `0.5` is the exact centre of a splat with
  `alpha = 1`, and the OOD cap maps to `4.5e-5` [audit r3a-02, r2-15].
- `m2m.ebm.energy_api.splat_energy` is the single implementation of the energy
  landscape: the input is normalized to the sphere, computed in log-space and
  capped at exactly `OOD_ENERGY` [audit r2-15].
- `search()` returns the legacy 4-tuple `(mu, alpha, kappa, splat_indices)`
  **always** (also for an empty database and in LSH mode). Callers that unpacked
  three values must be updated [audit r1-05, r1-18].
- `search()` and `search_batch()` no longer return soft-deleted or hard-deleted
  documents; `search()` backfills from further candidates while active
  documents remain [audit r1-05, r1-06].

**Metadata filters (fail closed)**

- One helper evaluates metadata filters everywhere (`search`, `delete`,
  `search_batch_filtered`, `M2MPersistence.filter_by_metadata`): operators
  `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, a bare value, or a
  callable `metadata -> bool` [audit r1-14].
- An unknown operator, an invalid value type or an empty condition `{}` raise
  `ValueError` **before** anything is read or written. `delete(filter=...)` no
  longer deletes the whole collection when given an invalid operator
  [audit r1-14].
- `k` must be an `int` (or an integer NumPy scalar); any float raises
  `TypeError` instead of being truncated [audit r1-27].
- `search_batch_filtered(queries, k, filter=...)` accepts the same dict or
  callable as `search(filter=...)`; passing both `filter` and `filter_fn`
  raises `ValueError`. Filter masks are cached (LRU, at most 32 entries) per
  canonical filter and metadata version; every write invalidates them
  [audit P13].

**Writes**

- `add()` and `update()` reject vectors containing `NaN` or `Inf`, and vectors
  whose norm is below `1e-6`, with `ValueError`, before mutating anything
  [audit r1-02, r1-19]. `update(..., upsert=True)` without a vector is rejected
  [audit r1-19].
- Duplicate or unhashable ids inside one `add()` batch raise `ValueError`
  before the store is touched [audit r1-23].
- Adding an id that was soft-deleted revives it; adding an active id rewrites
  its splat in place (no orphan splat) [audit r1-04].
- `add()` is atomic under the write lock and stores its own copy of the vectors
  (no aliasing with the caller's array) [audit r1-03, r1-20]. Vectors rejected
  because of `max_splats` are neither stored nor persisted; a warning lists
  the ids [audit r1-26].
- `delete(hard=True)` removes the splat from the index (compact + rebuild of the
  IVF index, `O(n)` per call) and frees capacity; `DeleteResult.deleted`
  counts only transitions from active to deleted [audit r1-06, r1-21].
- `update(vector=...)` keeps `engine.embeddings`, the coarse cluster
  embeddings and the persisted copy in sync; the vector is persisted in place
  through `M2MPersistence.update_vector` [audit r1-24, r1-09].

**Persistence**

- Reopening a database created with `storage_path` restores the splats, the
  IVF index, the splat-to-id order and the soft-deleted documents. Previously
  the dictionaries were restored but `search()` returned nothing [audit r1-01].
- `save()` without arguments checkpoints the WAL of `storage_path` and raises
  `RuntimeError` if the database has no `storage_path`. It used to be a silent
  no-op. `save(path)` writes a self-contained snapshot (shards, metadata,
  soft-deletes) and raises `FileExistsError` if `path` exists and is not empty.
  [audit r1-17]
- `load(path)` validates its argument: `FileNotFoundError` if it does not
  exist, `NotADirectoryError` for a file (including the `.bin` dumps of
  `M2MDatasetTransformer`, which never carried ids), `RuntimeError` when the
  HMAC signature does not verify, `ValueError` for path traversal. It used to
  return `self` with a warning on any failure [audit r1-17].
- With `M2M_HMAC_SECRET` set, `save()`/checkpoint signs the storage with an
  HMAC-SHA256 manifest that is verified on reopen and by `load()`;
  a mismatch raises `IndexTamperedError`. A storage written without a manifest
  loads with a warning. `M2MPersistence.save_index()`/`load_index()` require
  the secret and verify the signature over the whole payload before
  deserialization [audit r3a-17].
- Paths used to save/load the signed index and the shards are confined to the
  storage directory: `..`, absolute paths and symlinks that resolve outside
  it raise `ValueError` [audit r3a-17].
- Metadata of one `add()` is written in a single SQLite transaction
  (`M2MPersistence.save_metadata_many()` is also public) [audit r3a-17].
- `SplatStore.reset()` returns the store to its just-constructed state
  [audit r3a-17].

**Index lifecycle, gate and HRM2**

- Attaching new splats to an existing index aligns every cluster mask to the
  new size; `HRM2Engine.query_with_details` no longer raises `IndexError`
  after an attach [audit r1-11].
- `consolidate()` and `compact()` followed by `build_index()` retrain the IVF
  index instead of attaching to a stale one [audit r1-25].
- `gate_select_k()` no longer crashes for `N` between 64 and 177; a degenerate
  trajectory yields an `"inconclusivo"` verdict instead of an exception
  [audit r2-10]. The verdict cache is keyed by the content of the corpus
  (SHA-1) and the algorithm version, not only by `(N, D, target, seed)`
  [audit r2-11]. Discarded and omitted scales are reported
  (`verdict['escalas']`); stability across seeds is opt-in via
  `stability_seeds` and reported as "not evaluated" when off [audit r2-12].
- `HRM2Engine.query_batch()` returns, row by row, the same result as
  `query()` (normalized before routing, per-row `argsort`, adaptive `n_probe`
  per row) [audit r2-18]. `lod=0` and `lod=1` return the nearest splats of
  the clusters they probe [audit r2-19]. `HRM2Config` is wired in
  `HRM2Engine.__init__` and exported from the package [audit r2-18, r3a-10].
- `GPUVectorIndex` raises `ValueError` for `dim > 1024` before compiling
  shaders or starting Vulkan [audit r2-20].

**SQ8 disk index**

- `SQ8DiskStore.add()` and `build_from_float_npy()` reject `NaN`/`Inf` before
  writing anything [audit r2-05]. `search_batch()` works in chunks so that the
  squared-distance block stays within 64 MiB [audit r2-06].
- Spectral phase (opt-in): `_knee_rank` ignores noise knees
  (Marchenko-Pastur envelope, no curvature) and falls back to `r = dim`; the
  oversample default is 50 with `r < dim`. The phase trades RAM for
  rescoring work; it is not a speed-up. Dumps built with the old knee detection
  are still truncated when loaded and must be rebuilt [audit r2-01, r2-02].

**Package, API surface and defaults**

- `m2m.exp_map`, `log_map`, `geodesic_distance`, `project_to_tangent` and
  `normalize_sphere` are the Riemannian implementations of
  `m2m.geometry`. The Euclidean fallbacks that shadowed them were removed
  [audit r3a-10]. There are two classes named `M2MEngine`: `m2m.engine.M2MEngine`
  selects the hardware backend and `m2m.M2MEngine` wraps `M2MMemory`; both
  say so in their docstrings.
- `enable_lsh_fallback` defaults to `False`; the LSH and HNSW indexes are
  experimental [audit r3a-14].
- `python -m m2m` (also installed as the `m2m` script) listens on
  `127.0.0.1:8080` by default; `M2M_HOST`/`M2M_PORT` override it. The
  Dockerfiles set `M2M_HOST=0.0.0.0`. A warning is logged when binding to a
  non-loopback host without `M2M_API_KEY` [audit r3a-18, r3b-09].
- Boltzmann sampling (`EBMExploration.sample_uncertain`) validates the
  temperature (`T < 0` or `NaN` raises `ValueError`) and uses Gumbel-top-k for
  any `T >= 0`, with `T = 0` as the deterministic limit [audit r2-17].

### Fixed

- `search_batch()` and `search_batch_filtered()` score candidates **per query**
  (the rows of that query's probed clusters), not the union of the whole batch;
  probe expansion for filters is decided per query with its own cluster order
  [audit r4-07, r1-12, r1-22]. The LSH path of `search_batch_filtered` no
  longer returns all `-1` [audit r1-07]. Batch reads take the write lock for
  the whole read [audit r1-03]. An empty database returns `(B, 0)` [audit r1-27].
- The filtered path of `search()` ranks with the same normalized query as the
  unfiltered path, no longer builds an `O(n)` list per query and runs under the
  write lock [audit r1-10, r1-16, r1-18]. It no longer asks the index for
  `k + n_bad` results that it then discarded [audit P13].
- `update(vector=)` in LSH mode stores the normalized vector and re-indexes
  the LSH buckets [audit r1-08].
- With `device='cuda'` and no `torch` installed, the engine warns
  (`UserWarning`) and runs on CPU; `m2m.engine.M2MEngine.backend` and
  `get_backend_info()` report the backend actually in use. The log no longer
  claims that Vulkan is enabled after a failed initialization [audit r3a-09].
- HTTP API: a validation `ValueError` from `add`/`update`/`delete` is
  `400` with the message [audit r1-02, r1-19, r1-23]. Requests larger than
  `M2M_MAX_BODY_BYTES` are rejected with `413` before being read; the number of
  vectors is checked before validating the arrays [audit r3a-17]. The API key
  is compared in constant time, `/v1/admin/*` answers `503` when no key is
  configured (fail closed), `/v1/stats` no longer omits collections, and
  `checkpoint`/`backup` return the real error instead of a silent `200`
  [audit r3a-17]. `GET /v1/collections/{name}/suggest` no longer fails with
  `500` when a key is configured.
- LangChain `M2MVectorStore.similarity_search()` fills the requested top-k by
  skipping deleted documents [audit r3b-02].
- Unicode crash on Windows in benchmark scripts [audit r3b-14: only partially
  addressed; some scripts still open files without `encoding=`].
- HRM2 `query_with_details` `IndexError` with `precomputed_embeddings`
  (the same class of bug recurred in `query_batch(lod=2)`; covered by the
  mask alignment of r1-11).

### Added

- `DocResult.alpha` and `DocResult.distance` [audit r1-13].
- `search_batch_filtered(..., filter=dict | callable)` [audit P13].
- `SplatStore.reset()`, `M2MPersistence.save_metadata_many()`,
  `IndexTamperedError` [audit r3a-17].
- `HRM2Config` in `m2m.__all__` [audit r3a-10].
- `m2m.engine.M2MEngine.backend` and `get_backend_info()` [audit r3a-09].
- `m2m` console script (`python -m m2m`), `py.typed`, and the extras
  `core`, `api`, `langchain`, `bench`, `dev`, `all` (plus `vulkan`,
  `embeddings`, `distributed`) as the single source of dependencies;
  `requirements.txt` is `-e .[dev,bench]` [audit r3b-09, r3b-10].
- License declared as the SPDX expression `AGPL-3.0-only` (PEP 639) [audit r3b-09].
- Regression tests `tests/test_audit_*.py` for every audit id above.

### Changed (packaging, CI, deployment)

- `MANIFEST.in` lists only files that exist [audit r3b-09]. `pytest.ini` was
  removed; the single pytest configuration lives in `pyproject.toml`. `.flake8`
  is a real config, bandit reads `[tool.bandit]` via `-c pyproject.toml`,
  pre-commit hooks are pinned to verified revisions [audit r3b-07].
- CI runs lint, test (Python 3.10-3.12, coverage floor 68 %), build, security,
  docs and deploy-docs jobs with pinned action SHAs and minimal permissions
  [audit r3b-08]. `test_lsh_speedup` became a parity test
  (`test_lsh_matches_exact_scan`); the measurement moved to
  `benchmarks/bench_lsh.py` [audit r4 methodology].
- Dockerfiles install the `api` extra, ship the shaders and `py.typed`, and run
  as a non-root user [audit r3b-04, r3b-11].
- `deploy/docker-compose.yml` and `deploy/k8s/` require API keys, bind the
  published ports to `127.0.0.1`, and declare resources, probes and a
  securityContext [audit r3b-11]. These manifests were **not** validated with
  `docker` or `kubectl` (neither is available on the machine that produced
  them).
- `tests/README.md` describes the suite that exists [audit r3a-13].
- Documentation rewritten: `README.md`, `docs/CONFIGURATION.md` (new),
  `docs/API.md`, `docs/PERFORMANCE.md`, `docs/TROUBLESHOOTING.md`,
  `docs/cluster.md`, `SECURITY.md` and this file [audit r3a-01 ... r3a-19,
  r3b-05, r3b-06, r3b-12, r3b-13, r3b-14].

### Removed

- `SimpleVectorDB._search_batch_lsh` (dead after the per-query candidate fix)
  [audit r4-07].

### Known limitations in 2.4.0

See `[Unreleased]` above: vector `GET`/`PUT`/`DELETE` and the legacy `/ingest`
do not validate the API key, the rate limiter is inert, the coordinator does
not forward its key to the edges and has no `/health`, the root
`docker-compose.yml` publishes 8080 without a key, `restore()` does not
validate `..`. None of them is a regression; all are scheduled.
