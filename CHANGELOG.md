# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Tags such as `[audit r1-13]` refer to the findings of the 2026-11 audit of the
code base. Every id either names a test under `tests/` that cites it, or carries
`no unit test` with the reason (documentation, CI, packaging, deployment);
`tests/test_audit_id_index.py` keeps that true.

## [Unreleased]

Nothing pending. The entries that used to be listed here landed before the release and
are part of 2.4.0 — the version below, which is not published yet (it lives on `main`,
not on PyPI).


## [2.4.0] - Unreleased

Version `2.4.0` is set in `pyproject.toml` and `m2m.__version__` [audit r3a-04].
It is **not published on PyPI yet**: the `publish` workflow runs on
`release: published`, so nothing is uploaded until a GitHub Release is created.

### Note on the entries below

This release is the result of a full audit of the code, its claims and its
benchmarks (2026-11). 99 commits (merges excluded) fix the behaviour of the engine, the
persistence layer and the HTTP API; the documentation was rewritten to describe
what the code does. Every behavioural change listed here cites the audit id of
the finding it closes, and each has a regression test that cites the id; ids without a
test say so inline (`no unit test`). Entries of the versions before
2.4.0 are not reproduced here: they quoted benchmark figures the audit could not
sustain and remain in the git history of the README.

**Benchmark figures.** The figures that used to live in the README were measured
before these fixes and with a methodology the audit found flawed (for example
`search_batch` scoring the union of the batch instead of each query's own candidates);
they were dropped. The tables were re-measured on the final code with a fixed protocol
and live in `docs/BENCHMARKS.md`, with the headline numbers in the README and
`docs/PERFORMANCE.md` (`benchmarks/results/*.json`, the run on the commit before the
audit in `before_a74d9e4/`). In one line: a single query on clustered data is
1.99× / 3.39× a NumPy linear scan at 10k / 50k and 1.5× /
2.13× the commit before the audit; in batches (B = 200) the linear scan is as
fast or faster; on uniform data the recall@10 of the approximate path is 0.149 /
0.093.

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
- With a metadata filter the top-k is **exact** in IVF/HRM2 mode: a cluster is skipped
  only when the bound `(dist(q, c) - r_c)**2 >= d2_k` proves it cannot hold a top-k
  document; the per-cluster radii are maintained by `add`/`update` and invalidated on
  rebuild, compaction, feedback drift and MapReduce. Recall@10 is 1.0 on the four
  benchmark filters (before: 0.458–1.000); the unfiltered path
  returns the same ids as before [P17-1; `tests/test_p17_exact_filter.py`].

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
- In LSH mode `_doc_results_from_lsh` evaluates the filter against the real metadata
  also with `include_metadata=False`; it evaluated `{}` and returned an empty list
  [audit P15; `tests/test_audit_closure.py`].

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
- `SimpleVectorDB.feedback(query, relevant_ids=..., irrelevant_ids=...) ->
  FeedbackResult`: Hebbian update by **document id** under the write lock (relevant ids:
  `α`, `κ` up and `μ` towards the query; irrelevant ids: `α`, `κ` down). Unknown or
  soft-deleted ids are reported in `unknown_ids` and not applied; the HRM2 index is kept
  in sync so `search()` sees the drift. `FeedbackResult` is exported [audit r3a-08, P15;
  `tests/test_audit_learning_soc.py`].
- The Hebbian drift re-projects `μ` onto the sphere and rejects a degenerate step:
  after 500 relevant feedbacks `‖μ‖` stays at 1.0000 (it had decayed to 0.073)
  [audit r2-14; `tests/test_audit_learning_soc.py`].
- The learned state (`α`, `κ`, `μ`) survives `save()` + reopen for `AdvancedVectorDB`
  and `SimpleVectorDB`, and is rewritten by `update(vector=)` and `delete(hard=True)`
  (`energy/landscape.npz`) [audit r3a-07, P15; `tests/test_audit_learning_soc.py`,
  `tests/test_audit_closure.py`].

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
- `M2MPersistence.restore(backup_path)` validates its source **before** removing the
  storage: `..` → `ValueError`, missing → `FileNotFoundError`, not a directory →
  `NotADirectoryError`, source overlapping the storage (symlinks resolved) →
  `ValueError`. It used to delete the storage first [audit P15;
  `tests/test_audit_closure.py`].

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
- `HNSWIndex`: `_search_layer` returns candidates closest-first and `_insert` enters
  the next layer from the closest candidate [audit r2-07]; the default level multiplier
  is `1/ln(M)` as in the paper [audit r2-09]. Measured once during the fix pass on the
  developer's machine (N = 2500, not part of `benchmarks/results/`): the fraction of
  nodes above layer 0 went from 0.488-0.499 to 0.054-0.057 and the CPU build time
  dropped by 31% to 48%, at a cost of about 0.8 recall@10 points on uniform data,
  averaged over 5 level seeds (0.265 against 0.274 at `ef_search=32`, 0.417 against
  0.425 at 64) — a cost, not an improvement. The tests assert the structure of the
  index, not these numbers. HNSW and LSH stay experimental and are slower than a linear scan
  (N = 2500: 302 and 123 QPS against 18,000) [audit r2-08; no unit test: the QPS
  comparison is `benchmarks/results/bench_index_types.json`; `tests/test_audit_hnsw_lsh.py`
  covers the structural fixes of r2-03, r2-04 and r2-07].
- `CrossPolytopeLSH` uses `np.random.default_rng(seed)` and a deterministic query
  fallback instead of seeding NumPy's global RNG [audit r2-04], and warns with
  `UserWarning` that the mode is experimental: with the defaults its recall on uniform
  data stays low [audit r2-03; `tests/test_audit_hnsw_lsh.py`].
- SOC engine of `AdvancedVectorDB`: the criticality index is computed over the real
  clusters of the store (11 clusters at N = 120; SUPERCRITICAL is reachable:
  0.905 where the previous metric topped at 0.5649); `relax()` no longer
  raises the free energy (F 14.431 → 13.351) nor moves `μ`; `trigger_avalanche()` and
  `relax()` write to the real `SplatStore` and survive `save()` + reopen [audit r2-16,
  r3a-07; `tests/test_audit_learning_soc.py`; the values are reproduced by
  `bench/bench_learning_soc.py` in `benchmarks/results/bench_learning.json`; 0.5649 and
  0.073 were measured on the pre-fix code].

**SQ8 disk index**

- `SQ8DiskStore.add()` and `build_from_float_npy()` reject `NaN`/`Inf` before
  writing anything [audit r2-05]. `search_batch()` works in chunks so that the
  squared-distance block stays within 64 MiB [audit r2-06].
- Spectral phase (opt-in): `_knee_rank` ignores noise knees
  (Marchenko-Pastur envelope, no curvature) and falls back to `r = dim`; the
  oversample default is 50 with `r < dim`. The phase trades RAM for
  rescoring work; it is not a speed-up. Dumps built with the old knee detection
  are still truncated when loaded and must be rebuilt [audit r2-01, r2-02].
- `SQ8DiskStore.load(dump_dir, float_path=None)` restores the `float32` source of a
  dump built by `build_from_float_npy()`: the path is recorded in `float_source.json`
  when the dump is built; when it cannot be resolved the call raises
  `FileNotFoundError` naming the missing file and the `float_path=` remedy, and a
  source with a different number of rows raises `ValueError` [P17-2;
  `tests/test_sq8_disk_load.py`].
- `SQ8DiskStore.search_batch()` rejects queries with `NaN`/`Inf` (`ValueError`)
  before searching [audit P15; `tests/test_audit_closure.py`].
- On disk the index takes 3.7 MB per 10k vectors of 384 dimensions in full
  mode and 1.2 MB in spectral mode, both with recall@10 = 1.000 at the
  default oversampling (`benchmarks/results/bench_index_types.json`).

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
  `500` when a key is configured.  With `M2M_API_KEY` set, `GET`/`PUT`/`DELETE`
  `/v1/collections/{name}/vectors/{id}` and the read-only `GET /v1/collections`,
  `GET /v1/collections/{name}` and `GET .../stats` validate the key (`401`; the vector
  endpoints answer before the collection lookup, and `PUT`/`DELETE` count toward the
  rate limit) [audit P15; `tests/test_audit_closure.py`].

- LangChain `M2MVectorStore.similarity_search()` fills the requested top-k by
  skipping deleted documents [audit r3b-02].
- Unicode crash on Windows in benchmark scripts [audit r3b-14: only partially
  addressed; some scripts still open files without `encoding=`].
- HRM2 `query_with_details` `IndexError` with `precomputed_embeddings`
  (the same class of bug recurred in `query_batch(lod=2)`; covered by the
  mask alignment of r1-11).

- `POST /v1/collections/{name}/search` with an invalid metadata filter answers `400`;
  it answered `500` (the `ValueError` of the filter validation was not caught)
  [validator D02; `tests/test_p19_api_defects.py`].
- `POST /v1/collections/{name}/query` works on collections without EBM; it answered `500`
  on every call (the fallback called the `/search` handler without the `request`). The
  body field is `vector`. With `include_energy=True` on a collection created with
  `enable_ebm=True` a filter is now rejected with `400` instead of being ignored
  [validator D04; `tests/test_p19_api_defects.py`].
- `M2M_CORS_ORIGINS="https://a, https://b"` (a space after the comma) admits both
  origins; the second one used to be left out [validator D15;
  `tests/test_p19_api_defects.py`].
- The coordinator answers `401`, not `500`, to a non-ASCII API key [validator D19;
  `tests/test_p19_api_defects.py`].
- Documentation: the README example said the nearest neighbour scored about `0.23`; the
  block prints `0.3024` for the top hit (`0.22` is the mean of the ten). `docs/API.md`
  described the body of `/query` and the `400` row incorrectly, promised `metadata`
  without `include_metadata=True` and named the arguments of `HRM2Engine.query` wrong
  [validator D01, D03, D06, D14, D18]. The CHANGELOG claimed a regression test for
  every audit id; the ids without one now say so inline.

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
  `requirements.txt` is `-e .[dev,bench]` [audit r3b-09, r3b-10; no unit test:
  packaging, checked by building the sdist and the wheel and importing from the
  wheel].
- License declared as the SPDX expression `AGPL-3.0-only` (PEP 639) [audit r3b-09].
- `SimpleVectorDB.feedback()` and `FeedbackResult` [audit r3a-08];
  `AdvancedVectorDB.maybe_consolidate()` (`ConsolidationDecision`: consolidates only
  when the criticality index is SUPERCRITICAL, idempotent) and the constructor flag
  `auto_consolidate` (default `False`) [audit r3a-07; `tests/test_audit_learning_soc.py`].
- A real, opt-in rate limit: `M2M_RATE_LIMIT_PER_MIN` per client IP (absent, `0` or
  negative = off); `429` with `Retry-After` above the limit [audit P15;
  `tests/test_audit_closure.py`].
- Coordinator: `GET /health` (no key) and forwarding of `M2M_EDGE_API_KEY` to the
  edges on every fan-out [audit P15; `tests/test_audit_closure.py`].
- `get_stats()['backend']` and `['requested_device']`; `M2MConfig.requested_device`
  and `cuda_unavailable_reason()` [audit P15; `tests/test_audit_closure.py`].
- `SQ8DiskStore.load(dump_dir, float_path=None)` [P17-2; `tests/test_sq8_disk_load.py`].
- `docs/BENCHMARKS.md` (tables, protocol, figure legends) and the re-measured
  `benchmarks/results/*.json`, with the run on the pre-audit commit in
  `before_a74d9e4/`; the figures are generated from those JSON files only.
- Regression tests `tests/test_audit_*.py` for every audit id above.

### Changed (packaging, CI, deployment)

- `MANIFEST.in` lists only files that exist [audit r3b-09]. `pytest.ini` was
  removed; the single pytest configuration lives in `pyproject.toml`. `.flake8`
  is a real config, bandit reads `[tool.bandit]` via `-c pyproject.toml`,
  pre-commit hooks are pinned to verified revisions [audit r3b-07; no unit test: configuration files].
- CI runs lint, test (Python 3.10-3.12, coverage floor 68 %), build, security,
  docs and deploy-docs jobs with pinned action SHAs and minimal permissions
  [audit r3b-08; no unit test: CI workflow]. `test_lsh_speedup` became a parity test
  (`test_lsh_matches_exact_scan`); the measurement moved to
  `benchmarks/bench_lsh.py` [audit r4 methodology].
- Dockerfiles install the `api` extra, ship the shaders and `py.typed`, and run
  as a non-root user [audit r3b-04, r3b-11; no unit test: deployment manifests, not
  validated with `docker` or `kubectl`].
- `deploy/docker-compose.yml` and `deploy/k8s/` require API keys, bind the
  published ports to `127.0.0.1`, and declare resources, probes and a
  securityContext [audit r3b-11]. These manifests were **not** validated with
  `docker` or `kubectl` (neither is available on the machine that produced
  them).
- `tests/README.md` describes the suite that exists [audit r3a-13; no unit test: documentation].
- The root `docker-compose.yml` publishes `8080` on `127.0.0.1` only and passes
  `M2M_API_KEY` / `M2M_RATE_LIMIT_PER_MIN` through to the container;
  `deploy/docker-compose.yml` and `deploy/.env.example` take three keys (`M2M_API_KEY`,
  `M2M_COORDINATOR_API_KEY`, `M2M_EDGE_API_KEY`) [audit P15; `tests/test_audit_closure.py`].
- `md5` in `cluster/sharding.py` and `sha1` in `gate_modulus.py` are annotated
  `usedforsecurity=False`, and bandit no longer skips B324 [audit P15].
- Benchmarks re-measured with a common protocol (`bench/_protocol.py`); the CI
  benchmark job reports informative thresholds and sets no time gates; 15 benchmark
  JSON files without a producer were archived in `bench/history/` [audit r4 methodology;
  pack P9].
- Style pass over `src/` and `tests/` (isort + black with AST equivalence checks;
  flake8 208 → 0 findings, bandit 0): no behaviour change [P16; no unit test: style pass, equivalence of behaviour checked by AST comparison of the imports and by the suite]. The historical
  re-exports of `optimized_api`, `interfaces` and `semantic_memory` are explicit
  [cierre].
- `MANIFEST.in` ships `CONTRIBUTING.md` and `docs/` in the sdist; `pyproject.toml`
  declares PyPI topics and classifiers [P10b; no unit test: documentation and packaging,
  checked by building the sdist and the wheel].
- Documentation rewritten: `README.md`, `docs/CONFIGURATION.md` (new),
  `docs/API.md`, `docs/PERFORMANCE.md`, `docs/TROUBLESHOOTING.md`,
  `docs/cluster.md`, `SECURITY.md` and this file [audit r3a-01, r3a-13, r3a-15, r3a-16,
  r3a-19, r3b-05, r3b-06, r3b-12, r3b-13, r3b-14; no unit test: documentation and
  packaging findings]. An independent validator then checked the rewritten text
  claim by claim against the code, the tests and the benchmark JSON (585 checks,
  20 defects: 4 false, 7 not traceable, 2 misleading, 7 minor); the defects found in
  the text are closed in this release, and what it could not verify without
  `docker`/`kubectl` or the `embeddings` extra is labelled as such in the documentation.

### Removed

- `SimpleVectorDB._search_batch_lsh` (dead after the per-query candidate fix)
  [audit r4-07].

### Known limitations in 2.4.0

- Without a filter, a search does not widen its probe to replace soft-deleted
  documents: in HRM2 mode (`n > 3000`) `search_batch()` can return fewer than `k`
  positions (`-1`) and `search()` fewer than `k` results when most documents of the
  probed clusters are deleted. A filter widens the probe and is exact
  (`tests/test_audit_batch.py` covers the exclusion and the backfill; the missing
  widening was confirmed by a probe, there is no dedicated test for it).
- The unfiltered path is approximate on data without clusters (recall@10 0.149 at
  10k, 0.093 at 50k on uniform data) and there is no exact-scan switch; see
  `docs/PERFORMANCE.md`.
- HNSW and LSH are experimental and slower than a linear scan.
- No TLS, no user management, no encryption at rest, no `/metrics` endpoint.
- The filtered batch path still probes with `expand=True`, slower than the
  unfiltered one [audit P15, item left open on purpose].
- Unicode on Windows in benchmark scripts is only partially addressed
  [audit r3b-14; no unit test: benchmark scripts]; the deployment manifests were not validated with `docker` or
  `kubectl`.
