# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-03-13

### Added

- **Premium Visualization Assets**: Replaced legacy charts with 6 ultra-premium, dark-themed dashboard assets:
  - `performance_overview.png`: Metric dashboard (QPS, Latency, Ingestion, Memory).
  - `throughput_comparison.png`: 2.18x speedup visualization vs. Linear Scan.
  - `latency_distribution.png`: Detailed percentile breakdown.
  - `architecture_overview.png`: 4-layer glassmorphism diagram.
  - `dataset_statistics.png`: Vector dimensionality and memory profile.
  - `test_coverage.png`: CI/CD status with 100% pass rate.

### Changed

- **README Documentation**:
  - Validated and updated all performance metrics with real DBpedia 1M dataset results.
  - Corrected license badges and mentions from Apache 2.0 to **GNU AGPLv3** to match the repository's `LICENSE` file.
- **CI/CD & Code Quality**:
  - Fixed Ruff linting errors (`F541` f-string without placeholders) across the test suite.
  - Applied strict `black` and `isort` formatting to `test_complete_validated.py` for pipeline compliance.

### Removed

- Deleted 7 obsolete visualization assets from the `assets/` directory to reduce repository size.

## [2.0.0] - 2026-03-09

### Added

- **Full CRUD Operations** (`SimpleVectorDB` & `AdvancedVectorDB`):
  - `add()` with explicit `ids`, `metadata`, `documents` and auto-ID generation. Backward-compatible with positional `add(vectors)` call.
  - `update()` to patch vector, metadata or document text; supports `upsert`.
  - `delete()` with soft-delete and hard-delete by `id`, list of `ids`, or metadata `filter` (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`).
  - `search()` returns `List[DocResult]` (with filter support and deleted-doc exclusion) or legacy tuple for backward compat.
- **Persistence Layer** (`storage/`):
  - `WriteAheadLog` — durable operation logging with `msgpack`/JSON serialization, auto-sync, and checkpoint/replay.
  - `M2MPersistence` — layered storage combining NumPy vector shards, SQLite metadata, pickle index, and WAL integration. Includes backup/restore.
- **Energy-Based Model (EBM) Features** (`ebm/`):
  - `EBMEnergy` — computes energy `E(x)`, gradient, free energy, and local 2D energy maps.
  - `EBMExploration` — identifies high-uncertainty regions, Boltzmann-weighted sampling, agent exploration suggestions.
  - `SOCEngine` — Self-Organized Criticality: criticality detection, BFS cascade avalanche triggering, system relaxation.
  - `SimpleVectorDB` gains: `search_with_energy()`, `get_energy()`, `suggest_exploration()`, `find_knowledge_gaps()`.
  - `AdvancedVectorDB` gains: `check_criticality()`, `trigger_avalanche()`, `relax()`.
- **REST API v2** (`api/edge_api.py`) — collections-based architecture:
  - Full CRUD endpoints: `POST/GET/DELETE /v1/collections/{name}`, `POST/PUT/DELETE /v1/collections/{name}/vectors/{id}`.
  - Search with metadata filters: `POST /v1/collections/{name}/search`.
  - EBM endpoints: `/energy`, `/explore`, `/suggest`.
  - Admin endpoints: `/v1/admin/checkpoint`, `/v1/admin/backup`.
  - Legacy `/ingest` and `/search` endpoints retained for backward compatibility.
- **EnergyRouter** (`cluster/router.py`) — optional energy-based routing for distributed clusters:
  - Five strategies: `energy_balanced`, `round_robin`, `least_loaded`, `locality_aware`, `hybrid`.
  - TTL-based energy cache, routing statistics, and Boltzmann probabilistic node selection.
- **Test Suite** (`tests/test_crud.py`) — 32 new tests covering CRUD, filters, EBM features, and SOC.
- Added `msgpack>=1.0.0` as a required dependency.

### Changed

- `pyproject.toml` version bumped `1.5.0 → 2.0.0`.
- `search()` default changed to `include_metadata=False` for legacy tuple compatibility.
- `ClusterRouter` now optionally wraps `EnergyRouter`; existing API unchanged when energy router is disabled.
- Removed `vulkan>=1.3.0` from required dependencies (now optional).

### Fixed

- `float(alpha[i])` `TypeError` when `alpha[i]` is a multi-dimensional NumPy array.
- `condition` undefined variable in `_match_filter` (renamed to `cond`).
- `add(np_array)` positional first-arg call no longer raises `ValueError`.
- LSH path in `search()` now correctly builds `DocResult` list when `include_metadata=True`.
- `edge_node.py` handles both tuple and list returns from `search()`.

## [1.5.0] - 2026-03-08
### Added
- **M2M Native Entity Extractor**: Integrated zero-dependency deterministic entity extraction.
- **Gaussian Graph Store**: Extended HRM2 to support knowledge graphs via `NodeType` and `GraphEdge` structures. Combined with `M2MGraphEntityExtractor` for automated relation extraction.

## [1.4.0] - 2026-03-08
### Added
- **M2M Cluster Operations (Phase 4)**: Introduced containerization and cloud orchestration tools.
  - Added `deploy/Dockerfile` and `deploy/docker-compose.yml` for unified local cluster testing.
  - Added Kubernetes manifests in `deploy/k8s/` mapping out scalable deployments and services.
  - Provided `monitoring/prometheus.yml` targets for cluster scraping.
  - Created an official `docs/cluster.md` architectural deployment guide.

## [1.3.0] - 2026-03-08
### Added
- **M2M Cluster Advanced Features (Phase 3)**: Introduced powerful new intelligence edge logic.
  - Added semantic and geo-aware sharding implementations mapping Vectors via KMeans Centroids and Haversine distance, respectively.
  - Decoupled Coordinator load balancing into `balancer.py` introducing Least-Latency round robin logic.
  - Integrated `asyncio` base Edge Node offline queuing: Nodes automatically cache synchronization intents if the main swarm coordinator goes offline and flush payloads automatically.

## [1.2.0] - 2026-03-07
### Added
- **M2M Cluster Communication Layer (Phase 2)**: Transformed the local cluster architecture into a fully distributed microservices network via HTTP/REST.
  - Added `src/m2m/api/` with `FastAPI` and `uvicorn` web servers for edge and coordinator networking.
  - Added `protocol.py` defining cluster-wide `Pydantic` schema models.
  - Refactored `M2MClusterClient` to issue `requests.post()` async API calls instead of internal python function mapping.
  - Implemented automatic network failover (e.g. querying fallback edges directly if coordinator times out).

## [1.1.0] - 2026-03-07
### Added
- **M2M Cluster Architecture (Phase 1)**: Introduced distributed vector networking for horizontal scalability and high availability.
  - Implemented `EdgeNode` wrapping `SimpleVectorDB` for local-first operations.
  - Added `ClusterRouter` for query coordination and data locality tracking logic.
  - Enabled multi-node result merging via `ResultAggregator` using Reciprocal Rank Fusion (RRF).
  - Designed `M2MClusterClient` to manage failover and query orchestration across the cluster.
  - Built geographic, hash, and semantic sharding strategies for dynamic data routing.

## [1.0.8] - 2026-03-07
### Added
- Added `M2MDatasetTransformer` support and documentation for optimal format ingestion.
- Added comprehensive explanation of LSH integration for homogeneous distributions in `README.md`.
- Updated benchmark configurations for heterogeneous distributions with clustering comparisons.

### Changed
- Improved overall latency and throughput (Transformed datasets achieve ~8.68ms vs 47.80ms Linear baseline on Edge tier hardware).
- Fixed relative import issues in the benchmark logic (`dataset_transformer` and `splat_types`).

## [1.0.5] - 2026-03-05
### Added
- Explicit PyPI distribution setup (`pyproject.toml`, `MANIFEST.in`).
- Initial implementation of the 3-Tier Memory System (VRAM -> RAM -> SSD).
- Support for LangChain and LlamaIndex integrations.
- GitHub Actions CI pipeline.
- Improved documentation with Architecture diagrams and troubleshooting guides.

### Changed
- Migrated codebase to standard `src/m2m` structure.
- Cleaned up obsolete `.test_env` and cached build artifacts.
- Removed PyTorch dependencies in favor of pure NumPy / Vulkan.
