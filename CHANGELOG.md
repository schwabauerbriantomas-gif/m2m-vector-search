# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
