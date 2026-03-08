# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
