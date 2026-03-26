# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- HRM2 query_with_details IndexError with precomputed_embeddings
- Unicode crash on Windows in benchmark scripts

### Changed
- Repo cleanup, test suite expanded 53→352

## [2.1.0] - 2026-03-17

### Added
- Security audit, pentest report, and chaos testing reports
- Optimization analysis with concrete patches
- GitHub issue and PR templates
- CUDA backend support (NVIDIA GPUs)
- Transformed backend with pre-computed index
- Advanced cluster features: semantic/geo sharding, load balancing
- Offline sync queue for edge nodes
- LSH (Locality-Sensitive Hashing) integration
- Entity extractor with n-gram analysis and semantic validation
- LangChain retriever interface
- Full CRUD API with collection management
- EBM (Energy-Based Model) features: energy computation, exploration suggestions
- SOC (Self-Organized Criticality) for memory consolidation
- Query optimizer with LRU cache
- Comprehensive test suite (415 tests, 100% pass rate)

### Changed
- Improved HRM2 engine with batch query support
- Refactored storage layer with WAL (Write-Ahead Log)
- Enhanced edge API with v1 endpoints

## [2.0.0] - 2026-02-24

### Added
- Initial open-source release
- Gaussian Splat-based vector representation
- HRM2 (Hierarchical Routing with Mixture Models) engine
- CPU and Vulkan backends
- Edge/Coordinator cluster architecture
- SimpleVectorDB and AdvancedVectorDB interfaces
- Persistence layer with checkpoint/backup

### Known Limitations
- M2M is slower than linear search for N ≤ 10K (see benchmark_stats.md)
- Vulkan backend overhead on NVIDIA GPUs
- Cluster communication without encryption
- No authentication on REST APIs
