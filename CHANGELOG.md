# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
