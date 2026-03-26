# M2M Vector Search — Lessons Learned

## EnergyFunction Stubs — FIXED ⏱️
- **Date**: 2026-03-19
- **Problem**: `energy.py` had 3 functions (E_splats, E_geom, E_comp) that returned `np.zeros()`. SOC consolidation was non-functional.
- **Impact**: `compute_energy()` always returned 0. SOC avalanches triggered by variance only, not energy.
- **Fix**: Implemented real energy: E_splats = negative log-density of Gaussian mixture, E_geom = sphere deviation penalty
- **Commit**: `5884436`

## config.temperature Bug — FIXED ⏱️
- **Date**: 2026-03-19
- **Problem**: `__init__.py` referenced `self.config.temperature` which doesn't exist. Should be `self.config.global_temperature`.
- **Fix**: Changed to `self.config.global_temperature`
- **Commit**: `5884436`

## Silhouette Lazy Computation — FAILED ⏱️
- **Date**: 2026-03-19
- **Problem**: Attempted to make silhouette computation lazy (every 500 adds instead of every add). Caused 2 test failures.
- **Cause**: `_use_lsh` flag was only reset inside the silhouette check block. Skipping the check meant the flag stayed True from a previous test.
- **Lesson**: When adding conditional caching, ensure all flags are reset unconditionally, not inside the cached path.
- **Resolution**: Reverted. Overhead is negligible since silhouette only runs when `enable_lsh_fallback=True` (off by default).

## BOM Character — FIXED
- **Date**: Prior to 2026-03-19
- **Problem**: `gpu_vector_index.py` had UTF-8 BOM (0xEF 0xBB 0xBF)
- **Fix**: Re-encoded without BOM

## License/Version Inconsistency — FIXED ⏱️
- **Date**: 2026-03-19
- **Problem**: README badge said Apache 2.0, SECURITY.md said version 2.0.0, __init__.py docstring said version 1.0.0
- **Fix**: Unified to AGPL-3.0, version 2.1.0 everywhere
- **Source of truth**: `pyproject.toml`
- **Commit**: `5884436`
