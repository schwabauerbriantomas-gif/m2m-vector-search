# Troubleshooting Guide

This guide will help you resolve common issues when using M2M Vector Search.

## 1. Vulkan Device Not Found

**Issue**: You receive `RuntimeError: Vulkan device not found` when initializing `AdvancedVectorDB(device='vulkan')`.
**Solution**:
- Ensure you have the Vulkan SDK installed.
- Ensure your GPU drivers are up-to-date and natively support Vulkan > 1.0.
- Fallback: Change `device='vulkan'` to `device='cpu'` in your initialization.

## 2. Slow Performance / 0.3x Speedup

**Issue**: Benchmarks or queries run slower than standard linear scans.
**Solution**:
- This usually indicates the underlying `sklearn` fallback dataset is too homogeneous. M2M clusters and hierarchical routing benefit significantly from datasets with clear semantic structures (such as text embeddings from real-world documents). 
- Verify you are submitting **batches** of embeddings or queries. Doing 1,000 queries sequentially one-by-one introduces high loop overhead compared to an explicitly batched tensor `np.random.randn(1000, 640)`.

## 3. "No Module Named '__init__'"

**Issue**: Import errors when running from source.
**Solution**:
- Since the package is structured under `src/m2m/`, make sure you install in editable mode `pip install -e .` or set your `PYTHONPATH` correctly if running scripts directly.
