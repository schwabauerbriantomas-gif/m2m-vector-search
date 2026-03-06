# Performance Guide

M2M Vector Search leverages Vulkan and modern multi-threading to achieve high performance. To get the best out of the engine, consider the following optimization strategies:

## 1. Input Dimensionality and Distribution

- **Dimensionality**: We recommend an embedding dimension between `384` and `1536` (e.g., outputs from standard embedding models like `all-MiniLM` or OpenAI embeddings).
- **Distribution**: HRM2 clustering heavily relies on distinct semantic groupings. If you supply purely random homogeneous noise, the K-Means index struggles to create distinct boundaries, dropping your speedup from $O(\sqrt{N})$ down to near $O(N)$ linear scan speeds. 

## 2. Platform considerations

- **Vulkan**: To use `device='vulkan'`, ensure you have the Vulkan SDK installed and up-to-date drivers. If Vulkan fails to initialize, the engine falls back to a highly-optimized CPU pathway.
- **Batching**: Always batch your `add` and `search` queries. Submitting 10,000 queries in a single matrix is much faster than running 10,000 separate `search()` calls due to memory overhead.

## 3. Tiered Memory (Advanced)

If you have vectors that are rarely used, let `SOC` (Self-Organized Criticality) push them to RAM or Disk. Keeping only your most recently accessed "hot" splats in `VRAM` keeps latency under 1ms.
