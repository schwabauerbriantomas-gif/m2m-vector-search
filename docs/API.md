# API Reference

Welcome to the M2M Vector Search API reference. 

## Core Interfaces

- **`SimpleVectorDB(device='cpu')`**: Lightweight interface for basic vector search and retrieval. Good for edge devices.
  - `db.add(vectors)`: Add a batch of vectors.
  - `db.search(queries, k)`: Retrieve top-k nearest neighbors.
  - `db.save(path)` / `db.load(path)`: Persist to disk.

- **`AdvancedVectorDB(device='vulkan')`**: Advanced interface with memory tiering and generative features.
  - `db.add(vectors)`: Add vectors to hot memory.
  - `db.search(queries, k)`: Search across tiers.
  - `db.generate(query, n_steps)`: Run Langevin dynamics exploration.
  - `db.consolidate(threshold)`: Clean up passive memory.

For full type stubs and internal methods, please refer to the docstrings in the `src/m2m/` directory.
