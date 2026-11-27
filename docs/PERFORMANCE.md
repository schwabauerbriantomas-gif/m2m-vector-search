# Performance

<!-- P10b: cifras -->
This page explains **what** determines the speed and recall of M2M and **how** to
measure it. It deliberately quotes no benchmark figures: the numbers that used to
be here were produced before the 2026-11 fixes (per-query batch candidates, cosine
scores, filter-mask cache, spectral knee detection) and are being re-measured with
the protocol of the benchmark pack. They will be added here once that run exists.

## What you are measuring

- **`search()`** routes the query to `n_probe` coarse clusters (HRM2), scores the
  splats in them and returns the top-k. With `n <= 3000` the batch path degenerates
  to an exact scan, which is also fast. Recall is therefore the recall of an IVF
  index with that `n_probe`: it rises with `n_probe` and falls with `k`.
- **`search_batch()`** scores each query against the rows of **its own** probed
  clusters (not the union of the batch, as versions before 2.4.0 did by accident).
  On **clustered** data the batch and the single-query path agree. On **uniform**
  data (random vectors) the batch recall equals the single-query IVF recall, which
  is low with the default `n_probe`; it is not `1.0`. A benchmark on uniform data
  measures `n_probe`, not the kernel.
- **Filters** (`search(filter=...)`, `search_batch_filtered(filter=...)`) are
  applied before the top-k, with per-query probe expansion when too few allowed
  documents are found. The filter mask is cached per metadata version, so the
  second query with the same filter costs a cache hit; the first evaluation after a
  write is `O(n)` in Python.
- **`add()`** appends splats and attaches them to the existing index; the IVF index
  is retrained only after substantial growth. `delete(hard=True)` compacts and
  rebuilds the index (`O(n)` per call); soft deletes do not.

## Knobs

| Knob | Where | Effect |
|---|---|---|
| `n_probe` | `SimpleVectorDB(n_probe=...)`, `HRM2Config` | More probes: higher recall, proportionally more work. The main knob. |
| `adaptive_probing` | `HRM2Config` | Adds probes for a query whose first probes yield too few candidates. |
| `lod` | `HRM2Engine.query(q, k, lod=...)` | `0` and `1` scan less and return the nearest splats of what they probed; recall is lower than `lod=2`. |
| `k_selection="auto"`, `gate_enabled=True` | `SimpleVectorDB` | Chooses the number of coarse clusters with the gate of modulus instead of a fixed `n_coarse`. |
| `enable_lsh_fallback` | `SimpleVectorDB` | Cross-Polytope LSH for data without clusters. Experimental: it is parity-tested against an exact scan, not tuned for throughput. |
| `HNSWIndex` | `m2m.hnsw_index` | Pure-Python HNSW. Experimental; defaults may change. <!-- P10b: LSH/HNSW --> |
| `SQ8DiskStore(spectral_r=...)` | `m2m.sq8_disk` | Smaller codes (less RAM) in exchange for more rescoring per query. Not a speed-up. |

## Data distribution matters

HRM2 assumes the data has clusters. On embeddings from real text this holds; on
isotropic random data (`np.random.randn`) KMeans finds no structure, the routing
degenerates towards a scan, and the energy of every result is the OOD cap. Use
clustered data or real embeddings to evaluate the engine, and say which one you
used when you report a number.

## Batching

`search_batch(Q)` amortizes the per-query Python overhead. The win is largest for
large batches; for a single query `search()` is the right call. Keep `B * n`
(distances block) within memory: the SQ8 path chunks its work so that the block
stays below 64 MiB.

## GPU backends

`device="cuda"` needs the `embeddings` extra (`torch`); `device="vulkan"` needs the
`vulkan` extra plus the `glslc` shader compiler from the Vulkan SDK at run time.
When a backend is missing the engine falls back to CPU and says so:
`db.engine.backend` and `db.engine.get_backend_info()` report what is actually
running. GPU execution is not exercised in CI (no GPU on the runners).

## Measuring

- Pin BLAS threads (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`) on small
  matrices: with the default thread count, small operations spin-wait and run
  several times slower.
- Fix seeds, compute exact ground truth in `float64`, report recall and QPS
  together, and state the hardware and the load.
- `bench/bench_baseline.py` and `benchmarks/bench_lsh.py` are the scripts kept for
  re-measurement; the `bench/*.json` files are historical.
