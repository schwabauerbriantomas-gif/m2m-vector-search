# Benchmarks y experimentos

- `bench_baseline.py` — QPS/recall del pipeline público vs linear scan (seed fija).
- `final_ab.json` — A/B final: single vs search_batch (quieto, misma corrida).
- `expA_ncoarse.py — grid n_coarse x n_probe (monkeypatch, sin tocar código).
- `exp_h9_binary.py` — H9 "RaBitQ light": binarización 1-bit + rescoring.
  **Resultado: REFUTADA** en este dataset (50k x 384 clustered): recall 0.41 @8x
  oversampling a 73 QPS vs 658 QPS recall 1.0 float32 IVF (ver `h9_binary.json`).
