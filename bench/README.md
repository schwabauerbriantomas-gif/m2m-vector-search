# Benchmarks y experimentos

- `bench_baseline.py` — QPS/recall del pipeline público vs linear scan (seed fija).
- `final_ab.json` — A/B final: single vs search_batch (quieto, misma corrida).
- `expA_ncoarse.py — grid n_coarse x n_probe (monkeypatch, sin tocar código).
- `exp_h9_binary.py` — H9 "RaBitQ light": binarización 1-bit + rescoring.
  **Resultado: REFUTADA** en este dataset (50k x 384 clustered): recall 0.41 @8x
  oversampling a 73 QPS vs 658 QPS recall 1.0 float32 IVF (ver `h9_binary.json`).

- `exp_h7_churn.py` — H7: churn 10%/lote sobre 20k (protocolo IP-DiskANN).
  **Resultado: con attach incremental, add ~0.05s/lote (antes 6-8s de KMeans
  por add) y recall@10=1.0000 constante hasta 30% de churn.** Los centroides
  se re-entrenan solo al crecer >25% desde el último entrenamiento
  (`build_index(retrain=True)` fuerza re-entrenar). `h7_attach_ab.json` = A/B.
