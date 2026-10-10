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

- `exp_h8_filtered.py` — H8: pre-filtro sobre candidatos (ACORN-lite).
  **Post-filter tenia recall 0.26 (selectividad 25%); el pre-filter da 1.0000
  en 5/25/50% de selectividad a 2.5-3.2k QPS** (`h8_filtered.json`).
- `exp_h10_sq8.py` — H10: SQ8 + rescoring. **Validada: recall 1.0000 igual a
  float32, QPS comparable (630-720 vs 693), 4x menos RAM (76.8→19.2MB)**
  (`h10_sq8.json`). Candidato a store comprimido para datasets >RAM.
- `kmeans_bs_ab.json` — batch_size del KMeans: 10000→1000 = 3.8x build con
  igual recall (default cambiado).
- `exp_h11_rabitq.py` — H11: RaBitQ (arXiv:2405.12497) en NumPy puro.
  **REFUTADA en este runtime**: estimador verificado (corr 0.966 vs d2 real,
  sanity √(2/π) ✓) pero fase-1 recall 0.46 vs SQ8 0.97; e2e 0.95 @ 388 QPS
  vs float32 1.0 @ 693 y SQ8 1.0 @ 630. La ventaja de RaBitQ es popcount
  SIMD; sin C/SIMD, SQ8 domina. Packed 2.4MB vs 76.8MB (32x) — el uso es
  memoria-extrema, no velocidad (`h11_rabitq.json`, con lecciones de la
  implementación: frame rotado, q residual por-cluster, ensamblado d2).
- `exp_h13_sq8disk.py` — H13: `SQ8DiskStore` (H10 integrado como store real).
  SQ8 en RAM + float32 en disco (mmap) para rescoring. **600k x 384 sin
  normalizar: recall 1.0 con 3.9x menos RAM (921.6 -> 235.2MB de arrays),
  2.2x menos QPS (27 vs 60)** — el tradeoff del modo comprimido, opt-in
  (`src/m2m/sq8_disk.py`, 6 tests nuevos; suite 433 passed).
  Protocolo: dataset pre-generado en disco, cada escenario en subprocesso
  con maxrss aislado, GT exacto precomputado. Lecciones atrapadas por tests:
  rescoring debe ser d2 completa (la similitud 2-2<x,q> sesga sin
  normalizar); GT incremental por chunks corrompe indices (d2 completo
  preasignado es trivial y correcto).
