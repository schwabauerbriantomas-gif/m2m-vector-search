# Benchmarks y experimentos


> **P9 — benchmarks reproducibles:** las tablas de este README que citan cifras viejas (H7–H13)
> quedan reemplazadas por las mediciónes de `bench/bench_search.py` (tabla 1: single/batch),
> `bench/bench_filtered.py` (tabla 2: filtros), `bench/bench_index_types.py` (tabla 3: lod/SQ8/
> HNSW/LSH/gate) y `bench/bench_storage.py` (tabla 4), todos con el protocolo común
> `bench/_protocol.py` (datasets con semilla, recall@10 vs fuerza bruta float64, QPS por repetición,
> hilos=1, meta con commit/uptime). Resultados: `benchmarks/results/*.json` y la columna "before"
> medida sobre el commit base en `benchmarks/results/before_a74d9e4/`. Los JSON históricos sin
> productor viven en `bench/history/`. Ejemplo:
> `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/bench_search.py`.


- `bench_baseline.py` — QPS/recall del pipeline público vs linear scan (seed fija).
- `bench/bench_exact_uniform.py` — opciones que devuelven el top-10 exacto en datos uniformes
  (10k × 384): default vs `n_probe=<n_coarse>` (leído del índice) vs filtro acepta-todo (camino
  exacto P17) vs escaneo lineal. Resultado: `benchmarks/results/bench_exact_uniform.json`
  (tabla «What gives exact results» de `docs/PERFORMANCE.md`).
- `bench/bench_noise.py` — no mide: compara fila a fila dos corridas archivadas de
  `bench_search.py` (`bench_search_run1_2109459.json` contra `bench_search.json`) →
  `benchmarks/results/bench_noise.json` (sección «Noise between runs» de `docs/PERFORMANCE.md`).
- `bench/bench_learning_soc.py` — no mide tiempo: reproduce con setups deterministas las cifras
  de aprendizaje online y SOC del README (feedback, ‖μ‖, criticidad, relax, maybe_consolidate) →
  `benchmarks/results/bench_learning.json`.
- `bench/history/final_ab.json` — A/B final: single vs search_batch (quieto, misma corrida).
- `expA_ncoarse.py — grid n_coarse x n_probe (monkeypatch, sin tocar código).
- `exp_h9_binary.py` — H9 "RaBitQ light": binarización 1-bit + rescoring.
  **Resultado: REFUTADA** en este dataset (50k x 384 clustered): recall 0.41 @8x
  oversampling a 73 QPS vs 658 QPS recall 1.0 float32 IVF (ver `bench/history/h9_binary.json`).

- `exp_h7_churn.py` — H7: churn 10%/lote sobre 20k (protocolo IP-DiskANN).
  **Resultado: con attach incremental, add ~0.05s/lote (antes 6-8s de KMeans
  por add) y recall@10=1.0000 constante hasta 30% de churn.** Los centroides
  se re-entrenan solo al crecer >25% desde el último entrenamiento
  (`build_index(retrain=True)` fuerza re-entrenar). `bench/history/h7_attach_ab.json` = A/B.

- `exp_h8_filtered.py` — H8: pre-filtro sobre candidatos (ACORN-lite).
  **Post-filter tenia recall 0.26 (selectividad 25%); el pre-filter da 1.0000
  en 5/25/50% de selectividad a 2.5-3.2k QPS** (`bench/history/h8_filtered.json`).
- `exp_h10_sq8.py` — H10: SQ8 + rescoring. **Validada: recall 1.0000 igual a
  float32, QPS comparable (630-720 vs 693), 4x menos RAM (76.8→19.2MB)**
  (`bench/history/h10_sq8.json`). Candidato a store comprimido para datasets >RAM.
- `bench/history/kmeans_bs_ab.json` — batch_size del KMeans: 10000→1000 = 3.8x build con
  igual recall (default cambiado).
- `exp_h11_rabitq.py` — H11: RaBitQ (arXiv:2405.12497) en NumPy puro.
  **REFUTADA en este runtime**: estimador verificado (corr 0.966 vs d2 real,
  sanity √(2/π) ✓) pero fase-1 recall 0.46 vs SQ8 0.97; e2e 0.95 @ 388 QPS
  vs float32 1.0 @ 693 y SQ8 1.0 @ 630. La ventaja de RaBitQ es popcount
  SIMD; sin C/SIMD, SQ8 domina. Packed 2.4MB vs 76.8MB (32x) — el uso es
  memoria-extrema, no velocidad (`bench/history/h11_rabitq.json`, con lecciones de la
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
  **H13b (correccion):** el float32 del rescoring son los ORIGINALES
  entregados a `add()`, no dequantizados desde SQ8. En la version original,
  el camino `add()+flush()` guardaba float32 dequantizado: el rescoring no
  aportaba informacion nueva y el recall quedaba clavado en el techo SQ8
  (~0.97) sin importar el oversample — auditado externamente (recall 0.967
  fijo en ov=2/4/8 a 100k x 384, queries in/out-dataset); el camino
  `build_from_float_npy` siempre fue correcto (recall 1.0 verificado).
  Fix: spill incremental de originales a disco (append por add, RAM
  acotada), rebuild con escritura .tmp + rename atomico, y merge real en
  `load()`/`build_from_float_npy` + `add()` (el dump re-materializado
  conserva los originales de ambas bases). Verificacion post-fix: recall
  1.0000 in/out-dataset a 100k x 384 estable en ov=2/4/8, sin NaN en filas
  constantes. 9 tests (3 nuevos atrapan la regresion: fallan contra el
  modulo viejo, pasan contra el parche).
