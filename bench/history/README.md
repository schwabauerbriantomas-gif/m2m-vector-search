# bench/history — corridas históricas (sin productor reproducible)

JSON de benchmarks viejos movidos acá en el pack P9. **No son fuente de ninguna figura ni tabla
del README**: salen de corridas con metodología distinta (host cargado, sin protocolo común, en
algunos casos con el "before" agregado a mano — ver auditoría r4/r5) y no hay un script que los
regenere tal cual. Se conservan como registro histórico.

Las cifras vigentes están en `benchmarks/results/*.json` (rama) y
`benchmarks/results/before_a74d9e4/*.json` (commit base, medido con los mismos scripts), generados
por `bench/bench_*.py` con el protocolo `bench/_protocol.py`, y resumidos en
`docs/bench-manifest.json`. Los scripts `benchmark_final.py`, `benchmark_supplement.py` y
`benchmark_ann_standard.py` (si se re-corren) escriben sus salidas acá.
