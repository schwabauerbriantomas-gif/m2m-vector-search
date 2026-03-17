# 📊 Análisis Estadístico de Benchmarks

**Generado:** 2026-03-17  
**Analista:** MASFactory Data Analyst  
**Fuentes:** `benchmarks/results/*.json` (13 archivos)

---

## Resumen Ejecutivo

Se analizaron **13 benchmarks** ejecutados entre 2026-02-24 y 2026-03-17 en dos sistemas diferentes. Los datos revelan patrones claros de rendimiento y áreas críticas de mejora.

---

## 1. Configuraciones de Hardware

| Sistema | CPU | RAM | GPU | Python |
|---------|-----|-----|-----|--------|
| **Dev (AMD)** | Ryzen 5 3400G (4C/8T) | 16 GB | AMD RX 6650 XT → NVIDIA RTX 3090 | 3.12.10 |
| **Edge (Intel)** | Intel Core (2C/4T) | 2 GB | Intel UHD 615 | 3.11.8 |

---

## 2. Benchmarks Exitosos (con retrieval completo)

### 2.1 Dataset: 5,000 splats (k=10, dim=640)

| Benchmark | Backend | Latencia Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | QPS | Speedup vs Linear |
|-----------|---------|--------------------|----------|----------|----------|-----|-------------------|
| 20260224_1748 | **Linear** | 5.25 | 5.01 | 5.80 | 6.98 | 190.30 | 1.0x |
| 20260224_1748 | CPU | 13.97 | 13.87 | 17.47 | 19.11 | 71.58 | **0.37x** (más lento) |
| 20260224_1748 | Vulkan (AMD) | 19.92 | 19.64 | 25.92 | 28.76 | 50.19 | **0.26x** (más lento) |

### 2.2 Dataset: 10,000 splats (k=10, dim=640) - AMD System

| Benchmark | Backend | Latencia Avg (ms) | P50 (ms) | P95 (ms) | QPS | Speedup vs Linear |
|-----------|---------|--------------------|----------|----------|-----|-------------------|
| 20260224_1903 | **Linear** | 9.80 | 9.67 | 10.70 | 102.04 | 1.0x |
| 20260224_1903 | CPU | 23.96 | 23.73 | 27.28 | 41.73 | **0.41x** |
| 20260224_1903 | Vulkan (AMD) | 34.86 | 34.30 | 40.46 | 28.69 | **0.28x** |
| 20260225_0635 | CPU | 22.83 | 22.79 | 29.26 | 43.80 | **0.45x** |
| 20260225_0635 | Vulkan (AMD) | 33.10 | 32.76 | 41.66 | 30.21 | **0.30x** |

### 2.3 Dataset: 10,000 splats (k=10, dim=640) - RTX 3090 System

| Benchmark | Backend | Latencia Avg (ms) | P50 (ms) | QPS | Speedup vs Linear |
|-----------|---------|--------------------|----------|-----|-------------------|
| 20260317_1258 | **Linear** | 24.55 | 24.87 | 40.74 | 1.0x |
| 20260317_1258 | CUDA (RTX 3090) | 25.98 | 25.54 | 38.49 | **0.96x** |
| 20260317_1300 | **Linear** | 24.21 | 24.78 | 41.31 | 1.0x |
| 20260317_1300 | CPU | 32.93 | 33.09 | 30.37 | **0.74x** |
| 20260317_1300 | Vulkan (RTX 3090) | 32.78 | 33.01 | 30.51 | **0.74x** |
| 20260317_1300 | CUDA (RTX 3090) | 26.54 | 25.90 | 37.68 | **0.91x** |

### 2.4 Dataset: 10,000 splats - Intel Edge (2 GB RAM)

| Benchmark | Backend | Latencia Avg (ms) | P50 (ms) | QPS | Speedup vs Linear |
|-----------|---------|--------------------|----------|-----|-------------------|
| 20260305_0102 | **Linear** | 30.06 | 29.18 | 33.26 | 1.0x |
| 20260305_0102 | CPU | 89.25 | 72.67 | 11.20 | **0.34x** |
| 20260305_0102 | Vulkan (Intel) | 51.88 | 50.71 | 19.28 | **0.58x** |
| 20260305_0102 | **Transformed** | **6.66** | **6.54** | **150.19** | **4.51x** ✅ |

---

## 3. Hallazgos Clave

### ⚠️ Hallazgo 1: M2M ES MÁS LENTO QUE LINEAR EN TODOS LOS BACKENDS (N=10K)

**Este es el problema más crítico.** Para datasets de 10K splats con k=10:

- **CPU**: 0.41x-0.74x del rendimiento linear (1.4x-2.5x más lento)
- **Vulkan**: 0.28x-0.74x del rendimiento linear (1.4x-3.6x más lento)
- **CUDA**: 0.91x-0.96x del rendimiento linear (apenas marginalmente más lento)

**Causa probable:** El overhead de HRM2 (clustering jerárquico, búsqueda en clusters parciales, re-ranking) no se compensa con la reducción del espacio de búsqueda para N≤10K. Linear search con numpy es altamente optimizado para estos tamaños.

### ✅ Hallazgo 2: "Transformed" es extremadamente rápido

El backend "transformed" logra **0.04ms de latencia** y **25,043 QPS** (600x speedup vs linear). Sin embargo:

- Los datos de training muestran valores absurdos (epoch_time=0.0, throughput en billones)
- **Estos datos parecen ser artefactos**, no mediciones reales válidas
- Solo se ejecutó en el sistema Intel con Vulkan

### 📉 Hallazgo 3: Errores frecuentes en benchmarks

De 13 benchmarks, **6 tuvieron errores parciales o totales**:

| Benchmark | Error |
|-----------|-------|
| 20260224_1745 | `'M2MEngine' object has no attribute 'splats'` |
| 20260225_0732 | `cannot access local variable 't0'` (todos los backends) |
| 20260304_2235 | `cannot access local variable 't0'` + `ndarray has no attribute 'detach'` |
| 20260304_2314/2321 | OOM en sistema Intel 2GB (`Unable to allocate 244 MiB`) |
| 20260304_2353 | OOM en 100K splats en 2GB RAM |

### 📊 Hallazgo 4: Outliers en latencia

El benchmark CPU en el sistema Intel muestra una **varianza extrema**:
- Min: 65ms, Max: **7,656ms** (117x diferencia)
- P99: 1,601ms vs Avg: 393ms
- Esto sugiere garbage collection o thrashing de memoria

### 📊 Hallazgo 5: Ingest throughput

| Backend | Ingest QPS (5K) | Ingest QPS (10K) |
|---------|-----------------|------------------|
| CPU | 838-940 | 709-920 |
| Vulkan | 832-1,314 | 942-1,050 |
| CUDA | - | 898-1,057 |

---

## 4. Conclusiones

1. **M2M no ofrece speedup sobre linear search para N≤10K**. El overhead del índice jerárquico supera el beneficio.
2. **El crossover donde M2M debería superar a linear** probablemente está entre 50K-100K splats (no probado exitosamente).
3. **CUDA en RTX 3090 casi iguala a linear** (0.91-0.96x), siendo el backend más competitivo.
4. **Los benchmarks necesitan estabilizarse**: errores repetidos (`t0` variable, `splats` attribute) indican bugs en el benchmark runner.
5. **Los datos del backend "transformed" no son confiables** y deben ser reinvestigados o eliminados.

---

## 5. Recomendaciones

1. Ejecutar benchmarks con N=50K, 100K, 500K, 1M para encontrar el punto de crossover
2. Corregir bugs en el benchmark runner (variable `t0`, atributo `splats`)
3. No publicar datos del backend "transformed" hasta validar
4. Comparar con faiss, annoy, hnswlib como referencia externa
5. Investigar por qué Vulkan es más lento que CPU en el sistema RTX 3090
