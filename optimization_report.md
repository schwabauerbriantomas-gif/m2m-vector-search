# ⚡ Reporte de Optimización - M2M Vector Search

**Generado:** 2026-03-17  
**Optimizador:** MASFactory Performance Team  
**Versión:** 2.1.0

---

## Problema Principal: ¿Por qué M2M es más lento que linear para N=10K?

### Análisis del Cuello de Botella

El benchmark más reciente (RTX 3090, N=10K, k=10):

| Backend | Latencia (ms) | Overhead vs Linear |
|---------|---------------|-------------------|
| Linear (numpy) | 24.21 | 1.0x (baseline) |
| M2M CPU | 32.93 | **+36% overhead** |
| M2M Vulkan | 32.78 | **+35% overhead** |
| M2M CUDA | 26.54 | **+10% overhead** |

**Raíz del problema:** Para N=10K, el overhead de HRM2 (transform de distancias a centroids, indexing, concatenación de resultados parciales) supera el ahorro de no calcular todas las distancias.

### Desglose del overhead en CPU path (`hrm2_engine.py`):

1. **`coarse_model.transform()`**: ~2-3ms - Calcula distancias a todos los centroids coarse
2. **`np.argsort()` de distancias coarse**: ~0.1ms
3. **Loop sobre n_probe clusters**: Para cada cluster probed:
   - Extraer embeddings con masking: ~1-2ms
   - Calcular distancias euclidianas: ~5-10ms
4. **Re-ranking y ordenamiento**: ~1ms
5. **Mapeo de resultados a splats**: ~2-5ms (dict lookups en `find_neighbors`)

**Total overhead de indexación:** ~10-20ms por query
**Linear search puro:** ~10ms (numpy vectorized para 10K×640)

---

## Optimización 1: Usar `einsum` en vez de loop por cluster (CPU)

**Archivo:** `src/m2m/hrm2_engine.py` - método `query()`

**Problema:** El loop `for coarse_id in closest_coarse` crea arrays temporales por cada cluster.

**Patch:**

```python
# ANTES (loop con concatenación):
for coarse_id in closest_coarse:
    mask = self.coarse_assignments == coarse_id
    cluster_indices = np.where(mask)[0]
    cluster_embeddings = self.embeddings[mask]
    expert_embeddings.append(cluster_embeddings)
    expert_indices.append(cluster_indices)

if expert_embeddings:
    expert_embeddings = np.vstack(expert_embeddings)
    expert_indices = np.concatenate(expert_indices)
    distances = np.linalg.norm(expert_embeddings - query_vector, axis=1)

# DESPUÉS (advanced indexing directo):
# Pre-compute all indices for probed clusters
probed_mask = np.isin(self.coarse_assignments, closest_coarse)
probed_indices = np.where(probed_mask)[0]
probed_embeddings = self.embeddings[probed_indices]

# Vectorized squared distance with einsum (evita sqrt)
diff = probed_embeddings - query_vector
distances_sq = np.einsum('ij,ij->i', diff, diff)

# Fast top-k with argpartition (O(n) en vez de O(n log n))
if len(distances_sq) > k:
    topk_idx = np.argpartition(distances_sq, k - 1)[:k]
    sort_order = np.argsort(distances_sq[topk_idx])
    topk_idx = topk_idx[sort_order]
else:
    topk_idx = np.argsort(distances_sq)
```

**Impacto estimado:** -5 a -10ms por query (elimina overhead de vstack/concatenate).

**Nota:** Este patrón YA está implementado en `query_batch()` pero NO en `query()` individual. Corregir esta inconsistencia.

---

## Optimización 2: Eliminar sqrt redundante

**Problema:** Se calcula `np.linalg.norm()` (incluye sqrt) para comparar distancias. Para top-k, el orden es idéntico sin sqrt.

**Patch (ya parcialmente implementado en query_batch):**
```python
# Reemplazar:
distances = np.linalg.norm(expert_embeddings - query_vector, axis=1)
# Con:
diff = expert_embeddings - query_vector
distances_sq = np.einsum('ij,ij->i', diff, diff)  # sin sqrt
# Solo aplicar sqrt a los k resultados finales para el usuario
```

**Impacto estimado:** -1 a -2ms por query.

---

## Optimización 3: Pre-compute cluster masks como boolean arrays

**Problema:** `self.coarse_assignments == coarse_id` se recalcula cada query.

**Patch:**
```python
# En index():
self._cluster_masks = {}
for cid in range(n_coarse_effective):
    self._cluster_masks[cid] = (self.coarse_assignments == cid)

# En query():
mask = self._cluster_masks.get(coarse_id)
if mask is not None:
    cluster_indices = np.where(mask)[0]  # O(n) pero con cache del mask
```

**Impacto estimado:** -0.5ms por query (menor para pocas queries, mayor para batches).

---

## Optimización 4: Batch size óptimo para GPU

**Análisis actual:**
- `GPUVectorIndex`: chunk_size = min(32768, N, VRAM_limit). Para N=10K, todo cabe en un chunk.
- El overhead de GPU viene de `vkQueueWaitIdle()` por cada dispatch.

**Problema:** Cada query individual causa un GPU dispatch + wait. Para batches pequeños, el overhead de dispatch (~0.5ms) domina.

**Recomendación:**
```python
# Para queries individuales, usar CPU si N < 50K
# Para batches > 10, usar GPU
THRESHOLD_GPU_BATCH = 10
THRESHOLD_GPU_N = 50000
```

---

## Optimización 5: Cache de query results (YA IMPLEMENTADO)

`QueryOptimizer` ya tiene LRU cache. Verificar que está habilitado por defecto en `M2MOptimized`.

---

## Optimización 6: Reducir overhead en `find_neighbors()`

**Archivo:** `src/m2m/splats.py`

**Problema:** El loop anidado `for i in range(batch_size): for j, (splat, dist)` hace lookups individuales a `self.mu[idx]`.

**Patch:**
```python
# En vez de loop anidado, usar indexing directo:
idx_array = np.array([splat.id for splat, dist in results])
mu_out[i] = self.mu[idx_array]
alpha_out[i] = self.alpha[idx_array]
kappa_out[i] = self.kappa[idx_array]
```

---

## Resumen de Impacto Estimado

| Optimización | Reducción estimada (ms) | Esfuerzo |
|-------------|------------------------|----------|
| 1. einsum + advanced indexing | -5 a -10ms | Bajo (copiar de query_batch) |
| 2. Eliminar sqrt | -1 a -2ms | Bajo |
| 3. Pre-compute masks | -0.5ms | Bajo |
| 4. GPU threshold tuning | Variable | Medio |
| 5. Cache (ya existe) | Variable | Ninguno |
| 6. Vectorized find_neighbors | -2 a -3ms | Medio |

**Total estimado:** Reducción de 8-15ms por query, lo que podría llevar a M2M CPU a ~17-24ms (competitivo o mejor que linear para N=10K).

---

## Punto de Crossover Estimado

Basado en el análisis:
- **N < 5K:** Linear siempre gana
- **N = 10K:** M2M CPU iguala a linear con optimizaciones
- **N > 50K:** M2M debería superar a linear significativamente
- **N > 100K:** HRM2 con GPU debería dar speedups de 5-50x

**Recomendación:** Ejecutar benchmarks en N=50K y N=100K para validar.
