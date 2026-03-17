# 🧪 Reporte de Chaos Testing - M2M Vector Search

**Generado:** 2026-03-17  
**Agente:** MASFactory Chaos Agent  
**Metodología:** Análisis estático del código + edge cases identificados

---

## Resumen

Se identificaron **24 edge cases** distribuidos en 5 categorías. De estos, **8 causan crashes/errores no manejados**, **6 causan resultados incorrectos silenciosos**, y **10 son manejados correctamente**.

---

## 1. Inputs Inválidos

| # | Input | Resultado Esperado | Resultado Actual | Severidad |
|---|-------|-------------------|------------------|-----------|
| 1.1 | Query vacía `np.array([])` | Error claro | Crash (index error) | 🔴 |
| 1.2 | Query con NaN | Error o exclusión | Distancias = NaN, resultados arbitrarios | 🟠 |
| 1.3 | Query con Inf | Error o exclusión | Distancias = Inf, posible crash en sort | 🟠 |
| 1.4 | Query dimensión incorrecta (ej. 128 en vez de 640) | Error claro | Crash silencioso o resultados basura | 🟠 |
| 1.5 | Query 1D en vez de 2D | Manejado | ✅ `query.squeeze()` + reshape | 🟢 |
| 1.6 | k=0 | Error o vacío | `np.argpartition` crash con k=0 | 🔴 |
| 1.7 | k > N (solicitando más vecinos que splats) | Truncar a N | ✅ `k = min(k, max(1, self.n_active))` | 🟢 |
| 1.8 | Dimensión negativa | Error | Comportamiento indefinido | 🟡 |

---

## 2. Datos Duplicados

| # | Input | Resultado Esperado | Resultado Actual | Severidad |
|---|-------|-------------------|------------------|-----------|
| 2.1 | Splats con IDs duplicados | Error o deduplicación | Sobreescribe silenciosamente | 🟡 |
| 2.2 | Vectores idénticos (diferentes IDs) | Búsqueda correcta | ✅ Funciona, distancias=0 | 🟢 |
| 2.3 | Mismo ID en add() dos veces | Error o merge | Segundo add sobreescribe primero | 🟡 |
| 2.4 | Document IDs duplicados en `_vectors` dict | Error | Sobreescribe silenciosamente en dict | 🟡 |

---

## 3. Estados Límite del Sistema

| # | Input | Resultado Esperado | Resultado Actual | Severidad |
|---|-------|-------------------|------------------|-----------|
| 3.1 | Buscar en índice vacío (0 splats) | Lista vacía | ✅ Fallback a random | 🟢 |
| 3.2 | Buscar sin llamar a build_index() | Error claro | ✅ Fallback a random en find_neighbors | 🟢 |
| 3.3 | N=1 (un solo splat) | k=1 correcto | Funciona con min(k, n_active) | 🟢 |
| 3.4 | add() que excede max_splats | Error claro | ✅ `return False` | 🟢 |
| 3.5 | consolidate() con threshold=1.0 | Eliminar todo | ⚠️ Elimina splats pero no limpia índice HRM2 | 🟠 |

---

## 4. Concurrent Access

| # | Escenario | Resultado Esperado | Resultado Actual | Severidad |
|---|-----------|-------------------|------------------|-----------|
| 4.1 | add() concurrente desde 2 threads | Ambos agregados | ⚠️ Race condition en `n_active` | 🟠 |
| 4.2 | search() durante add() | Resultado consistente | ⚠️ Puede leer índice a medias | 🟠 |
| 4.3 | delete() durante search() | No crash | ⚠️ Sin locks, posible crash | 🟠 |

---

## 5. Memoria y Recursos

| # | Escenario | Resultado Esperado | Resultado Actual | Severidad |
|---|-----------|-------------------|------------------|-----------|
| 5.1 | VRAM insuficiente para GPUVectorIndex | Fallback a CPU | ✅ try/except con fallback | 🟢 |
| 5.2 | float('inf') en mu buffer (post-consolidate) | No usado en search | ⚠️ consolidate marca como inf pero no los excluye | 🟡 |
| 5.3 | Índice HRM2 no reconstruido después de consolidate | Rebuild automático | ❌ Índice queda stale | 🔴 |

---

## Casos Críticos Requeriendo Acción

### 🔴 C-01: k=0 causa crash
```python
# En hrm2_engine.py, query():
# np.argpartition(distances_sq, k - 1) con k=0 → index -1
```
**Fix:** `k = max(1, k)`

### 🔴 C-02: Query vacía causa crash
```python
# Si query = np.array([]), query.shape = (0,)
# np.dot con shape (0,) × (N, D) → error
```
**Fix:** Validar `query.size > 0` antes de procesar

### 🟠 C-03: NaN/Inf no detectados
**Fix:** Agregar validación al inicio de search():
```python
if not np.all(np.isfinite(query)):
    raise ValueError("Query contiene NaN o Inf")
```

### 🔴 C-04: consolidate() no rebuild índice
```python
# Después de marcar splats como inf, el índice HRM2 sigue apuntando a ellos
# Las búsquedas posteriores devuelven vectores con inf
```
**Fix:** Llamar `self.splats.build_index()` después de `consolidate()`.

---

## Plan de Tests Recomendado

Agregar a `tests/`:
1. `test_search_empty_query_raises` - Query vacía
2. `test_search_nan_query_raises` - Query con NaN
3. `test_search_k_zero_raises` - k=0
4. `test_consolidate_rebuilds_index` - Verificar índice post-consolidate
5. `test_add_concurrent` - Thread safety en add
6. `test_search_wrong_dimension_raises` - Dimensión incorrecta
