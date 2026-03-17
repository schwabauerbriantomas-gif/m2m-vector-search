# 🔬 Investigación: Estado del Arte en Vector Search y Gaussian Splats para Retrieval

**Generado:** 2026-03-17  
**Investigador:** MASFactory Research Team  
**Método:** Web research con fuentes citadas

---

## 1. Estado del Arte en Vector Search (2025-2026)

### 1.1 Algoritmos Dominantes

#### HNSW (Hierarchical Navigable Small World)
- **Paper original:** Malkov & Yashunin (2018) [1]
- **Estado actual:** Implementación de referencia en `hnswlib` (C++/Python) y FAISS
- **Complejidad:** O(log N) búsqueda, O(N log N) construcción
- **Recall típico:** >95% con ef_construction=200, M=16
- **Ventaja principal:** Graph-based, sin necesidad de estructuras adicionales
- **Implementaciones notables:**
  - `hnswlib` (https://github.com/nmslib/hnswlib) - ~5K stars
  - FAISS IVF-HNSW hybrid
  - Milvus, Weaviate, Qdrant (lo usan internamente)

#### IVF (Inverted File Index)
- **Paper original:** Jégou et al. (2011) [2]
- **Estado actual:** Implementado en FAISS como `IndexIVFFlat`, `IndexIVFPQ`
- **Complejidad:** O(N/√nlist) búsqueda
- **Ventaja:** Simple, paralelizable, buen throughput con batches
- **Desventaja:** Recall baja para nprobe pequeño

#### FAISS (Facebook AI Similarity Search)
- **Estado actual:** v1.8+, la librería más usada para ANN
- **índices principales:** Flat, IVFFlat, IVFPQ, HNSW, IVF-HNSW
- **Soporte GPU:** CUDA optimizado
- **Performance:** ~1M vectors 768D → <1ms query en GPU [3]

#### SPTAG (Microsoft)
- **Paper:** VBASE (OSDI 2023) [4], SPFresh (SOSP 2023) [5]
- **Estado actual:** Soporta actualización incremental in-place a escala de billones
- **Métodos:** KD-tree + RNG o Balanced K-means + RNG

#### DiskANN (Microsoft Research)
- **Paper:** Subramanya et al. (2019) [6]
- **Concepto:** Búsqueda ANN desde disco, soporta 100B+ vectores
- **Ventaja:** No requiere que todo el índice esté en RAM

### 1.2 Tendencias 2025-2026

1. **Quantization avanzada:** Product Quantization (PQ), Scalar Quantization (SQ), Optimized Product Quantization (OPQ)
2. **Híbridos graph-partition:** IVF-HNSW combina coarse partitioning con grafos para búsquedas eficientes
3. **Multi-modal search:** Búsqueda cruzada texto-imagen con embeddings compartidos (CLIP, etc.)
4. **Learned indices:** Redes neuronales que aprenden la estructura del espacio para mejor particionamiento
5. **Streaming updates:** Soporte para inserción/deleción en tiempo real sin rebuild completo

---

## 2. Gaussian Splatting para Retrieval

### 2.1 Estado Actual

Gaussian Splatting (3DGS) se introdujo en 2023 por Kerbl et al. [7] para novel view synthesis. Su aplicación a retrieval es un área de investigación **naciente y sin precedentes significativos**.

### 2.2 Similitudes con Vector Search

3DGS representa datos como colecciones de Gaussianas 3D con parámetros:
- **μ (media):** posición en el espacio
- **Σ (covarianza):** forma/rotación
- **α (opacidad):** importancia
- **s (scale):** tamaño

Esto es análogo a cómo M2M representa vectores como:
- **μ:** centro del splat (embedding)
- **Σ/κ:** ancho (kappa = precisión)
- **α:** peso/activación

### 2.3 Trabajo Relacionado

- **Superpoint Graphs** (Landrieu & Simonovsky, 2018) [8]: Graph-based point cloud segmentation
- **PointNet++** (Qi et al., 2017) [9]: Deep learning on point sets
- **Radiance Fields for retrieval:** Investigación limitada; principalmente usada para generation, no search
- **Energy-Based Models para retrieval:** LeCun et al. (2006) [10] propusieron EBMs para ranking, precursor conceptual de M2M

### 2.4 Evaluación Crítica

**M2M es innovador** en aplicar el paradigma de Gaussian Splats a vector search, pero:
- No existe literatura directa que compare gaussian splats con HNSW/IVF para ANN
- El enfoque teórico tiene mérito (representación probabilística permite uncertainty quantification)
- **Falta evidencia empírica** de que supere a métodos establecidos en recall/latencia

---

## 3. HRM2 vs HNSW vs IVF

### 3.1 Comparación Teórica

| Aspecto | HRM2 (M2M) | HNSW | IVF |
|---------|-----------|------|-----|
| **Estructura** | Clustering jerárquico probabilístico | Grafo multinivel | Partitioning + brute force |
| **Construcción** | K-means multinivel + training | Inserción incremental | K-means offline |
| **Búsqueda** | Coarse → fine con probabilidades | Greedy en grafo | Probe + scan |
| **Complejidad búsqueda** | O(N/nprobe + n_coarse) | O(log N × ef) | O(N/nlist × nprobe) |
| **Actualización** | Rebuild parcial | Incremental | Incremental (FAISS) |
| **Uncertainty** | ✅ Nativo (EBM) | ❌ No | ❌ No |
| **Multi-GPU** | ✅ (Vulkan/CUDA) | Parcial (FAISS) | ✅ (FAISS) |
| **Memoria** | O(N × D) + centroids | O(N × M × D) | O(N × D) + centroids |
| **Implementación** | Python/numpy | C++ | C++ (FAISS) |

### 3.2 Performance Observada (M2M benchmarks)

Basado en los benchmarks analizados:

| N | Linear | M2M CPU | M2M GPU | HNSW (estimado) |
|---|--------|---------|---------|-----------------|
| 5K | 5.2ms, 190 QPS | 14.0ms, 72 QPS | 19.9ms, 50 QPS | ~0.1ms, 10000 QPS |
| 10K | 9.8ms, 102 QPS | 23.0ms, 44 QPS | 33.0ms, 30 QPS | ~0.2ms, 5000 QPS |

**Nota:** Los benchmarks de HNSW son estimaciones basadas en [3]. No se ejecutaron benchmarks comparativos directos.

### 3.3 Donde M2M podría destacar

1. **Uncertainty quantification:** Ningún ANN index ofrece esto nativamente
2. **Energy-based exploration:** Descubrir regiones no cubiertas del espacio
3. **Multi-backend:** Vulkan + CUDA + CPU con misma API
4. **Probabilistic retrieval:** Scores con significado probabilístico

### 3.4 Donde M2M necesita mejorar

1. **Latencia pura:** 10-100x más lento que HNSW para N≤10K
2. **Index construction:** No incremental (requiere rebuild)
3. **Recall medido:** No se reporta recall@k en los benchmarks
4. **Escalabilidad:** No probado exitosamente sobre 10K splats

---

## Fuentes

[1] Malkov, Y.A., & Yashunin, D.A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI*. https://arxiv.org/abs/1603.09320

[2] Jégou, H., Douze, M., & Schmid, C. (2011). "Product quantization for nearest neighbor search." *IEEE TPAMI*.

[3] Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." *IEEE TBD*. https://github.com/facebookresearch/faiss

[4] Zhang, Q., et al. (2023). "VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity." *OSDI 2023*.

[5] Zhang, Q., et al. (2023). "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search." *SOSP 2023*.

[6] Subramanya, S., et al. (2019). "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." *NeurIPS 2019*.

[7] Kerbl, B., et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM TOG*. https://arxiv.org/abs/2308.04079

[8] Landrieu, L., & Simonovsky, M. (2018). "Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs." *CVPR 2018*.

[9] Qi, C.R., et al. (2017). "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." *NeurIPS 2017*.

[10] LeCun, Y., et al. (2006). "A Tutorial on Energy-Based Learning." In *Predicting Structured Data*.

---

## Conclusiones

1. **HNSW es el estándar de facto** para ANN search en 2025-2026
2. **M2M ocupa un nicho único** con uncertainty quantification via EBMs, pero necesita demostrar competitividad en latencia/recall
3. **Gaussian Splats para retrieval es una idea novedosa** sin comparables directos en la literatura
4. **El punto de crossover** donde M2M supera a linear search aún no se ha demostrado empíricamente
5. **Recomendación de investigación:** Ejecutar benchmarks head-to-head con hnswlib y FAISS en N=10K-1M para validar la propuesta
