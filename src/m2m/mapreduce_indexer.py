"""
MapReduce Indexer — Patrón Map-Reduce para indexación paralela en HRM2.

Acelera el clustering de HRM2 dividiendo el dataset en chunks (Map),
procesando cada chunk en paralelo, y combinando los clusters parciales (Reduce).

Basado en MASFactory's MapReduceAgent pattern.

Uso:
    >>> from .hrm2_engine import HRM2Engine
    >>> engine = HRM2Engine(n_coarse=100)
    >>> engine.add_splats(splats)
    >>> # En lugar de engine.index(), usar:
    >>> build_time = parallel_index(engine, n_workers=4)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChunkResult:
    """Resultado del procesamiento de un chunk (fase Map)."""
    chunk_id: int
    coarse_labels: np.ndarray
    fine_labels: np.ndarray
    coarse_centers: np.ndarray
    fine_centers: Dict[int, np.ndarray]
    indices: np.ndarray
    build_time: float = 0.0


@dataclass
class ReduceResult:
    """Resultado de la fase Reduce."""
    total_build_time: float
    map_time: float
    reduce_time: float
    n_coarse_clusters: int
    n_fine_clusters: int
    n_splats: int
    n_workers_used: int


def _process_chunk(
    chunk_id: int,
    embeddings: np.ndarray,
    indices: np.ndarray,
    n_coarse: int,
    n_fine: int,
    batch_size: int,
    random_state: int,
) -> ChunkResult:
    """
    Procesa un chunk del dataset (fase Map).

    Args:
        chunk_id: ID del chunk.
        embeddings: Embeddings del chunk [N, D].
        indices: Índices originales en el dataset completo.
        n_coarse: Número de coarse clusters.
        n_fine: Número de fine clusters por coarse.
        batch_size: Batch size para K-Means.
        random_state: Semilla aleatoria.

    Returns:
        ChunkResult con labels y centros del chunk.
    """
    from .clustering import KMeans

    start = time.time()
    n_samples = len(embeddings)

    # Coarse clustering del chunk
    n_coarse_eff = min(n_coarse, max(1, n_samples // 10))
    coarse_model = KMeans(
        n_clusters=n_coarse_eff,
        batch_size=batch_size,
        random_state=random_state + chunk_id,
    )
    coarse_labels = coarse_model.fit_predict(embeddings)
    coarse_centers = coarse_model.cluster_centers_ if hasattr(coarse_model, 'cluster_centers_') else np.array([])

    # Fine clustering dentro de cada coarse cluster del chunk
    fine_labels = np.zeros(n_samples, dtype=np.int32)
    fine_centers: Dict[int, np.ndarray] = {}

    for cid in range(n_coarse_eff):
        mask = coarse_labels == cid
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) < 2:
            continue

        cluster_emb = embeddings[mask]
        n_fine_eff = min(n_fine, max(1, len(cluster_indices) // 5))
        fine_model = KMeans(
            n_clusters=n_fine_eff,
            batch_size=min(batch_size, len(cluster_indices)),
            random_state=random_state + chunk_id * 1000 + cid,
        )
        fine_labels[mask] = fine_model.fit_predict(cluster_emb)
        fine_centers[cid] = fine_model.cluster_centers_ if hasattr(fine_model, 'cluster_centers_') else np.array([])

    return ChunkResult(
        chunk_id=chunk_id,
        coarse_labels=coarse_labels,
        fine_labels=fine_labels,
        coarse_centers=coarse_centers,
        fine_centers=fine_centers,
        indices=indices,
        build_time=time.time() - start,
    )


def _reduce_chunks(
    chunk_results: List[ChunkResult],
    n_coarse_global: int,
    embeddings_full: np.ndarray,
    n_fine: int,
    batch_size: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Combina clusters parciales en clusters globales (fase Reduce).

    Args:
        chunk_results: Resultados de la fase Map.
        n_coarse_global: Número objetivo de coarse clusters global.
        embeddings_full: Embeddings completas.
        n_fine: Fine clusters por coarse.
        batch_size: Batch size.
        random_state: Semilla.

    Returns:
        (coarse_assignments, fine_models_dict, fine_assignments_dict, coarse_cluster_indices)
    """
    from .clustering import KMeans

    # Juntar todos los coarse centers de los chunks
    all_centers = []
    for cr in chunk_results:
        if len(cr.coarse_centers) > 0:
            all_centers.append(cr.coarse_centers)

    if not all_centers:
        # Fallback: todo en un cluster
        n = len(embeddings_full)
        return (
            np.zeros(n, dtype=np.int32),
            {},
            {0: np.zeros(n, dtype=np.int32)},
            {0: np.arange(n)},
        )

    all_centers = np.vstack(all_centers)

    # Re-cluster los centers en n_coarse_global clusters
    n_coarse_eff = min(n_coarse_global, max(1, len(all_centers) // 2))
    global_coarse = KMeans(
        n_clusters=n_coarse_eff,
        batch_size=batch_size,
        random_state=random_state,
    )
    # Asignar cada chunk center a un cluster global
    global_labels = global_coarse.fit_predict(all_centers)

    # Asignar embeddings a clusters globales usando mapeo de chunks
    n_total = len(embeddings_full)
    coarse_assignments = np.zeros(n_total, dtype=np.int32)

    # Mapeo: chunk_coarse_label -> global_coarse_label
    offset = 0
    for cr in chunk_results:
        n_centers = len(cr.coarse_centers)
        if n_centers == 0:
            coarse_assignments[cr.indices] = 0
            offset += n_centers
            continue

        # Mapear labels locales del chunk a globales
        local_to_global = global_labels[offset:offset + n_centers]
        coarse_assignments[cr.indices] = local_to_global[cr.coarse_labels]
        offset += n_centers

    # Fine clustering por coarse cluster global
    fine_models: Dict[int, Any] = {}
    fine_assignments: Dict[int, np.ndarray] = {}
    coarse_cluster_indices: Dict[int, np.ndarray] = {}

    for cid in range(n_coarse_eff):
        mask = coarse_assignments == cid
        indices = np.where(mask)[0]
        coarse_cluster_indices[cid] = indices

        if len(indices) < 2:
            fine_models[cid] = None
            fine_assignments[cid] = np.zeros(len(indices), dtype=np.int32)
            continue

        cluster_emb = embeddings_full[mask]
        n_fine_eff = min(n_fine, max(1, len(indices) // 5))
        fine_model = KMeans(
            n_clusters=n_fine_eff,
            batch_size=min(batch_size, len(indices)),
            random_state=random_state + cid,
        )
        fine_assignments[cid] = fine_model.fit_predict(cluster_emb)
        fine_models[cid] = fine_model

    return coarse_assignments, fine_models, fine_assignments, coarse_cluster_indices


def parallel_index(
    engine: Any,
    n_workers: int = 4,
    chunk_size: Optional[int] = None,
) -> float:
    """
    Indexa el HRM2 Engine usando paralelización Map-Reduce.

    Divide el dataset en chunks, procesa cada chunk en paralelo (Map),
    y combina los clusters parciales en clusters globales (Reduce).

    Args:
        engine: Instancia de HRM2Engine ya poblada con splats.
        n_workers: Número de workers para la fase Map.
        chunk_size: Tamaño de cada chunk (None = auto).

    Returns:
        Tiempo total de indexación en segundos.

    Raises:
        RuntimeError: Si el engine no tiene splats.
    """
    from .clustering import KMeans
    from .encoding import FullEmbeddingBuilder

    if not engine.splats:
        raise RuntimeError("Engine no tiene splats. Llama add_splats() primero.")

    start_total = time.time()

    # --- Build embeddings ---
    if engine.embeddings is not None:
        embeddings = engine.embeddings
    else:
        positions = np.array([s.position for s in engine.splats])
        colors = np.array([s.color for s in engine.splats])
        opacities = np.array([s.opacity for s in engine.splats])
        scales = np.array([s.scale for s in engine.splats])
        rotations = np.array([s.rotation for s in engine.splats])
        embeddings = engine.encoder.build(positions, colors, opacities, scales, rotations)
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    n_splats = len(embeddings)

    # Auto chunk size
    if chunk_size is None:
        chunk_size = max(1, n_splats // n_workers)

    # --- MAP phase: procesar chunks en paralelo ---
    map_start = time.time()
    chunk_results: List[ChunkResult] = []

    chunks: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for i in range(0, n_splats, chunk_size):
        chunk_idx = i // chunk_size
        chunk_emb = embeddings[i:i + chunk_size]
        chunk_indices = np.arange(i, min(i + chunk_size, n_splats))
        chunks.append((chunk_idx, chunk_emb, chunk_indices))

    # Para un solo chunk, no usar thread pool (overhead innecesario)
    if len(chunks) <= 1:
        for chunk_idx, chunk_emb, chunk_indices in chunks:
            cr = _process_chunk(
                chunk_idx, chunk_emb, chunk_indices,
                engine.n_coarse, engine.n_fine,
                engine.batch_size, engine.random_state,
            )
            chunk_results.append(cr)
    else:
        n_workers_actual = min(n_workers, len(chunks))
        with ThreadPoolExecutor(max_workers=n_workers_actual) as executor:
            futures = {
                executor.submit(
                    _process_chunk,
                    chunk_idx, chunk_emb, chunk_indices,
                    engine.n_coarse, engine.n_fine,
                    engine.batch_size, engine.random_state,
                ): chunk_idx
                for chunk_idx, chunk_emb, chunk_indices in chunks
            }
            for future in as_completed(futures):
                chunk_results.append(future.result())

    # Ordenar por chunk_id
    chunk_results.sort(key=lambda cr: cr.chunk_id)
    map_time = time.time() - map_start

    # --- REDUCE phase: combinar clusters parciales ---
    reduce_start = time.time()
    n_coarse_global = min(engine.n_coarse, max(1, n_splats // 10))

    coarse_assignments, fine_models, fine_assignments, coarse_cluster_indices = _reduce_chunks(
        chunk_results, n_coarse_global, embeddings,
        engine.n_fine, engine.batch_size, engine.random_state,
    )

    # --- Actualizar engine state ---
    engine.embeddings = embeddings
    engine.coarse_model = KMeans(
        n_clusters=n_coarse_global,
        batch_size=engine.batch_size,
        random_state=engine.random_state,
    )
    engine.coarse_model.fit(embeddings)
    engine.coarse_assignments = coarse_assignments
    engine.fine_models = fine_models
    engine.fine_assignments = fine_assignments
    engine.coarse_cluster_indices = coarse_cluster_indices

    # Build coarse_cluster_embeddings
    engine.coarse_cluster_embeddings = {}
    for cid, indices in coarse_cluster_indices.items():
        if len(indices) > 0:
            engine.coarse_cluster_embeddings[cid] = np.ascontiguousarray(
                embeddings[indices].astype(np.float32)
            )
        else:
            engine.coarse_cluster_embeddings[cid] = np.zeros(
                (0, engine.embedding_dim), dtype=np.float32
            )

    reduce_time = time.time() - reduce_start
    engine._is_indexed = True

    # Update stats
    engine._stats.n_splats = n_splats
    engine._stats.n_coarse_clusters = n_coarse_global
    engine._stats.n_fine_clusters = sum(
        m.n_clusters if m else 0 for m in fine_models.values()
    )
    engine._stats.build_time = time.time() - start_total

    return engine._stats.build_time
