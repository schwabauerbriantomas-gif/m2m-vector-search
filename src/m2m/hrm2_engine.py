"""
HRM2 Engine - Hierarchical Retrieval Model 2

Implements a two-level hierarchical index for fast similarity search
in large-scale Gaussian splat datasets.
"""

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from .clustering import KMeans
from .encoding import FullEmbeddingBuilder
from .engine import M2MEngine
from .splat_types import GaussianSplat


@dataclass
class SearchResult:
    """Result of a similarity search."""

    splat_id: int
    distance: float
    coarse_cluster: int
    fine_cluster: int


@dataclass
class HRM2Config:
    """Configuration for HRM2 Engine.

    Cableada en HRM2Engine(config=...) [audit r2-19; antes era un dataclass
    muerto]: sus campos son los defaults de los knobs HRM2 y un kwarg
    explícito de HRM2Engine manda sobre el config (None = tomar el default).
    Nota: batch_size=1000 (A/B: 3.8x más rápido que 10000 con igual recall)
    difiere del default histórico de HRM2Engine (10000), que se conserva si
    no se pasa config.
    """

    n_coarse: int = 100  # Number of coarse clusters
    n_fine: int = 1000  # Number of fine clusters per coarse
    embedding_dim: int = 640  # Embedding dimension
    n_probe: int = 5  # Clusters to probe during search
    batch_size: int = 1000  # Batch size for K-Means (A/B: 1000 es 3.8x mas
    # rapido que 10000 con igual recall; bench/kmeans_bs_ab.json)
    random_state: int = 42
    adaptive_probing: bool = False  # SPEC 3: adaptive probe count
    adaptive_threshold: float = 1.5  # score ratio top1/top2 to use fewer probes
    adaptive_min_probe: int = 1  # minimum probes for "easy" queries
    metric: str = "euclidean"  # "euclidean" or "cosine"
    auto_k: bool = False  # Auto-detect optimal k via silhouette analysis (legacy)
    # Elección de K: 'fixed' usa n_coarse tal cual; 'silhouette' es el sweep
    # legacy (auto_k=True); 'gate' usa el gate de modulus (gate_modulus.py).
    k_selection: str = "fixed"
    gate_target_docs_per_cluster: int = 25  # K = N // target si el gate dice jerárquico


@dataclass
class HRM2Stats:
    """Statistics for HRM2 Engine."""

    n_splats: int = 0
    n_coarse_clusters: int = 0
    n_fine_clusters: int = 0
    build_time: float = 0.0
    avg_query_time: float = 0.0
    total_queries: int = 0


class HRM2Engine:
    """
    Hierarchical Retrieval Model 2 (HRM2) Engine.

    Implements a two-level hierarchical index:
    - Level 1 (Coarse): K-Means clusters for fast pruning
    - Level 2 (Fine): Additional clustering within each coarse cluster

    This provides significant speedup over brute-force search while
    maintaining high recall.

    Example:
        >>> engine = HRM2Engine(n_coarse=100, n_fine=1000)
        >>> engine.add_splats(splats)
        >>> engine.index()
        >>> results = engine.query(query_vector, k=10)
    """

    def __init__(
        self,
        n_coarse: Optional[int] = None,
        n_fine: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        n_probe: Optional[int] = None,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        config=None,
        metric: Optional[str] = None,
        auto_k: Optional[bool] = None,
        k_selection: Optional[str] = None,
        gate_target_docs_per_cluster: Optional[int] = None,
    ):
        """
        Initialize HRM2 Engine.

        Args:
            n_coarse: Number of coarse clusters
            n_fine: Fine clusters per coarse cluster
            embedding_dim: Dimension of embedding vectors
            n_probe: Coarse clusters to search
            batch_size: Batch size for K-Means
            random_state: Random seed
            config: M2MConfig for hardware acceleration; o HRM2Config: en
                ese caso sus campos son los defaults de los knobs HRM2
                (n_probe, adaptive_probing, metric, ...) y los kwargs
                explícitos mandan sobre él [audit r2-19]. Con None en un knob
                se toma el default (el de HRM2Config si se pasó, si no el
                histórico: 100, 1000, 640, 5, 10000, 42, "euclidean", ...)
            metric: Distance metric - "euclidean" or "cosine"
            auto_k: Auto-detect optimal k via silhouette analysis (legacy;
                equivalente a k_selection='silhouette')
            k_selection: 'fixed' (n_coarse tal cual, default) | 'silhouette'
                (sweep legacy) | 'gate' (gate de modulus, gate_modulus.py:
                mono-escala -> knee; jerárquico -> K operacional)
            gate_target_docs_per_cluster: target de docs/cluster cuando el
                gate dictamina corpus jerárquico (K = N // target)
        """
        # [audit r2-19] HRM2Config era un dataclass muerto (nunca consumido:
        # adaptive probing inalcanzable por API). Resolución de defaults:
        # kwarg explícito > campo de HRM2Config (si `config` lo es) > default
        # histórico de __init__. Con M2MConfig (o sin config) no cambia nada.
        defaults = dict(
            n_coarse=100,
            n_fine=1000,
            embedding_dim=640,
            n_probe=5,
            batch_size=10000,
            random_state=42,
            metric="euclidean",
            auto_k=False,
            k_selection="fixed",
            gate_target_docs_per_cluster=25,
        )
        adaptive = dict(adaptive_probing=False, adaptive_threshold=1.5, adaptive_min_probe=1)
        if isinstance(config, HRM2Config):
            defaults.update({k: getattr(config, k) for k in defaults})
            adaptive.update({k: getattr(config, k) for k in adaptive})
        resolved: Dict[str, Any] = {
            k: (v if v is not None else defaults[k])
            for k, v in (
                ("n_coarse", n_coarse),
                ("n_fine", n_fine),
                ("embedding_dim", embedding_dim),
                ("n_probe", n_probe),
                ("batch_size", batch_size),
                ("random_state", random_state),
                ("metric", metric),
                ("auto_k", auto_k),
                ("k_selection", k_selection),
                ("gate_target_docs_per_cluster", gate_target_docs_per_cluster),
            )
        }
        n_coarse = resolved["n_coarse"]
        n_fine = resolved["n_fine"]
        embedding_dim = resolved["embedding_dim"]
        n_probe = resolved["n_probe"]
        batch_size = resolved["batch_size"]
        random_state = resolved["random_state"]
        metric = resolved["metric"]
        auto_k = resolved["auto_k"]
        k_selection = resolved["k_selection"]
        gate_target_docs_per_cluster = resolved["gate_target_docs_per_cluster"]
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.embedding_dim = embedding_dim
        self.n_probe = n_probe
        self.batch_size = batch_size
        self.random_state = random_state
        self.config = config
        self.adaptive_probing = adaptive["adaptive_probing"]  # SPEC 3
        self.adaptive_threshold = adaptive["adaptive_threshold"]
        self.adaptive_min_probe = adaptive["adaptive_min_probe"]
        self.metric = metric
        self.auto_k = auto_k
        if auto_k and k_selection == "fixed":
            # Backward-compat: auto_k=True mantiene el comportamiento silhouette
            k_selection = "silhouette"
        if k_selection not in ("fixed", "silhouette", "gate"):
            raise ValueError(
                f"k_selection debe ser 'fixed'|'silhouette'|'gate', no {k_selection!r}"
            )
        self.k_selection = k_selection
        self.gate_target_docs_per_cluster = int(gate_target_docs_per_cluster)

        # Initialize hardware router if configured
        self.router = M2MEngine(config) if config else None

        # Statistics
        self._is_indexed = False
        self._stats = HRM2Stats()

        # Storage
        self.splats: List[GaussianSplat] = []
        self.embeddings: Optional[np.ndarray] = None

        # Index
        self.coarse_model: Optional[KMeans] = None
        self.coarse_assignments: Optional[np.ndarray] = None
        self.fine_models: Dict[int, KMeans] = {}
        self.fine_assignments: Dict[int, np.ndarray] = {}

        # Fast caching
        self.coarse_cluster_indices: Dict[int, np.ndarray] = {}
        self.coarse_cluster_embeddings: Dict[int, np.ndarray] = {}
        # [P17] radio por cluster: r_c = max_x ||x - centroide_c|| sobre las filas
        # asignadas a c. Cota superior para podar clusters en la búsqueda CON
        # filtro (dist(q,x) >= dist(q,c) - r_c). None = inválido: se recalcula en
        # una pasada en la primera búsqueda con filtro (compute_cluster_radii).
        self.coarse_cluster_radii: Optional[Dict[int, float]] = None

        # Encoder
        self.encoder = FullEmbeddingBuilder()

        # Quality diagnostics
        self._silhouette_score: Optional[float] = None
        self._calinski_harabasz: Optional[float] = None
        self._auto_k_cache: Optional[int] = None
        self._gate_selection = None  # GateKSelection si k_selection='gate'
        # Diagnostics (silhouette + Calinski-Harabasz) are O(sample^2 * d) and cost
        # ~27% of build time; opt-in via engine.diagnostics = True.
        self.diagnostics = False

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine distance (makes euclidean ≡ cosine)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vectors / norms

    def _auto_detect_k(self, embeddings: np.ndarray) -> int:
        """Auto-detect optimal k using silhouette analysis."""
        n = embeddings.shape[0]
        max_k = min(50, max(5, int(math.sqrt(n))))
        # Use a subsample for speed
        sample_size = min(n, 5000)
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n, sample_size, replace=False)
        sample = embeddings[indices]

        best_k = 5
        best_score = -2.0

        for k in range(5, max_k + 1, 5):
            try:
                km = KMeans(
                    n_clusters=k,
                    batch_size=min(self.batch_size, sample_size),
                    random_state=self.random_state,
                    use_mini_batch=True,
                )
                labels = km.fit_predict(sample)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(sample, labels, metric="euclidean")
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        self._auto_k_cache = best_k
        return best_k

    def _adaptive_n_probe(self, coarse_distances: np.ndarray) -> int:
        """SPEC 3: Return adaptive probe count based on distance ratio."""
        if not self.adaptive_probing or len(coarse_distances) < 2:
            return self.n_probe
        # Use argpartition for O(N) instead of sort O(N log N)
        two_smallest = np.partition(coarse_distances, 1)[:2]
        if two_smallest[0] > 0:
            ratio = two_smallest[1] / two_smallest[0]
            if ratio > self.adaptive_threshold:
                return self.adaptive_min_probe
        return self.n_probe

    def add_splats(self, splats: List[GaussianSplat]) -> None:
        """
        Add splats to the engine.

        Args:
            splats: List of GaussianSplat objects
        """
        self.splats.extend(splats)
        self._is_indexed = False

    def index(self, precomputed_embeddings: np.ndarray = None) -> float:
        """
        Build the hierarchical index.

        Args:
            precomputed_embeddings: Optional direct embedding vectors to bypass the encoder

        Returns:
            Time taken to build index
        """
        start_time = time.time()

        if len(self.splats) == 0 and precomputed_embeddings is None:
            return 0.0

        if precomputed_embeddings is not None:
            self.embeddings = np.ascontiguousarray(precomputed_embeddings.astype(np.float32))
            n_samples = self.embeddings.shape[0]
        else:
            # Build embeddings
            positions = np.array([s.position for s in self.splats])
            colors = np.array([s.color for s in self.splats])
            opacities = np.array([s.opacity for s in self.splats])
            scales = np.array([s.scale for s in self.splats])
            rotations = np.array([s.rotation for s in self.splats])

            self.embeddings = self.encoder.build(positions, colors, opacities, scales, rotations)
            # Ensure correct dtype
            self.embeddings = np.ascontiguousarray(self.embeddings.astype(np.float32))
            n_samples = len(self.splats)

        # Normalize for cosine metric (euclidean on normalized vectors == cosine distance)
        if self.metric == "cosine":
            self.embeddings = self._normalize(self.embeddings)

        # Elección de K
        if self.k_selection == "gate":
            # Gate de modulus (index-time, una vez): mono-escala -> knee;
            # jerárquico -> K operacional N//target. Reemplaza el sweep
            # silhouette, que sobre curva lisa maximiza ruido (gate_integration_plan).
            from .gate_modulus import gate_select_k

            sel = gate_select_k(
                self.embeddings,
                target_docs_per_cluster=self.gate_target_docs_per_cluster,
                seed=self.random_state,
            )
            self._gate_selection = sel
            self.n_coarse = sel.k
            # Presupuesto de escaneo: el default n_probe=5 fue calibrado para
            # K~sqrt(N) (fracción 5/25 = 20% del corpus, splats.py). Si el gate
            # elige una K distinta, mantener la fracción de escaneo (knob real
            # de latencia/recall) subiendo el piso de n_probe. Solo en modo gate.
            scan_floor = math.ceil(sel.k * (5 / 25))
            if self.n_probe < scan_floor:
                print(
                    f"[HRM2] Gate: n_probe {self.n_probe}->{scan_floor} "
                    f"(fracción de escaneo >= 20% con K={sel.k})"
                )
                self.n_probe = scan_floor
            print(
                f"[HRM2] Gate k_selection: k={sel.k} jerarquico={sel.jerarquico} "
                f"slope_fino={sel.slope_fino} octava={sel.octava}"
            )
        elif self.k_selection == "silhouette" and self._auto_k_cache is None:
            detected_k = self._auto_detect_k(self.embeddings)
            self.n_coarse = detected_k
            print(f"[HRM2] Auto-detected n_coarse={detected_k} via silhouette analysis")

        # Level 1: Coarse clustering
        n_coarse_effective = min(self.n_coarse, n_samples // 10)
        n_coarse_effective = max(1, n_coarse_effective)

        self.coarse_model = KMeans(
            n_clusters=n_coarse_effective,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        self.coarse_assignments = self.coarse_model.fit_predict(self.embeddings)

        # Level 2: Fine clustering — LAZY (only built when query(lod=1) is called)
        # SplatStore.find_neighbors() only uses coarse clustering, so skip this
        # expensive step during index build. It saves O(n_coarse × n_fine) KMeans time.
        self.fine_models = {}
        self.fine_assignments = {}
        self._fine_built = False
        self.coarse_cluster_indices = {}
        self.coarse_cluster_embeddings = {}
        self.coarse_cluster_radii = None  # [P17] índice nuevo: radios sin calcular

        for coarse_id in range(n_coarse_effective):
            mask = self.coarse_assignments == coarse_id
            cluster_indices = np.where(mask)[0]

            self.coarse_cluster_indices[coarse_id] = cluster_indices
            cluster_embeddings = np.ascontiguousarray(self.embeddings[mask].astype(np.float32))
            self.coarse_cluster_embeddings[coarse_id] = cluster_embeddings

        # Fine clustering deferred to _build_fine_index() — called lazily by query(lod=1)

        # Opt #3: Pre-compute cluster masks as boolean arrays
        self._cluster_masks: Dict[int, np.ndarray] = {}
        for cid in range(n_coarse_effective):
            self._cluster_masks[cid] = self.coarse_assignments == cid

        self._is_indexed = False  # temp

        # Quality diagnostics (opt-in: O(sample^2*d) — ~27% of build time when on)
        if self.diagnostics and n_samples >= 100 and n_coarse_effective >= 2:
            try:
                sample_size = min(n_samples, 5000)
                rng = np.random.RandomState(self.random_state)
                diag_indices = rng.choice(n_samples, sample_size, replace=False)
                diag_sample = self.embeddings[diag_indices]
                diag_labels = self.coarse_assignments[diag_indices]
                unique_labels = len(set(diag_labels.tolist()))
                if unique_labels >= 2:
                    self._silhouette_score = silhouette_score(
                        diag_sample, diag_labels, metric="euclidean"
                    )
                    self._calinski_harabasz = calinski_harabasz_score(diag_sample, diag_labels)
                    metric_label = self.metric.upper()
                    print(
                        f"[HRM2] Diagnostics ({metric_label}, n={n_samples}, "
                        f"k={n_coarse_effective}): "
                        f"silhouette={self._silhouette_score:.4f}, "
                        f"CH={self._calinski_harabasz:.1f}"
                    )
                    if self._silhouette_score < 0.1:
                        warnings.warn(
                            f"HRM2 silhouette={self._silhouette_score:.4f} < 0.1. "
                            f"Consider using HNSW for better recall with dense embeddings.",
                            UserWarning,
                            stacklevel=2,
                        )
            except Exception as e:
                print(f"[HRM2] Diagnostics skipped: {e}")

        self._is_indexed = True

        # Update stats
        self._stats.n_splats = n_samples
        self._stats.n_coarse_clusters = n_coarse_effective
        self._stats.n_fine_clusters = 0  # Lazy: built only on query(lod=1)
        self._stats.build_time = time.time() - start_time

        return self._stats.build_time

    def _build_fine_index(self):
        """Build Level-2 fine clustering lazily (only when lod=1 queries are made)."""
        if self._fine_built or self.embeddings is None:
            return
        for coarse_id in range(len(self.coarse_cluster_indices)):
            cluster_indices = self.coarse_cluster_indices[coarse_id]
            if len(cluster_indices) < 2:
                self.fine_models[coarse_id] = None
                self.fine_assignments[coarse_id] = np.zeros(len(cluster_indices), dtype=np.int32)
                continue

            cluster_embeddings = self.coarse_cluster_embeddings[coarse_id]
            n_fine_effective = min(self.n_fine, len(cluster_indices) // 5)
            n_fine_effective = max(1, n_fine_effective)

            fine_model = KMeans(
                n_clusters=n_fine_effective,
                batch_size=min(self.batch_size, len(cluster_indices)),
                random_state=self.random_state + coarse_id,
            )
            self.fine_models[coarse_id] = fine_model
            self.fine_assignments[coarse_id] = fine_model.fit_predict(cluster_embeddings)

        self._fine_built = True
        self._stats.n_fine_clusters = sum(
            m.n_clusters if m else 0 for m in self.fine_models.values()
        )

    def _coarse_distances(self, query_vector: np.ndarray) -> np.ndarray:
        """Distancias a los centroides gruesos (numpy routing; idéntico a
        coarse_model.transform, ~20x más rápido por query). Fórmula única para
        query() y query_batch() [audit r2-18: paridad fila a fila]."""
        coarse_model = self.coarse_model
        assert coarse_model is not None
        centroids = getattr(coarse_model, "centroids_", None)
        if centroids is not None:
            return np.sqrt(np.maximum(((centroids - query_vector) ** 2).sum(axis=1), 0.0))
        return coarse_model.transform(query_vector.reshape(1, -1))[0]

    def _lod01_candidates(self, query_vector: np.ndarray, closest_coarse, k: int, lod: int) -> list:
        """Candidatos de lod 0/1: los k MÁS CERCANOS (distancia exacta) de los
        clústeres sondeados [audit r2-19; antes eran los primeros k por orden
        de índice — miembros arbitrarios del clúster, recall@10 1-17% en la
        sonda p6_hrm2.py].

        - lod 0 (coarse): los k más cercanos de cada clúster grueso sondeado.
        - lod 1 (fine): los k más cercanos de los clústeres finos más cercanos
          de cada clúster grueso sondeado, hasta reunir k por clúster grueso
          (aproximación: el recall depende del tamaño de los clústeres finos).

        Devuelve [(idx, dist, coarse_id, fine_id)] en orden de clúster sondeado
        (cercano -> lejano); el caller ordena por distancia y recorta a k.
        """
        candidates = []
        for coarse_id in closest_coarse:
            coarse_id = int(coarse_id)
            coarse_indices = np.where(self.coarse_assignments == coarse_id)[0]
            if lod == 0:
                diff = self.embeddings[coarse_indices] - query_vector
                dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                order = np.argsort(dist, kind="stable")[:k]
                candidates.extend(
                    (int(coarse_indices[i]), float(dist[i]), coarse_id, 0) for i in order
                )
            else:
                fine_model = self.fine_models.get(coarse_id)
                if fine_model is None:
                    continue
                fine_assigns = self.fine_assignments.get(
                    coarse_id, np.zeros(len(coarse_indices), dtype=np.int32)
                )
                fine_dists = fine_model.transform(query_vector.reshape(1, -1))[0]
                added = 0
                for fine_id in np.argsort(fine_dists, kind="stable"):
                    members = coarse_indices[fine_assigns == fine_id]
                    if len(members) == 0:
                        continue
                    diff = self.embeddings[members] - query_vector
                    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                    order = np.argsort(dist, kind="stable")[:k]
                    candidates.extend(
                        (int(members[i]), float(dist[i]), coarse_id, int(fine_id)) for i in order
                    )
                    added += len(order)
                    if added >= k:
                        break
            if len(candidates) >= k:
                break
        return candidates

    def query(
        self, query_vector: np.ndarray, k: int = 10, lod: int = 2
    ) -> List[Tuple[GaussianSplat, float]]:
        """
        Query for k most similar splats.

        Args:
            query_vector: Query embedding vector
            k: Number of results
            lod: Level of Detail (0=Coarse Approx, 1=Fine Approx, 2=Exact MoE Router)

        Returns:
            List of (Splat, distance) tuples
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_start = time.time()

        query_vector = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).flatten())

        # Normalize query for cosine metric
        if self.metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm > 1e-10:
                query_vector = query_vector / norm

        # Find nearest coarse clusters (numpy routing; _coarse_distances es la
        # fórmula única de query()/query_batch() [audit r2-18])
        coarse_distances = self._coarse_distances(query_vector)
        n_probe_actual = self._adaptive_n_probe(coarse_distances)
        closest_coarse = np.argsort(coarse_distances)[:n_probe_actual]

        candidates = []

        if lod == 0 or lod == 1:
            # LOD 0 (coarse) / LOD 1 (fine): los k MÁS CERCANOS de los
            # clústeres sondeados [audit r2-19; antes: los primeros k por
            # orden de índice, i.e. miembros arbitrarios del clúster].
            if lod == 1 and not self._fine_built:
                self._build_fine_index()
            candidates = [
                (i, d, c)
                for i, d, c, _ in self._lod01_candidates(query_vector, closest_coarse, k, lod)
            ]

        else:
            # LOD 2 (Exact): Collect candidates from nearest coarse clusters
            expert_indices_list = []
            expert_embeddings_list = []
            coarse_ids_list = []

            for coarse_id in closest_coarse:
                mask = self._cluster_masks.get(int(coarse_id))
                if mask is None:
                    mask = self.coarse_assignments == coarse_id
                cluster_indices = np.where(mask)[0]

                if len(cluster_indices) == 0:
                    continue

                # Reuse precomputed per-cluster embeddings (built in index())
                # instead of copying embeddings[mask] on every query.
                pre = self.coarse_cluster_embeddings.get(int(coarse_id))
                if pre is not None and len(pre) == len(cluster_indices):
                    cluster_embeddings = pre
                else:
                    cluster_embeddings = self.embeddings[mask]

                expert_embeddings_list.append(cluster_embeddings)
                expert_indices_list.append(cluster_indices)
                coarse_ids_list.append(np.full(len(cluster_indices), int(coarse_id)))

            if expert_indices_list:
                expert_embeddings = np.vstack(expert_embeddings_list)
                expert_indices = np.concatenate(expert_indices_list)
                coarse_ids = np.concatenate(coarse_ids_list)

                if self.router:  # Hardware-accelerated MoE distance
                    results = self.router.compute_expert_distances(
                        query_vector,
                        expert_embeddings,
                        expert_indices,
                        coarse_ids,
                        np.zeros(len(expert_indices), dtype=np.int32),
                    )
                    candidates = [(r[0], r[1], r[2]) for r in results]
                else:
                    # Vectorized distance computation via Gram-matrix trick
                    diff = expert_embeddings - query_vector
                    distances_sq = np.einsum("ij,ij->i", diff, diff)

                    # Fast top-k via argpartition + partial sort
                    if len(distances_sq) > k:
                        topk_idx = np.argpartition(distances_sq, k - 1)[:k]
                        subset_dists = distances_sq[topk_idx]
                        sort_idx = np.argsort(subset_dists)
                        topk_idx = topk_idx[sort_idx]
                    else:
                        topk_idx = np.argsort(distances_sq)

                    # Build candidates directly sorted (no Python list sort)
                    candidates = [
                        (
                            int(expert_indices[i]),
                            float(np.sqrt(distances_sq[i])),
                            int(coarse_ids[i]),
                        )
                        for i in topk_idx
                    ]

        # Sort by distance and return top-k
        candidates.sort(key=lambda x: x[1])
        results = candidates[:k]
        if not results:
            return []

        # Update stats
        self._stats.total_queries += 1
        query_time = time.time() - query_start
        self._stats.avg_query_time = (
            self._stats.avg_query_time * (self._stats.total_queries - 1) + query_time
        ) / self._stats.total_queries

        # Return (index, distance) tuples. If splats exist, return splat objects.
        if self.splats and len(self.splats) > max(r[0] for r in results):
            return [(self.splats[idx], dist) for idx, dist, _ in results]
        else:
            return [(idx, dist) for idx, dist, _ in results]

    def query_with_details(
        self, query_vector: np.ndarray, k: int = 10, lod: int = 2
    ) -> List[SearchResult]:
        """
        Query with detailed results including cluster info.

        Args:
            query_vector: Query embedding vector
            k: Number of results
            lod: Level of Detail (0=Coarse Approx, 1=Fine Approx, 2=Exact MoE Router)

        Returns:
            List of SearchResult objects
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_vector = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).flatten())

        # Normalize query for cosine metric
        if self.metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm > 1e-10:
                query_vector = query_vector / norm

        # Find nearest coarse clusters
        coarse_distances = self.coarse_model.transform(query_vector.reshape(1, -1))[0]
        n_probe_actual = self._adaptive_n_probe(coarse_distances)
        closest_coarse = np.argsort(coarse_distances)[:n_probe_actual]

        candidates = []

        _has_splats = self.splats and len(self.splats) > 0

        if lod == 0 or lod == 1:
            # [audit r2-19] mismo contrato que query(): ver _lod01_candidates
            if lod == 1 and not self._fine_built:
                self._build_fine_index()
            candidates = [
                (self.splats[i].id if _has_splats else i, d, c, f)
                for i, d, c, f in self._lod01_candidates(query_vector, closest_coarse, k, lod)
            ]

        else:
            expert_indices_list = []
            expert_embeddings_list = []
            coarse_ids_list = []

            for coarse_id in closest_coarse:
                mask = self._cluster_masks.get(int(coarse_id))
                if mask is None:
                    mask = self.coarse_assignments == coarse_id
                cluster_indices = np.where(mask)[0]

                if len(cluster_indices) == 0:
                    continue

                cluster_embeddings = self.embeddings[mask]

                expert_embeddings_list.append(cluster_embeddings)
                expert_indices_list.append(cluster_indices)
                coarse_ids_list.append(np.full(len(cluster_indices), int(coarse_id)))

            if expert_indices_list:
                expert_embeddings = np.vstack(expert_embeddings_list)
                expert_indices = np.concatenate(expert_indices_list)
                coarse_ids = np.concatenate(coarse_ids_list)

                if self.router:
                    results = self.router.compute_expert_distances(
                        query_vector,
                        expert_embeddings,
                        expert_indices,
                        coarse_ids,
                        np.zeros(len(expert_indices), dtype=np.int32),
                    )
                    candidates = [
                        (self.splats[int(r[0])].id if _has_splats else int(r[0]), r[1], r[2], r[3])
                        for r in results
                    ]
                else:
                    diff = expert_embeddings - query_vector
                    distances_sq = np.einsum("ij,ij->i", diff, diff)

                    # Fast top-k via argpartition
                    if len(distances_sq) > k:
                        topk_idx = np.argpartition(distances_sq, k - 1)[:k]
                        sort_idx = np.argsort(distances_sq[topk_idx])
                        topk_idx = topk_idx[sort_idx]
                    else:
                        topk_idx = np.argsort(distances_sq)

                    candidates = [
                        (
                            (
                                self.splats[int(expert_indices[i])].id
                                if _has_splats
                                else int(expert_indices[i])
                            ),
                            float(np.sqrt(distances_sq[i])),
                            int(coarse_ids[i]),
                            0,
                        )
                        for i in topk_idx
                    ]

        # Sort and return top-k
        candidates.sort(key=lambda x: x[1])

        return [
            SearchResult(splat_id=sid, distance=d, coarse_cluster=c, fine_cluster=f)
            for sid, d, c, f in candidates[:k]
        ]

    def query_batch(
        self, query_vectors: np.ndarray, k: int = 10, lod: int = 2
    ) -> List[List[Tuple[GaussianSplat, float]]]:
        """
        Batched query for higher CPU throughput.
        """
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_vectors = np.ascontiguousarray(np.asarray(query_vectors, dtype=np.float32))
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        B = query_vectors.shape[0]
        results_batch = []

        for b in range(B):
            # [audit r2-18] Mismo ruteo que query(), fila a fila: la query se
            # normaliza ANTES de elegir los clústeres gruesos (antes el
            # transform() recibía la query SIN normalizar bajo 'cosine' y
            # discrepaba con query() en ~16-22% de las queries no unitarias),
            # los clústeres se ordenan con argsort (mismo orden de candidatos
            # que query()) y se aplica _adaptive_n_probe por fila (antes se
            # ignoraba el probing adaptativo).
            query_vector = query_vectors[b]
            if self.metric == "cosine":
                norm = np.linalg.norm(query_vector)
                if norm > 1e-10:
                    query_vector = query_vector / norm
            coarse_distances = self._coarse_distances(query_vector)
            n_probe_actual = self._adaptive_n_probe(coarse_distances)
            closest_coarse = np.argsort(coarse_distances)[:n_probe_actual]

            if lod == 0 or lod == 1:
                # Modos no exactos: delegar a query() con la fila ORIGINAL
                # (query() normaliza ella misma).
                results_batch.append(self.query(query_vectors[b], k=k, lod=lod))
                continue

            # LOD 2 (Exact CPU routing)
            # Create list of indices
            expert_indices_list = [
                self.coarse_cluster_indices[c]
                for c in closest_coarse
                if len(self.coarse_cluster_indices[c]) > 0
            ]
            if not expert_indices_list:
                results_batch.append([])
                continue

            expert_indices = np.concatenate(expert_indices_list)
            # Advanced indexing (Zero copying until extraction)
            expert_embeddings = self.embeddings[expert_indices]

            # Vectorized squared distance (no sqrt, skips overhead)
            diff = expert_embeddings - query_vector
            distances_sq = np.einsum("ij,ij->i", diff, diff)

            # Fast top-K
            if len(distances_sq) > k:
                topk_idx = np.argpartition(distances_sq, k - 1)[:k]
                # Sort exactly within the k subset
                subset_dists = distances_sq[topk_idx]
                sort_idx = np.argsort(subset_dists)
                topk_idx = topk_idx[sort_idx]
            else:
                topk_idx = np.argsort(distances_sq)

            # Return real distances for interface
            # When the index was built with precomputed_embeddings, self.splats
            # is empty (no Splat objects) — return (int, dist) like query() does.
            if self.splats:
                local_res = [
                    (self.splats[expert_indices[i]], float(np.sqrt(distances_sq[i])))
                    for i in topk_idx
                ]
            else:
                local_res = [
                    (int(expert_indices[i]), float(np.sqrt(distances_sq[i]))) for i in topk_idx
                ]
            results_batch.append(local_res)

            self._stats.total_queries += 1

        return results_batch

    def get_stats(self) -> HRM2Stats:
        """Get engine statistics."""
        return self._stats

    # --- [P17] radios por cluster: cota de poda para la búsqueda con filtro ---
    def invalidate_cluster_radii(self) -> None:
        """Invalida el cache de radios: μ o el índice cambiaron por fuera de los
        caminos incrementales (drift Hebbiano, compact, rebuild). La próxima
        búsqueda con filtro lo recalcula en una pasada."""
        self.coarse_cluster_radii = None

    def compute_cluster_radii(self, rows: np.ndarray) -> Dict[int, float]:
        """Una pasada: ``r_c = max_x ||x - centroide_c||`` sobre las filas de
        ``coarse_cluster_indices[c]`` leídas de ``rows`` (el MISMO array con el que
        se puntúan los candidatos). Cota superior por construcción. Filas no
        finitas (splat muerto) dejan ``r_c = inf``: ese cluster nunca se poda."""
        cent = np.asarray(self.coarse_model.centroids_, dtype=np.float32)
        radii: Dict[int, float] = {}
        for c, idx in self.coarse_cluster_indices.items():
            if len(idx) == 0:
                radii[c] = 0.0
                continue
            diff = np.asarray(rows[idx], dtype=np.float32) - cent[c]
            d = np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))
            d = d[np.isfinite(d)]
            if d.size < len(idx):
                radii[c] = float("inf")
            else:
                # margen float32 mínimo: sigue siendo cota, nunca se achica
                radii[c] = float(d.max()) * (1.0 + 1e-6) + 1e-6
        self.coarse_cluster_radii = radii
        return radii

    def bump_cluster_radius(self, cid: int, row: np.ndarray) -> None:
        """Mantiene la cota tras AGREGAR o REESCRIBIR filas del cluster ``cid``:
        ``r_c = max(r_c, max_row ||row - centroide_c||)``. Quitar filas sólo puede
        achicar la distancia máxima: no hace falta tocar nada. No-op con el cache
        invalidado (se recalcula entero cuando haga falta)."""
        radii = getattr(self, "coarse_cluster_radii", None)
        if radii is None or self.coarse_model is None:
            return
        cent = np.asarray(self.coarse_model.centroids_, dtype=np.float32)
        cid = int(cid)
        if cid >= len(cent):
            return
        diff = np.asarray(row, dtype=np.float32) - cent[cid]
        if diff.ndim == 1:
            d2 = float(np.dot(diff, diff))
        else:
            d2 = float(np.einsum("ij,ij->i", diff, diff).max())
        d = float(np.sqrt(d2)) if np.isfinite(d2) else float("inf")
        d = d * (1.0 + 1e-6) + 1e-6
        radii[cid] = max(float(radii.get(cid, 0.0)), d)

    def clear(self) -> None:
        """Clear all data."""
        self.splats = []
        self.embeddings = None
        self.coarse_model = None
        self.coarse_assignments = None
        self.fine_models = {}
        self.fine_assignments = {}
        self.coarse_cluster_indices = {}
        self.coarse_cluster_embeddings = {}
        self._cluster_masks = {}
        self._is_indexed = False
        self.coarse_cluster_radii = None  # [P17]
        self._stats = HRM2Stats()
        self._silhouette_score = None
        self._calinski_harabasz = None
        self._gate_selection = None


def generate_test_splats(n_splats: int, seed: int = 42) -> List[GaussianSplat]:
    """
    Generate synthetic splats for testing.

    Args:
        n_splats: Number of splats
        seed: Random seed

    Returns:
        List of GaussianSplat objects
    """
    np.random.seed(seed)

    splats = []
    for i in range(n_splats):
        splat = GaussianSplat(
            id=i,
            position=np.random.randn(3).astype(np.float32) * 10,
            color=np.random.rand(3).astype(np.float32),
            opacity=np.random.rand(),
            scale=np.exp(np.random.randn(3).astype(np.float32) * -2),
            rotation=np.random.randn(4).astype(np.float32),
        )
        # Normalize quaternion
        splat.rotation /= np.linalg.norm(splat.rotation)
        splats.append(splat)

    return splats
