#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M (Machine-to-Memory) - EBM Vector Database
High-performance Gaussian Splat storage and retrieval with Energy-Based Model features.

Modos de operación:
  - EMBEDDED: SimpleVectorDB / AdvancedVectorDB (librería Python)
  - SERVER:   M2MClient (cliente REST)
  - CLUSTER:  M2MCluster (cluster distribuido con router energético opcional)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# M2M Core Modules
try:
    from .config import M2MConfig
    from .ebm import EBMEnergy, EBMExploration, SOCEngine
    from .ebm.energy_api import EnergyResult as EnergyResult  # re-export
    from .ebm.exploration import EnergyRegion, ExplorationSuggestion
    from .ebm.soc import AvalancheResult, CriticalityReport, RelaxationResult
    from .energy import EnergyFunction
    from .engine import M2MEngine
    from .entity_extractor import EntityCandidate as EntityCandidate
    from .entity_extractor import M2MEntityExtractor as M2MEntityExtractor
    from .entity_extractor import M2MGraphEntityExtractor as M2MGraphEntityExtractor
    from .geometry import (
        exp_map,
        geodesic_distance,
        log_map,
        normalize_sphere,
        project_to_tangent,
    )
    from .graph_splat import GaussianGraphStore as GaussianGraphStore
    from .graph_splat import GraphEdge as GraphEdge
    from .graph_splat import GraphSplat as GraphSplat
    from .graph_splat import NodeType as NodeType
    from .splats import SplatStore
    from .storage import M2MPersistence, WriteAheadLog as WriteAheadLog
except ImportError:
    from .config import M2MConfig

    M2MEngine = None


# ---------------------------------------------------------------------------
# Geometry utilities (fallback)
# ---------------------------------------------------------------------------


def normalize_sphere(x, dim=-1):
    norm = np.linalg.norm(x, axis=dim, keepdims=True)
    return x / (norm + 1e-8)


def geodesic_distance(x, y):
    return np.arccos(np.clip(np.dot(x, y.T), -1, 1))


def exp_map(base, tangent):
    return base + tangent


def log_map(base, point):
    return point - base


def project_to_tangent(base, vector):
    return vector - np.dot(vector, base.T) * base


# ---------------------------------------------------------------------------
# M2MMemory (internal)
# ---------------------------------------------------------------------------


class M2MMemory:
    """
    M2M Memory System.

    Manages:
    - Gaussian Splats (μ, α, κ)
    - 3-tier Memory Hierarchy (VRAM Hot, RAM Warm, SSD Cold)
    - SOC Controller (Self-Organized Criticality)
    """

    def __init__(self, config: M2MConfig):
        self.config = config

        # 3-tier Memory Hierarchy
        self.vram_hot = config.enable_3_tier_memory
        self.ram_warm = config.enable_3_tier_memory
        self.ssd_cold = config.enable_3_tier_memory

        # Splat Store
        self.splats = SplatStore(config)

        # SOC Controller
        self.soc_threshold = config.soc_threshold
        self.phi = 0.0  # Order parameter

        # Energy Function
        self.energy_fn = EnergyFunction(config)

        print(
            f"[INFO] M2M initialized on {config.device} (compute_device={config.compute_device})"
        )
        print(f"[INFO] Latent dim: {config.latent_dim}")
        print(f"[INFO] Max splats: {config.max_splats}")
        print(f"[INFO] 3-tier memory: {config.enable_3_tier_memory}")

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input x to spherical latent space S^639."""
        return normalize_sphere(x)

    def compute_energy(self, x: np.ndarray) -> np.ndarray:
        """Compute energy E(x) = E_splats + E_geom + E_comp - H[q(x)]"""
        x_encoded = self.encode(x)

        E_splats = self.energy_fn.E_splats(x_encoded, self.splats)
        E_geom = self.energy_fn.E_geom(x_encoded)
        E_comp = self.energy_fn.E_comp(x_encoded)
        H_q = self.splats.entropy(x_encoded)

        E = (
            self.config.energy_splat_weight * E_splats
            + self.config.energy_geom_weight * E_geom
            + self.config.energy_comp_weight * E_comp
            - self.config.temperature * H_q
        )
        return E

    def retrieve(
        self, query: np.ndarray, k: int = None, lod: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve k-nearest neighbors from splat store."""
        if k is None:
            k = self.config.knn_k
        query_norm = normalize_sphere(query)
        neighbors_mu, neighbors_alpha, neighbors_kappa = self.splats.find_neighbors(
            query_norm, k, lod=lod
        )
        return neighbors_mu, neighbors_alpha, neighbors_kappa

    def sample(self, x: np.ndarray, n_steps: int = None) -> np.ndarray:
        """Generate samples using Langevin dynamics."""
        if n_steps is None:
            n_steps = self.config.langevin_steps
        noise = np.random.randn(*x.shape).astype(np.float32) * self.config.langevin_dt
        return x + noise

    def consolidate(self, threshold: float = None) -> int:
        """Run SOC consolidation based on order parameter."""
        if threshold is None:
            threshold = self.soc_threshold

        if self.splats.n_active > 0:
            splats_to_remove = np.where(
                self.splats.alpha[: self.splats.n_active] < threshold
            )[0]
        else:
            splats_to_remove = []

        n_removed = 0
        for splat_idx in splats_to_remove:
            self.splats.mu[splat_idx] = float("inf")
            n_removed += 1

        if n_removed > 0:
            self.splats.compact()
            print(f"[INFO] SOC: Consolidated {n_removed} splats")

        return n_removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the M2M system."""
        return {
            "n_active_splats": self.splats.n_active,
            "max_splats": self.splats.max_splats,
            "mean_frequency": self.splats.frequency[: self.splats.n_active]
            .mean()
            .item(),
            "mean_kappa": self.splats.kappa[: self.splats.n_active].mean().item(),
            "mean_alpha": self.splats.alpha[: self.splats.n_active].mean().item(),
            "entropy": self.splats.entropy().item(),
            "soc_phi": self.phi.item() if hasattr(self.phi, "item") else float(self.phi),
        }

    def forward(self, x: np.ndarray, mode: str = "energy") -> np.ndarray:
        """Forward pass of M2M system."""
        if mode == "energy":
            return self.compute_energy(x)
        elif mode == "retrieve":
            neighbors_mu, neighbors_alpha, neighbors_kappa = self.retrieve(x)
            return neighbors_mu
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# M2MEngine (internal)
# ---------------------------------------------------------------------------


class M2MEngine:
    """M2M High-Performance Engine."""

    def __init__(self, config: M2MConfig):
        super().__init__()
        self.config = config
        self.m2m = M2MMemory(config)

        if config.enable_vulkan:
            try:
                from engine import M2MEngine as VulkanEngine

                self.vulkan_engine = VulkanEngine(config)
            except ImportError:
                print("[WARNING] Vulkan engine module not found, falling back to None.")
                self.vulkan_engine = None
        else:
            self.vulkan_engine = None
            print("[INFO] Vulkan acceleration disabled")

    def add_splats(self, vectors: np.ndarray, labels: List[str] = None) -> int:
        """Add new splats to the system."""
        vectors_norm = normalize_sphere(vectors)
        n_added = 0
        for i, vector_norm in enumerate(vectors_norm):
            if self.m2m.splats.add_splat(vector_norm):
                n_added += 1
            else:
                print(f"[WARNING] Failed to add splat {i}")

        if n_added > 0:
            self.m2m.splats.build_index()

        print(f"[INFO] Added {n_added} splats and built HRM2 index")
        return n_added

    def search(self, query: np.ndarray, k: int = None) -> np.ndarray:
        """Search for nearest neighbors."""
        return self.m2m.retrieve(query, k)

    def generate(self, query: np.ndarray, n_samples: int = 10) -> np.ndarray:
        """Generate samples starting from query."""
        return self.m2m.sample(query)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.m2m.get_statistics()

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to spherical latent space."""
        return self.m2m.encode(x)

    def compute_energy(self, x: np.ndarray) -> np.ndarray:
        """Compute energy of input."""
        return self.m2m.compute_energy(x)

    def get_splats_arrays(self):
        """Retorna mu, alpha, kappa como arrays numpy para EBM."""
        n = self.m2m.splats.n_active
        if n == 0:
            return None, None, None
        mu = self.m2m.splats.mu[:n].copy()
        alpha = self.m2m.splats.alpha[:n].copy()
        kappa = self.m2m.splats.kappa[:n].copy()
        return mu, alpha, kappa

    def export_to_dataloader(
        self,
        batch_size=32,
        num_workers=0,
        importance_sampling=False,
        generate_samples=False,
    ):
        """Export M2M Data Lake as a numpy-based iterable of batches."""
        from .data_lake import M2MDataLake

        dataset = M2MDataLake(
            m2m_engine=self,
            batch_size=batch_size,
            importance_sampling=importance_sampling,
            generate_samples=generate_samples,
        )
        return dataset

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass (energy computation)."""
        return self.m2m(x)

    def load_optimized(self, path: str):
        """Loads optimized dataset with pre-computed Gaussian Splats."""
        from loaders.optimized_loader import load_m2m_dataset

        splats = load_m2m_dataset(path)
        self._build_hrm2_from_splats(splats)
        return self

    def _build_hrm2_from_splats(self, splats):
        """Builds HRM2 index directly from pre-computed splats."""
        self.m2m.splats._build_hrm2_from_splats(splats)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class DocResult:
    """Resultado de una operación de base de datos."""

    def __init__(
        self,
        id: str,
        score: float = 0.0,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        document: Optional[str] = None,
        energy: Optional[float] = None,
        confidence: Optional[float] = None,
    ):
        self.id = id
        self.score = score
        self.vector = vector
        self.metadata = metadata or {}
        self.document = document
        self.energy = energy
        self.confidence = confidence

    def __repr__(self):
        return (
            f"DocResult(id={self.id!r}, score={self.score:.4f}, "
            f"energy={self.energy}, confidence={self.confidence})"
        )


class UpdateResult:
    """Resultado de una operación update."""

    def __init__(self, success: bool, energy_delta: float = 0.0, message: str = ""):
        self.success = success
        self.energy_delta = energy_delta
        self.message = message

    def __repr__(self):
        return f"UpdateResult(success={self.success}, energy_delta={self.energy_delta:.4f})"


class DeleteResult:
    """Resultado de una operación delete."""

    def __init__(self, deleted: int, energy_freed: float = 0.0):
        self.deleted = deleted
        self.energy_freed = energy_freed

    def __repr__(self):
        return f"DeleteResult(deleted={self.deleted}, energy_freed={self.energy_freed:.4f})"


class SearchResult:
    """Resultado de una búsqueda con información energética."""

    def __init__(
        self,
        results: List[DocResult],
        query_energy: float = 0.0,
        total_confidence: float = 0.0,
        uncertainty_regions: Optional[List] = None,
        search_time_ms: float = 0.0,
    ):
        self.results = results
        self.query_energy = query_energy
        self.total_confidence = total_confidence
        self.uncertainty_regions = uncertainty_regions or []
        self.search_time_ms = search_time_ms

    def __repr__(self):
        return (
            f"SearchResult({len(self.results)} results, "
            f"query_energy={self.query_energy:.4f}, "
            f"total_confidence={self.total_confidence:.4f})"
        )


# ---------------------------------------------------------------------------
# SimpleVectorDB - CRUD Completo + Persistencia + EBM opcional
# ---------------------------------------------------------------------------


class SimpleVectorDB:
    """
    SimpleVectorDB: 'The SQLite of Vector DBs'

    Versión 2.0: CRUD completo, metadata, WAL, y EBM opcional.

    Modos:
    - edge: Sin EBM, sin WAL (mínimo overhead)
    - standard: WAL + metadata SQLite
    - ebm: WAL + metadata + EBM features (energía, exploración)

    Uso:
        db = SimpleVectorDB(latent_dim=768, storage_path='./data', mode='standard')
        db.add(['doc1', 'doc2'], vectors, metadata=[{'cat': 'tech'}, {'cat': 'sci'}])
        results = db.search(query, k=10)
        db.update('doc1', metadata={'cat': 'technology'})
        db.delete('doc1')
    """

    def __init__(
        self,
        device: Optional[str] = None,
        latent_dim: int = 640,
        enable_lsh_fallback: bool = True,
        lsh_threshold: float = 0.15,
        storage_path: Optional[str] = None,
        enable_wal: bool = True,
        enable_ebm: bool = False,
        mode: str = "standard",  # 'edge', 'standard', 'ebm'
    ):
        config = M2MConfig.simple(device=device)
        config.latent_dim = latent_dim
        self.engine = M2MEngine(config)
        self.latent_dim = latent_dim

        self.enable_lsh_fallback = enable_lsh_fallback
        self.lsh_threshold = lsh_threshold
        self.lsh = None
        self._use_lsh = False

        # Almacenamiento interno en memoria (para compatibilidad)
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict] = {}
        self._documents: Dict[str, str] = {}
        self._deleted: set = set()

        # Persistencia
        self.storage = None
        if storage_path and mode != "edge":
            self.storage = M2MPersistence(storage_path, enable_wal=(enable_wal and mode != "edge"))

        # EBM
        self.ebm_enabled = enable_ebm or mode == "ebm"
        self._ebm_energy: Optional[EBMEnergy] = None
        self._ebm_exploration: Optional[EBMExploration] = None
        if self.ebm_enabled:
            self._ebm_energy = EBMEnergy()
            self._ebm_exploration = EBMExploration(self._ebm_energy)

    def _compute_silhouette(self, vectors: np.ndarray, sample_size: int = 1000) -> float:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            if len(vectors) < 3:
                return 1.0

            sample_idx = np.random.choice(
                len(vectors), min(sample_size, len(vectors)), replace=False
            )
            sample = vectors[sample_idx]
            n_clusters = max(2, int(np.sqrt(len(sample))))
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
            labels = kmeans.fit_predict(sample)
            return silhouette_score(sample, labels)
        except ImportError:
            print("[WARNING] scikit-learn is required for data distribution analysis.")
            return 1.0

    def _update_ebm_splats(self):
        """Actualiza los splats del EBM desde el engine."""
        if not self.ebm_enabled or self._ebm_energy is None:
            return
        mu, alpha, kappa = self.engine.get_splats_arrays()
        if mu is not None and len(mu) > 0:
            self._ebm_energy.update_splats(mu, alpha, kappa)
            # Actualizar exploración con los vectores actuales
            if self._ebm_exploration is not None and self._vectors:
                vecs = np.array(list(self._vectors.values()), dtype=np.float32)
                ids = list(self._vectors.keys())
                self._ebm_exploration.update_vectors(vecs, ids)

    def add(
        self,
        ids: Optional[Any] = None,  # Can be List[str] or np.ndarray (legacy compat)
        vectors: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
    ) -> int:
        """
        Añade vectores a la base de datos.

        Args:
            ids: Lista de IDs únicos (opcional; si no se pasa, se auto-generan).
                 Para compatibilidad legacy también acepta np.ndarray como primer arg.
            vectors: Array numpy [N, D]
            metadata: Lista de dicts de metadata por documento
            documents: Lista de textos originales

        Returns:
            Número de vectores añadidos.
        """
        # Backward compat: add(vectors_array) without keyword
        if ids is not None and isinstance(ids, np.ndarray):
            vectors = ids
            ids = None

        if vectors is None:
            raise ValueError("vectors no puede ser None")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        n = len(vectors)

        # Auto-generar IDs si no se proporcionan
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in range(n)]

        # Validar longitudes
        if len(ids) != n:
            raise ValueError(f"len(ids)={len(ids)} != len(vectors)={n}")
        if metadata and len(metadata) != n:
            raise ValueError(f"len(metadata)={len(metadata)} != len(vectors)={n}")
        if documents and len(documents) != n:
            raise ValueError(f"len(documents)={len(documents)} != len(vectors)={n}")

        # Comprobar LSH fallback
        if self.enable_lsh_fallback and n >= 3:
            silhouette = self._compute_silhouette(vectors)
            if silhouette < self.lsh_threshold:
                print(f"[M2M] Distribución uniforme (silhouette={silhouette:.4f})")
                print("[M2M] Activando LSH fallback...")
                try:
                    from .lsh_index import CrossPolytopeLSH, LSHConfig

                    config = LSHConfig(
                        dim=self.engine.config.latent_dim,
                        n_tables=15, n_bits=18, n_probes=50, n_candidates=500,
                    )
                    self.lsh = CrossPolytopeLSH(config)
                    self.lsh.index(vectors)
                    self._use_lsh = True
                    # Guardar metadata en memoria
                    for i, doc_id in enumerate(ids):
                        self._vectors[doc_id] = vectors[i]
                        self._metadata[doc_id] = (metadata[i] if metadata else {})
                        self._documents[doc_id] = (documents[i] if documents else None)
                    return n
                except ImportError as e:
                    print(f"[WARNING] Could not load LSH module: {e}.")

        self._use_lsh = False
        n_added = self.engine.add_splats(vectors)

        # Guardar en memoria
        for i, doc_id in enumerate(ids):
            self._vectors[doc_id] = vectors[i]
            self._metadata[doc_id] = (metadata[i] if metadata else {})
            self._documents[doc_id] = (documents[i] if documents else None)

        # Persistencia
        if self.storage:
            self.storage.save_vectors(vectors, ids)
            for i, doc_id in enumerate(ids):
                self.storage.save_metadata(
                    doc_id,
                    shard_idx=0,
                    vector_idx=i,
                    metadata=(metadata[i] if metadata else {}),
                    document=(documents[i] if documents else None),
                )

        # Actualizar EBM
        self._update_ebm_splats()

        return n_added

    def update(
        self,
        id: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        document: Optional[str] = None,
        upsert: bool = False,
    ) -> UpdateResult:
        """
        Actualiza un documento existente.

        Args:
            id: ID del documento a actualizar
            vector: Nuevo vector (opcional)
            metadata: Nueva metadata (opcional)
            document: Nuevo texto (opcional)
            upsert: Si True, crea el documento si no existe

        Returns:
            UpdateResult con éxito y delta de energía.
        """
        if id not in self._vectors and id not in self._deleted:
            if upsert:
                vec = vector if vector is not None else np.zeros(self.latent_dim, dtype=np.float32)
                self.add(ids=[id], vectors=vec[np.newaxis, :], metadata=[metadata or {}], documents=[document])
                return UpdateResult(success=True, message="upserted")
            return UpdateResult(success=False, message=f"Document {id!r} not found")

        if id in self._deleted:
            if upsert:
                self._deleted.discard(id)
                vec = vector if vector is not None else np.zeros(self.latent_dim, dtype=np.float32)
                self.add(ids=[id], vectors=vec[np.newaxis, :], metadata=[metadata or {}], documents=[document])
                return UpdateResult(success=True, message="restored and upserted")
            return UpdateResult(success=False, message=f"Document {id!r} is deleted")

        energy_delta = 0.0

        if vector is not None:
            old_vec = self._vectors.get(id)
            self._vectors[id] = np.asarray(vector, dtype=np.float32)

            # Calcular delta de energía si EBM habilitado
            if self.ebm_enabled and self._ebm_energy is not None and old_vec is not None:
                old_e = self._ebm_energy.energy(old_vec)
                new_e = self._ebm_energy.energy(vector)
                energy_delta = new_e - old_e

        if metadata is not None:
            self._metadata[id] = metadata
            if self.storage:
                self.storage.update_metadata(id, metadata=metadata)

        if document is not None:
            self._documents[id] = document
            if self.storage:
                self.storage.update_metadata(id, document=document)

        return UpdateResult(success=True, energy_delta=energy_delta)

    def delete(
        self,
        id: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
        hard: bool = False,
    ) -> DeleteResult:
        """
        Elimina documentos del índice.

        Args:
            id: ID único a eliminar
            ids: Lista de IDs a eliminar
            filter: Filtro de metadata para eliminar por condición
            hard: Si True, eliminación permanente. Si False, soft-delete.

        Returns:
            DeleteResult con conteo y energía liberada.
        """
        # Resolver targets
        to_delete = set()
        if id:
            to_delete.add(id)
        if ids:
            to_delete.update(ids)
        if filter and self.storage:
            filtered_ids = self.storage.filter_by_metadata(filter)
            to_delete.update(filtered_ids)
        elif filter:
            # Filtrar en memoria
            for doc_id, meta in self._metadata.items():
                if doc_id not in self._deleted and self._match_filter(meta, filter):
                    to_delete.add(doc_id)

        if not to_delete:
            return DeleteResult(deleted=0, energy_freed=0.0)

        energy_freed = 0.0
        deleted_count = 0

        for doc_id in to_delete:
            if doc_id in self._vectors:
                # Calcular energía liberada si EBM habilitado
                if self.ebm_enabled and self._ebm_energy is not None:
                    vec = self._vectors.get(doc_id)
                    if vec is not None:
                        energy_freed += float(self._ebm_energy.energy(vec))

                if hard:
                    self._vectors.pop(doc_id, None)
                    self._metadata.pop(doc_id, None)
                    self._documents.pop(doc_id, None)
                    self._deleted.discard(doc_id)
                    if self.storage:
                        self.storage.hard_delete(doc_id)
                else:
                    self._deleted.add(doc_id)
                    if self.storage:
                        self.storage.soft_delete(doc_id)

                deleted_count += 1

        return DeleteResult(deleted=deleted_count, energy_freed=energy_freed)

    def _match_filter(self, meta: dict, filter_dict: dict) -> bool:
        """Evalúa filtro de metadata."""
        if self.storage:
            return self.storage._matches_filter(meta, filter_dict)
        for key, cond in filter_dict.items():
            if key not in meta:
                return False
            val = meta[key]
            if isinstance(cond, dict):
                for op, comp in cond.items():
                    if op == "$eq" and val != comp:
                        return False
                    elif op == "$ne" and val == comp:
                        return False
                    elif op == "$gt" and not (val > comp):
                        return False
                    elif op == "$gte" and not (val >= comp):
                        return False
                    elif op == "$lt" and not (val < comp):
                        return False
                    elif op == "$lte" and not (val <= comp):
                        return False
            else:
                if val != cond:
                    return False
        return True

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter: Optional[Dict] = None,
        include_energy: bool = False,
        include_metadata: bool = False,  # Changed default to False for legacy compat
    ) -> Any:
        """
        Búsqueda de vecinos más cercanos.

        Args:
            query: Vector de consulta [D]
            k: Número de resultados
            filter: Filtro de metadata
            include_energy: Si True, incluye información energética (requiere ebm_enabled)
            include_metadata: Si True, retorna List[DocResult]. False retorna tupla legacy.

        Returns:
            Si include_energy=True o include_metadata=True: List[DocResult]
            Si no: Tuple (vectors, alphas, kappas) para compatibilidad legacy.
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim > 1:
            query = query.squeeze()

        if self._use_lsh and self.lsh is not None:
            indices, distances = self.lsh.query(query, k=k)
            candidate_vectors = self.lsh.vectors[indices]
            dummy_alpha = np.ones(len(indices), dtype=np.float32)
            dummy_kappa = np.ones(len(indices), dtype=np.float32) * 10.0

            # Return legacy tuple if no metadata/energy requested
            if not include_energy and not include_metadata and filter is None:
                return candidate_vectors, dummy_alpha, dummy_kappa

            # Build DocResult list from LSH indices using _vector store
            lsh_ids = list(self._vectors.keys())
            results = []
            for i, idx in enumerate(indices):
                if i >= k:
                    break
                # Map LSH index to doc_id if possible
                doc_id = lsh_ids[idx] if idx < len(lsh_ids) else f"lsh_{idx}"
                if doc_id in self._deleted:
                    continue
                meta = self._metadata.get(doc_id, {}) if include_metadata else {}
                if filter and not self._match_filter(meta, filter):
                    continue
                score = float(distances[i])
                energy_val = None
                if include_energy and self.ebm_enabled and self._ebm_energy:
                    vec = self._vectors.get(doc_id, candidate_vectors[i])
                    energy_val = self._ebm_energy.energy(vec)
                results.append(DocResult(
                    id=doc_id,
                    score=score,
                    vector=candidate_vectors[i] if include_metadata else None,
                    metadata=meta,
                    document=self._documents.get(doc_id) if include_metadata else None,
                    energy=energy_val,
                    confidence=1.0 / (1.0 + energy_val) if energy_val is not None else None,
                ))
            return results[:k]

        raw = self.engine.search(query, k)

        if not include_energy and not include_metadata and filter is None:
            return raw  # Compatibilidad legacy

        # Construir lista de DocResults
        mu, alpha, kappa = raw
        active_ids = [d for d in self._vectors if d not in self._deleted]

        results = []
        for i in range(min(k, len(mu))):
            if i >= len(active_ids):
                break
            doc_id = active_ids[i]

            # Filtro de metadata
            if filter:
                meta = self._metadata.get(doc_id, {})
                if not self._match_filter(meta, filter):
                    continue

            score = float(np.asarray(alpha[i]).flat[0]) if i < len(alpha) else 0.0
            energy_val = None
            confidence_val = None

            if include_energy and self.ebm_enabled and self._ebm_energy:
                vec = self._vectors.get(doc_id)
                if vec is not None:
                    energy_val = self._ebm_energy.energy(vec)
                    confidence_val = 1.0 / (1.0 + energy_val)

            results.append(
                DocResult(
                    id=doc_id,
                    score=score,
                    vector=mu[i] if include_metadata else None,
                    metadata=(self._metadata.get(doc_id, {}) if include_metadata else {}),
                    document=(self._documents.get(doc_id) if include_metadata else None),
                    energy=energy_val,
                    confidence=confidence_val,
                )
            )

        return results[:k]

    def search_with_energy(self, query: np.ndarray, k: int = 10) -> SearchResult:
        """
        Búsqueda enriquecida con información energética.

        Requiere que ebm_enabled=True.

        Returns:
            SearchResult con resultados, energías, confianzas y zonas de incertidumbre.
        """
        import time

        start = time.time()

        results = self.search(query, k=k, include_energy=True, include_metadata=True)
        if isinstance(results, tuple):
            results = []  # Si no hay EBM, retornar vacío

        query_energy = 0.0
        if self.ebm_enabled and self._ebm_energy:
            query_energy = self._ebm_energy.energy(query)

        confidences = [r.confidence for r in results if r.confidence is not None]
        total_confidence = float(np.mean(confidences)) if confidences else 0.0

        uncertainty = []
        if self.ebm_enabled and self._ebm_exploration:
            uncertainty = self._ebm_exploration.find_high_energy_regions(
                topic_vector=query, n_regions=3
            )

        return SearchResult(
            results=results,
            query_energy=query_energy,
            total_confidence=total_confidence,
            uncertainty_regions=uncertainty,
            search_time_ms=(time.time() - start) * 1000,
        )

    def get_energy(self, vector: np.ndarray) -> float:
        """Calcula la energía de un vector (requiere ebm_enabled=True)."""
        if not self.ebm_enabled or self._ebm_energy is None:
            raise RuntimeError("EBM features no habilitadas. Usa enable_ebm=True o mode='ebm'.")
        return self._ebm_energy.energy(vector)

    def suggest_exploration(self, n: int = 3) -> List[ExplorationSuggestion]:
        """Sugerencias de exploración para el agente (requiere ebm_enabled=True)."""
        if not self.ebm_enabled or self._ebm_exploration is None:
            raise RuntimeError("EBM features no habilitadas.")
        ids = [d for d in self._vectors if d not in self._deleted]
        return self._ebm_exploration.suggest_exploration(
            current_knowledge=ids, n_suggestions=n
        )

    def find_knowledge_gaps(self, n: int = 5) -> List[EnergyRegion]:
        """Encuentra huecos en el conocimiento (requiere ebm_enabled=True)."""
        if not self.ebm_enabled or self._ebm_exploration is None:
            raise RuntimeError("EBM features no habilitadas.")
        return self._ebm_exploration.find_knowledge_gaps(n_gaps=n)

    def save(self, path: str):
        """Guarda el índice en disco (WAL + metadata + vectores)."""
        if self.storage:
            self.storage.checkpoint()
            # Guardar vectores en memoria al storage
            if self._vectors:
                vecs = np.array(list(self._vectors.values()), dtype=np.float32)
                ids = list(self._vectors.keys())
                self.storage.save_vectors(vecs, ids)
        else:
            # Fallback: si no hay storage, intentar con engine
            try:
                self.engine.load_optimized(path)
            except Exception:
                pass

    def load(self, path: str):
        """Carga un índice vectorial pre-computado."""
        self.engine.load_optimized(path)
        return self

    def get_stats(self) -> Dict:
        """Estadísticas del sistema."""
        base = {
            "total_documents": len(self._vectors),
            "active_documents": len([d for d in self._vectors if d not in self._deleted]),
            "deleted_documents": len(self._deleted),
            "lsh_active": self._use_lsh,
            "ebm_enabled": self.ebm_enabled,
        }
        if self.storage:
            storage_stats = self.storage.get_stats()
            base.update({"storage": storage_stats})
        if self.ebm_enabled and self._ebm_energy:
            base["free_energy"] = self._ebm_energy.free_energy()
        return base


# ---------------------------------------------------------------------------
# AdvancedVectorDB - CRUD + SOC + EBM completo
# ---------------------------------------------------------------------------


class AdvancedVectorDB(SimpleVectorDB):
    """
    AdvancedVectorDB: Suite completa para agentes autónomos.

    Hereda de SimpleVectorDB y añade:
    1. Exploración (Langevin Dynamics): SOC + exploración del espacio latente.
    2. Context Management (3-tier memory): Hot VRAM, Warm RAM, Cold SSD.
    3. SOC Engine: Self-Organized Criticality con avalanches automáticas.
    4. EBM completo: Energía, exploración, confianza.

    Uso:
        db = AdvancedVectorDB(latent_dim=768)
        db.add(['doc1'], vectors, metadata=[{'cat': 'ai'}])
        result = db.search_with_energy(query, k=10)
        state = db.check_criticality()
        db.relax(iterations=10)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        latent_dim: int = 640,
        storage_path: Optional[str] = None,
        enable_soc: bool = True,
        enable_energy_features: bool = True,
    ):
        super().__init__(
            device=device,
            latent_dim=latent_dim,
            storage_path=storage_path,
            enable_ebm=enable_energy_features,
            mode="ebm" if enable_energy_features else "standard",
        )

        # Reconfigurar engine con config avanzada
        config = M2MConfig.advanced(device=device)
        config.latent_dim = latent_dim
        self.engine = M2MEngine(config)

        # SOC Engine
        self.soc_enabled = enable_soc
        self._soc: Optional[SOCEngine] = None
        if enable_soc and self.ebm_enabled:
            self._soc = SOCEngine(self._ebm_energy)

    def _update_ebm_splats(self):
        """Actualiza splats en EBM y SOC."""
        super()._update_ebm_splats()
        if self._soc is not None and self.ebm_enabled:
            mu, alpha, kappa = self.engine.get_splats_arrays()
            if mu is not None and len(mu) > 0:
                self._soc.update_splats(mu, alpha, kappa)

    def consolidate(self, threshold: float = None) -> int:
        """Run Self-Organized Criticality memory consolidation."""
        return self.engine.m2m.consolidate(threshold)

    def check_criticality(self) -> CriticalityReport:
        """
        Verifica el estado de criticalidad del sistema.

        Returns:
            CriticalityReport con estado SUBCRITICAL/CRITICAL/SUPERCRITICAL.
        """
        if self._soc is None:
            raise RuntimeError("SOC no habilitado. Usa enable_soc=True.")
        return self._soc.check_criticality()

    def trigger_avalanche(self, seed_point: Optional[np.ndarray] = None) -> AvalancheResult:
        """
        Dispara una avalanche de reorganización del sistema.

        Args:
            seed_point: Punto semilla. Si None, usa el cluster de mayor energía.

        Returns:
            AvalancheResult con estadísticas.
        """
        if self._soc is None:
            raise RuntimeError("SOC no habilitado. Usa enable_soc=True.")
        return self._soc.trigger_avalanche(seed_point)

    def relax(self, iterations: int = 10) -> RelaxationResult:
        """
        Relaja el sistema hacia un estado de menor energía.

        Args:
            iterations: Iteraciones de relajación.

        Returns:
            RelaxationResult con energía inicial y final.
        """
        if self._soc is None:
            raise RuntimeError("SOC no habilitado. Usa enable_soc=True.")
        return self._soc.relax(iterations)

    def generate(self, query: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """Perform Langevin dynamics generative exploration starting from query."""
        return self.engine.generate(query, n_samples=n_steps)

    def get_stats(self) -> Dict:
        """Estadísticas completas del sistema incluyendo SOC."""
        base = super().get_stats()
        if self._soc is not None:
            base["soc"] = self._soc.get_statistics()
        return base


# ---------------------------------------------------------------------------
# M2MClient (SERVER mode)
# ---------------------------------------------------------------------------


class M2MClient:
    """
    Cliente para M2M en modo SERVER.

    Conecta a un servidor M2M vía REST API.

    Uso:
        client = M2MClient(host='localhost', port=8000, api_key='your-key')
        collection = client.create_collection('documents', dimension=768)
        collection.add(vectors=vecs, ids=ids, metadata=meta)
        results = collection.search(query, k=10)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.timeout = timeout

        try:
            import requests

            self._requests = requests
        except ImportError:
            self._requests = None
            print("[WARNING] 'requests' no instalado. Instala con: pip install requests")

    def _headers(self) -> Dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, path: str) -> Dict:
        if self._requests is None:
            raise RuntimeError("requests no instalado")
        r = self._requests.get(f"{self.base_url}{path}", headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: Dict) -> Dict:
        if self._requests is None:
            raise RuntimeError("requests no instalado")
        r = self._requests.post(
            f"{self.base_url}{path}", json=data, headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict:
        """Health check del servidor."""
        return self._get("/v1/health")

    def create_collection(self, name: str, dimension: int, **kwargs) -> "M2MCollection":
        """Crea una nueva colección en el servidor."""
        self._post(
            "/v1/collections",
            {"name": name, "dimension": dimension, **kwargs},
        )
        return M2MCollection(name=name, client=self)

    def get_collection(self, name: str) -> "M2MCollection":
        """Obtiene una colección existente."""
        self._get(f"/v1/collections/{name}")
        return M2MCollection(name=name, client=self)

    def list_collections(self) -> List[str]:
        """Lista todas las colecciones."""
        return self._get("/v1/collections").get("collections", [])

    def delete_collection(self, name: str) -> bool:
        """Elimina una colección."""
        if self._requests is None:
            raise RuntimeError("requests no instalado")
        r = self._requests.delete(
            f"{self.base_url}/v1/collections/{name}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        return r.status_code == 200

    def stats(self) -> Dict:
        """Estadísticas globales del servidor."""
        return self._get("/v1/stats")


class M2MCollection:
    """Colección en un servidor M2M remoto."""

    def __init__(self, name: str, client: M2MClient):
        self.name = name
        self._client = client

    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
    ) -> Dict:
        """Inserta vectores en la colección."""
        return self._client._post(
            f"/v1/collections/{self.name}/vectors",
            {
                "vectors": np.asarray(vectors, dtype=np.float32).tolist(),
                "ids": ids,
                "metadata": metadata,
                "documents": documents,
            },
        )

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        filter: Optional[Dict] = None,
        include: Optional[List[str]] = None,
    ) -> Dict:
        """Búsqueda de similitud."""
        return self._client._post(
            f"/v1/collections/{self.name}/search",
            {
                "vector": np.asarray(vector, dtype=np.float32).tolist(),
                "k": k,
                "filter": filter,
                "include": include or ["metadata", "documents"],
            },
        )

    def update(self, id: str, **kwargs) -> Dict:
        """Actualiza un vector."""
        if "vector" in kwargs and isinstance(kwargs["vector"], np.ndarray):
            kwargs["vector"] = kwargs["vector"].tolist()
        if self._client._requests is None:
            raise RuntimeError("requests no instalado")
        r = self._client._requests.put(
            f"{self._client.base_url}/v1/collections/{self.name}/vectors/{id}",
            json=kwargs,
            headers=self._client._headers(),
        )
        r.raise_for_status()
        return r.json()

    def delete(self, id: str) -> Dict:
        """Elimina un vector."""
        if self._client._requests is None:
            raise RuntimeError("requests no instalado")
        r = self._client._requests.delete(
            f"{self._client.base_url}/v1/collections/{self.name}/vectors/{id}",
            headers=self._client._headers(),
        )
        r.raise_for_status()
        return r.json()

    def get_energy_map(self, center: np.ndarray, radius: float = 1.0) -> Dict:
        """Obtiene el mapa de energía alrededor de un punto."""
        return self._client._post(
            f"/v1/collections/{self.name}/energy",
            {"center": center.tolist(), "radius": radius},
        )

    def find_knowledge_gaps(self) -> Dict:
        """Encuentra huecos en el conocimiento."""
        return self._client._get(f"/v1/collections/{self.name}/suggest")

    def suggest_exploration(self, n: int = 5) -> Dict:
        """Sugerencias de exploración."""
        return self._client._post(
            f"/v1/collections/{self.name}/explore", {"n_suggestions": n}
        )

    def get_stats(self) -> Dict:
        """Estadísticas de la colección."""
        return self._client._get(f"/v1/collections/{self.name}/stats")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_m2m(config: M2MConfig) -> M2MEngine:
    """Factory function to create M2M Engine."""
    return M2MEngine(config)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def main():
    """Main CLI entry point for M2M."""
    import argparse

    parser = argparse.ArgumentParser(
        description="M2M (Machine-to-Memory) - EBM Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "vulkan"],
        help="Device to use",
    )
    parser.add_argument("--n-splats", type=int, default=10000)
    parser.add_argument("--max-splats", type=int, default=100000)
    parser.add_argument("--knn-k", type=int, default=64)
    parser.add_argument("--enable-vulkan", action="store_true")
    parser.add_argument("--disable-vulkan", action="store_true")
    parser.add_argument(
        "--transform-dataset", nargs=2, metavar=("INPUT.npy", "OUTPUT.bin"),
        help="Transform embeddings dataset to M2M hierarchical splats",
    )
    parser.add_argument(
        "--load-optimized", type=str, metavar="INPUT.bin",
        help="Start M2M from a pre-transformed dataset",
    )
    parser.add_argument(
        "--mode", type=str, default="standard",
        choices=["edge", "standard", "ebm"],
        help="Modo de operación",
    )

    args = parser.parse_args()

    if args.transform_dataset:
        input_path, output_path = args.transform_dataset
        print("=" * 60)
        print("M2M Dataset Transformer")
        print("=" * 60)

        import numpy as np
        from dataset_transformer import M2MDatasetTransformer

        vectors = np.load(input_path)
        transformer = M2MDatasetTransformer(vectors)
        transformer.save_for_m2m(output_path)
        return

    device = args.device or "cpu"
    config = M2MConfig(
        device=device,
        n_splats_init=args.n_splats,
        max_splats=args.max_splats,
        knn_k=args.knn_k,
        enable_vulkan=args.enable_vulkan or (not args.disable_vulkan and device == "cuda"),
    )

    print("=" * 60)
    print("M2M (Machine-to-Memory) EBM Vector Database")
    print("=" * 60)

    m2m = create_m2m(config)

    if args.load_optimized:
        print(f"[LOAD] Loading optimized dataset from {args.load_optimized}...")
        try:
            m2m.load_optimized(args.load_optimized)
            print("[LOAD] Successfully booted from pre-computed hierarchy.")
        except Exception as e:
            print(f"[ERROR] Failed to load optimized dataset: {e}")
            return
    else:
        print("[EXAMPLE] Adding 100 random splats...")
        random_vectors = np.random.randn(100, config.latent_dim).astype(np.float32)
        n_added = m2m.add_splats(random_vectors)
        print(f"[EXAMPLE] Added {n_added} splats")

    stats = m2m.get_statistics()
    print("\n[STATS]")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n[SUCCESS] M2M initialized successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
