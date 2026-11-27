#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M (Machine-to-Memory) - EBM Vector Database
High-performance Gaussian Splat storage and retrieval with Energy-Based Model features.

Version: 2.4.0
Author: Brian Schwabauer
License: AGPL-3.0

Modos de operación:
  - EMBEDDED: SimpleVectorDB / AdvancedVectorDB (librería Python)
  - SERVER:   M2MClient (cliente REST)
  - CLUSTER:  M2MCluster (cluster distribuido con router energético opcional)
"""

__version__ = "2.4.0"
__author__ = "Brian Schwabauer"
__email__ = "schwabauerbriantomas@gmail.com"
__license__ = "AGPL-3.0"

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import json
import operator
from collections import OrderedDict

logger = logging.getLogger(__name__)

# M2M Core Modules
try:
    from .backend_comm import (
        BackendComm,
        BackendHealth,
        BackendMessage,
        BackendMetrics,
        BackendMsgType,
    )
    from .config import M2MConfig
    from .ebm import EBMEnergy, EBMExploration, SOCEngine
    from .ebm.energy_api import EnergyResult as EnergyResult  # re-export
    from .ebm.energy_api import confidence_from_energy, splat_energy
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
    from .hrm2_engine import HRM2Config
    from .hrm2_engine import HRM2Engine
    from .mapreduce_indexer import parallel_index
    from .quality_reflector import QualityLevel, QualityReflector, QualityReport
    from .query_router import QueryProfile, QueryRouter, SearchStrategy
    from .search_supervisor import BackendType, QueryComplexity, SearchSupervisor
    from .semantic_memory import MemoryResult, SemanticMemoryDB
    from .splats import SplatStore
    from .storage import M2MPersistence
    from .storage import WriteAheadLog as WriteAheadLog
except ImportError as exc:
    # audit r3a-10: este branch dejaba `M2MEngine = None` y el resto de los nombres
    # sin definir, así que una instalación rota explotaba con un NameError confuso
    # en el primer uso. Un módulo del núcleo que no importa = instalación
    # incompleta: fallar acá, con la causa real.
    raise ImportError(
        f"m2m: no se pudo importar un módulo del núcleo ({exc!r}); "
        "la instalación está incompleta o rota"
    ) from exc

# audit r3a-10: acá había redefiniciones "fallback" de normalize_sphere,
# geodesic_distance, exp_map, log_map y project_to_tangent que SOMBREABAN las de
# .geometry importadas arriba: `from m2m import exp_map` devolvía `base + tangent`.
# Las únicas implementaciones son las de m2m.geometry (operaciones riemannianas
# sobre la esfera unitaria). normalize_sphere es numéricamente idéntico al
# fallback (mismo eje -1, mismo eps), así que SimpleVectorDB no cambia.
#
# `M2MEngine` en __all__ es el envoltorio de M2MMemory definido abajo (lo que
# `from m2m import M2MEngine` siempre devolvió); el selector de hardware es
# m2m.engine.M2MEngine.

__all__ = [
    "__version__",
    "AdvancedVectorDB",
    "AvalancheResult",
    "BackendComm",
    "BackendHealth",
    "BackendMessage",
    "BackendMetrics",
    "BackendMsgType",
    "BackendType",
    "CriticalityReport",
    "DeleteResult",
    "DocResult",
    "EBMEnergy",
    "EBMExploration",
    "EnergyFunction",
    "EnergyRegion",
    "EnergyResult",
    "EntityCandidate",
    "ExplorationSuggestion",
    "FeedbackResult",
    "GaussianGraphStore",
    "GraphEdge",
    "GraphSplat",
    "HRM2Config",
    "HRM2Engine",
    "M2MClient",
    "M2MCollection",
    "M2MConfig",
    "M2MEngine",
    "M2MEntityExtractor",
    "M2MGraphEntityExtractor",
    "M2MMemory",
    "M2MPersistence",
    "MemoryResult",
    "NodeType",
    "QualityLevel",
    "QualityReflector",
    "QualityReport",
    "QueryComplexity",
    "QueryProfile",
    "QueryRouter",
    "RelaxationResult",
    "SOCEngine",
    "SearchResult",
    "SearchStrategy",
    "SearchSupervisor",
    "SemanticMemoryDB",
    "SimpleVectorDB",
    "SplatStore",
    "UpdateResult",
    "WriteAheadLog",
    "create_m2m",
    "exp_map",
    "geodesic_distance",
    "log_map",
    "main",
    "match_metadata_filter",
    "normalize_sphere",
    "parallel_index",
    "project_to_tangent",
    "validate_metadata_filter",
]

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

        logger.info(
            "M2M initialized on %s (compute_device=%s)", config.device, config.compute_device
        )
        logger.info("Latent dim: %s", config.latent_dim)
        logger.info("Max splats: %s", config.max_splats)
        logger.info("3-tier memory: %s", config.enable_3_tier_memory)

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
            - self.config.global_temperature * H_q
        )
        return E

    def retrieve(
        self, query: np.ndarray, k: int = None, lod: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve k-nearest neighbors from splat store."""
        if k is None:
            k = self.config.knn_k
        query_norm = normalize_sphere(query)
        neighbors_mu, neighbors_alpha, neighbors_kappa, neighbors_idx = self.splats.find_neighbors(
            query_norm, k, lod=lod
        )
        return neighbors_mu, neighbors_alpha, neighbors_kappa, neighbors_idx

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
            splats_to_remove = np.where(self.splats.alpha[: self.splats.n_active] < threshold)[0]
        else:
            splats_to_remove = []

        n_removed = 0
        for splat_idx in splats_to_remove:
            self.splats.mu[splat_idx] = float("inf")
            n_removed += 1

        if n_removed > 0:
            self.splats.compact()
            # C-04 fix: rebuild HRM2 index after consolidate. retrain=True
            # (r1-25): compact() desplaza las filas de mu, el camino attach
            # dejaría coarse_assignments/_cluster_masks con posiciones viejas.
            self.splats.build_index(retrain=True)
            logger.info("SOC: Consolidated %d splats", n_removed)

        return n_removed

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the M2M system."""
        return {
            "n_active_splats": self.splats.n_active,
            "max_splats": self.splats.max_splats,
            "mean_frequency": self.splats.frequency[: self.splats.n_active].mean().item(),
            "mean_kappa": self.splats.kappa[: self.splats.n_active].mean().item(),
            "mean_alpha": self.splats.alpha[: self.splats.n_active].mean().item(),
            "entropy": self.splats.entropy().item(),
            "soc_phi": (self.phi.item() if hasattr(self.phi, "item") else float(self.phi)),
        }

    def forward(self, x: np.ndarray, mode: str = "energy") -> np.ndarray:
        """Forward pass of M2M system."""
        if mode == "energy":
            return self.compute_energy(x)
        elif mode == "retrieve":
            neighbors_mu, _, _, _ = self.retrieve(x)
            return neighbors_mu
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# M2MEngine (internal)
# ---------------------------------------------------------------------------


class M2MEngine:
    """M2M High-Performance Engine: envoltorio interno de :class:`M2MMemory`.

    Este NO es el selector de backend de hardware. Ese es ``m2m.engine.M2MEngine``
    (``src/m2m/engine.py``), que ``HRM2Engine`` crea como ``router`` y es el único
    que decide qué corre (CUDA > Vulkan > CPU, según disponibilidad real).
    ``SimpleVectorDB.engine`` es una instancia de ESTA clase y
    ``from m2m import M2MEngine`` la sigue devolviendo (compatibilidad): todos
    los métodos delegan en ``self.m2m`` (la ``M2MMemory``).

    Estado del backend (audit r3a-09): :attr:`backend` y :meth:`get_backend_info`
    delegan en el selector real. :attr:`vulkan_engine` se conserva por
    compatibilidad y siempre es ``None`` (Vulkan se maneja en ``gpu_vector_index``).
    """

    def __init__(self, config: M2MConfig):
        self.config = config
        self.m2m = M2MMemory(config)
        self.vulkan_engine = None  # Vulkan GPU is handled via gpu_vector_index

        # audit r3a-09: un solo log de backend, el del estado EFECTIVO que resolvió
        # el selector real (m2m.engine.M2MEngine, que ya loguea con WARNING cada
        # caída con su motivo). Antes este bloque logueaba 'Vulkan GPU acceleration
        # enabled' con enable_vulkan aunque Vulkan hubiera fallado y corriera en CPU.
        info = self.get_backend_info()
        logger.info(
            "M2M backend efectivo: %s (pedido: %s)", info["effective"], info["requested"]
        )

    @property
    def backend(self) -> str:
        """Backend EFECTIVO en el que corre: 'cuda' | 'vulkan' | 'cpu'."""
        return self.get_backend_info()["effective"]

    def get_backend_info(self) -> Dict[str, Any]:
        """Delega en el selector real: ``{'requested', 'effective', 'reasons'}``."""
        router = getattr(
            getattr(getattr(self.m2m, "splats", None), "engine", None), "router", None
        )
        if not hasattr(router, "get_backend_info"):
            return {"requested": "cpu", "effective": "cpu", "reasons": {}}
        return router.get_backend_info()
    def add_splats(self, vectors: np.ndarray, labels: List[str] = None) -> int:
        """Add new splats to the system (batch-vectorized for large datasets)."""
        vectors_norm = normalize_sphere(vectors)

        # Batch path: pass entire array at once (1000x faster than per-vector loop)
        n_added = self.m2m.splats.add_splat(vectors_norm)

        if n_added and n_added > 0:
            self.m2m.splats.build_index()

        logger.info("Added %d splats and built HRM2 index", n_added)
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
        return self.m2m.forward(x)

    def load_optimized(self, path: str):
        """Loads optimized dataset with pre-computed Gaussian Splats.

        Levanta FileNotFoundError si `path` no existe y propaga los errores de
        parseo: antes los tragaba con un warning y devolvía self, así que
        SimpleVectorDB.load() de una ruta inexistente "funcionaba" (audit r1-17).
        """
        from pathlib import Path

        if not path or not Path(str(path)).exists():
            raise FileNotFoundError(f"load_optimized(): dataset not found: {path!r}")
        from .loaders.optimized_loader import load_m2m_dataset

        splats = load_m2m_dataset(str(path))
        self._build_hrm2_from_splats(splats)
        return self

    def _build_hrm2_from_splats(self, splats):
        """Builds HRM2 index directly from pre-computed splats."""
        self.m2m.splats._build_hrm2_from_splats(splats)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def _cosine_to_query(qn: np.ndarray, mu_i: np.ndarray) -> float:
    """Similitud coseno entre la query normalizada y el vector almacenado (normalizado)."""
    return float(np.clip(np.dot(qn, mu_i), -1.0, 1.0))


def _result_energy(qn: np.ndarray, mu_i, alpha_i: float, kappa_i: float) -> float:
    """Energía por resultado [audit r3a-02]: la implementación canónica aplicada al
    splat de UN resultado, Eᵢ(q̂) = -ln αᵢ + κᵢ·‖q̂−μᵢ‖² (tope OOD_ENERGY). αᵢ y κᵢ
    son los del splat que produjo el resultado (los mismos de ``DocResult.alpha``)."""
    return splat_energy(qn, np.asarray(mu_i, dtype=np.float32)[None, :], [alpha_i], [kappa_i])


def _l2_to_query(qn: np.ndarray, mu_i: np.ndarray) -> float:
    """Distancia L2 entre la query normalizada y el vector almacenado (d² = 2 − 2·cos)."""
    diff = np.asarray(mu_i, dtype=np.float64) - qn
    return float(np.sqrt(np.maximum((diff**2).sum(), 0.0)))


class DocResult:
    """Resultado de una operación de base de datos.

    Semántica de los campos [audit r1-13]:

    - ``score``: similitud COSENO en [-1, 1] entre la query normalizada a la esfera y
      el vector almacenado (normalizado). IGUAL en todos los caminos: ``search()`` con
      o sin metadata, con o sin filtro, con o sin energía, modo LSH y
      ``search_batch(include_metadata=True)``. Mayor = más parecido. (Antes eran
      varias semánticas: α = 1.0 constante sin filtro, -dist con filtro, -sqrt(d²)
      contra la query cruda en batch y la distancia propia del LSH.)
    - ``alpha``: peso αᵢ del splat (lo que antes devolvía ``score`` sin filtro).
      None en modo LSH (ahí no hay splats por documento).
    - ``distance``: distancia L2 entre la query normalizada y el vector almacenado;
      sobre la esfera ``distance² = 2 − 2·score``.
    - ``energy`` (solo con ``include_energy=True`` y ``enable_ebm=True``): la energía
      canónica del paisaje (``m2m.ebm.energy_api.splat_energy``) de la query normalizada
      bajo el splat de ESTE resultado, ``Eᵢ(q̂) = -ln αᵢ + κᵢ·‖q̂−μᵢ‖²`` con tope
      ``OOD_ENERGY`` (10). Depende de la relación query-resultado, así que resultados
      distintos tienen energías distintas [audit r3a-02]; es ≥ la energía del paisaje
      en la query (``-ln Σⱼ exp(−Eⱼ)``). Antes se evaluaba el paisaje en la posición
      del propio documento: un valor igual para todos los resultados. None si no se
      pidió energía.
    - ``confidence``: ``sigmoid(−energy) = 1 / (1 + e^E)`` ∈ (0, 1), monotona
      decreciente en la energía (elección del pack P7a [audit r2-15]; antes
      ``1/(1+E)``, que daba negativa para E < −1). None si no se pidió energía.
    - ``vector``: el vector almacenado NORMALIZADO (μᵢ), no el crudo que se insertó;
      None en ``search_batch(include_metadata=True)``.
    """

    def __init__(
        self,
        id: str,
        score: float = 0.0,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        document: Optional[str] = None,
        energy: Optional[float] = None,
        confidence: Optional[float] = None,
        alpha: Optional[float] = None,
        distance: Optional[float] = None,
    ):
        self.id = id
        self.score = score
        self.vector = vector
        self.metadata = metadata or {}
        self.document = document
        self.energy = energy
        self.confidence = confidence
        self.alpha = alpha  # aditivo [audit r1-13]: default None, no rompe constructores
        self.distance = distance

    def __repr__(self):
        return (
            f"DocResult(id={self.id!r}, score={self.score:.4f}, "
            f"alpha={self.alpha}, distance={self.distance}, "
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


# ---------------------------------------------------------------------------
# Filtros de metadata (audit r1-14)
# Un solo lugar de semantica para SimpleVectorDB._match_filter (memoria) y
# M2MPersistence._matches_filter (storage), que antes duplicaban el bucle y
# divergian en silencio: un operador desconocido ($in, typos como $gtt) o una
# condicion {} caian en la rama default y NO filtraban (dejaban pasar todo, y
# delete(filter=...) borraba de mas). Ahora: fail closed (ValueError).
# ---------------------------------------------------------------------------
_METADATA_FILTER_OPERATORS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin")
_FILTER_IN_OPS = ("$in", "$nin")
# Contenedores: unicos argumentos validos de $in/$nin y NUNCA validos como
# condicion directa (ambiguos: {"c": [3]} no es ni igualdad ni $in).
_FILTER_CONTAINER_TYPES = (list, tuple, set, frozenset, np.ndarray)


def _invalid_filter(msg: str) -> ValueError:
    return ValueError(f"filtro de metadata invalido: {msg}")


def validate_metadata_filter(filter_dict) -> None:
    """Valida la ESTRUCTURA de un filtro sin evaluarlo contra ningun doc (fail closed).

    Acepta ``None`` (sin filtro), un callable ``metadata -> bool`` o un dict
    ``{campo: valor | {op: valor}}`` con ops en ``_METADATA_FILTER_OPERATORS``.
    Operador desconocido, condicion ``{}``, filtro ``{}`` o tipo invalido ->
    ``ValueError``. Se llama antes de tocar el indice (o de borrar algo, via
    ``_match_filter``), asi un filtro mal escrito nunca "deja pasar todo".
    """
    if filter_dict is None or callable(filter_dict):
        return
    if not isinstance(filter_dict, dict):
        raise _invalid_filter(
            "se esperaba un dict o un callable metadata -> bool, "
            f"se recibio {type(filter_dict).__name__}"
        )
    if not filter_dict:
        raise _invalid_filter("filter vacio ({}) ; pasa filter=None para no filtrar")
    for key, cond in filter_dict.items():
        if isinstance(cond, dict):
            if not cond:
                raise _invalid_filter(f"condicion vacia para el campo {key!r} ({{}})")
            for op, comp in cond.items():
                if op not in _METADATA_FILTER_OPERATORS:
                    raise _invalid_filter(
                        f"operador {op!r} no soportado para {key!r} "
                        f"(soportados: {', '.join(_METADATA_FILTER_OPERATORS)})"
                    )
                if op in _FILTER_IN_OPS and not isinstance(comp, _FILTER_CONTAINER_TYPES):
                    raise _invalid_filter(
                        f"{op} requiere una lista/conjunto de valores para {key!r}, "
                        f"se recibio {type(comp).__name__}"
                    )
        elif isinstance(cond, _FILTER_CONTAINER_TYPES):
            raise _invalid_filter(
                f"condicion invalida para {key!r}: se esperaba un valor o un dict de "
                f"operadores (p. ej. {{'$in': [...]}}), se recibio {type(cond).__name__}"
            )


def match_metadata_filter(meta: dict, filter_dict) -> bool:
    """Evalua un filtro de metadata contra un doc (misma semantica en memoria y storage).

    ``filter_dict`` es ``None`` (matchea todo), un callable ``metadata -> bool`` o
    un dict ``{campo: valor | {op: valor}}``: todas las condiciones deben
    cumplirse y un campo ausente en ``meta`` no cumple. La estructura se valida
    primero: operador desconocido / condicion ``{}`` / tipos incomparables ->
    ``ValueError`` [audit r1-14].
    """
    validate_metadata_filter(filter_dict)
    if filter_dict is None:
        return True
    if callable(filter_dict):
        return bool(filter_dict(meta))
    for key, cond in filter_dict.items():
        if key not in meta:
            return False
        val = meta[key]
        if isinstance(cond, dict):
            for op, comp in cond.items():
                try:
                    if op == "$eq":
                        ok = bool(val == comp)
                    elif op == "$ne":
                        ok = bool(val != comp)
                    elif op == "$gt":
                        ok = bool(val > comp)
                    elif op == "$gte":
                        ok = bool(val >= comp)
                    elif op == "$lt":
                        ok = bool(val < comp)
                    elif op == "$lte":
                        ok = bool(val <= comp)
                    elif op == "$in":
                        ok = val in comp
                    else:  # $nin
                        ok = val not in comp
                except (TypeError, ValueError) as e:
                    raise _invalid_filter(
                        f"no se puede evaluar {op} en {key!r}: valor {val!r} vs "
                        f"comparador {comp!r} ({e})"
                    ) from e
                if not ok:
                    return False
        elif val != cond:
            return False
    return True


# ---------------------------------------------------------------------------
# Cache de máscaras de filtro + evaluación vectorizada (P13, medición del orquestador)
# search(filter=dict) armaba `allowed` con un bucle O(n) EN PYTHON por query (dos
# funciones por posición): 20k docs -> 32-37 ms/query (27-31 QPS) vs 0.5 ms sin
# filtro; 100k -> 190 ms. Ahora: (1) la máscara se cachea por (forma canónica del
# filtro, _meta_version, n_splats) y TODO writer de metadata / posiciones /
# soft-deletes incrementa _meta_version; (2) la primera evaluación va por columnas
# de metadata (cacheadas por versión) con operadores vectorizados; callable u
# operadores no vectorizables caen al camino Python de siempre. Paridad EXACTA con
# match_metadata_filter (helper único de semántica): ante la duda se vuelve a Python.
# ---------------------------------------------------------------------------
_FILTER_MASK_CACHE_MAX = 32  # entradas (LRU) de máscaras por filtro
_META_COL_CACHE_MAX = 64  # columnas de metadata cacheadas por versión
_FILTER_SCALARS = (type(None), bool, int, float, str, bytes)  # tipos EXACTOS vectorizables
_FILTER_VEC_OPS = {
    "$eq": operator.eq,
    "$ne": operator.ne,
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
}


def _canonical_filter_value(v):
    """Valor canónico (JSON) de un componente de un filtro: escalares tal cual,
    listas/tuplas etiquetadas por tipo, sets ordenados, numpy pasado a Python,
    dicts como pares ordenados. None si no es representable (-> no se cachea)."""
    t = type(v)
    if t is bytes:
        return ["bytes", v.hex()]
    if t in (type(None), bool, int, float, str):
        return v
    if t in (list, tuple, set, frozenset):
        items = [_canonical_filter_value(x) for x in v]
        if any(i is None for i in items):
            return None
        if t in (set, frozenset):
            items.sort(key=lambda i: json.dumps(i, sort_keys=True))
            return ["set", items]
        return [t.__name__, items]
    if isinstance(v, np.ndarray):
        items = [_canonical_filter_value(x) for x in v.reshape(-1).tolist()]
        if any(i is None for i in items):
            return None
        return ["ndarray", v.dtype.str, [int(d) for d in v.shape], items]
    if isinstance(v, np.generic):
        return _canonical_filter_value(v.item())
    if isinstance(v, dict):
        pairs = []
        for k, val in v.items():
            ck = _canonical_filter_value(k)
            cv = _canonical_filter_value(val)
            if ck is None or cv is None:
                return None
            pairs.append([ck, cv])
        pairs.sort(key=lambda pr: json.dumps(pr, sort_keys=True))
        return ["dict", pairs]
    return None


def _canonical_filter(filter_dict) -> Optional[str]:
    """Clave de cache: forma canónica (JSON ordenado) de un filtro dict. Dos filtros
    con la misma clave evalúan igual sobre los mismos docs. None si tiene un
    componente no representable: entonces no se cachea (se evalúa en Python en
    cada query, como antes) [P13]."""
    canon = _canonical_filter_value(filter_dict)
    if canon is None:
        return None
    try:
        return json.dumps(canon, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - defensivo
        return None


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
        # T2: default False. Evidencia (bench_t2_results.json + sondas, OMP=4):
        #   rag_test 659x384:  LSH single 76 QPS R@10=0.717 | HRM2 4269 QPS R@10=1.0
        #   uniforme 5000x384: LSH batch R@10=0.116            | HRM2 batch R@10=0.966
        #   uniforme 20000x640 (design target de LSH): LSH batch R@10=0.108
        #                                              | HRM2 batch 1055 QPS R@10=0.868
        # LSH pierde recall Y QPS en todos los escenarios medidos; el routing
        # LSH de search()/search_batch() queda disponible via flag explicito.
        enable_lsh_fallback: bool = False,
        lsh_threshold: float = 0.15,
        storage_path: Optional[str] = None,
        enable_wal: bool = True,
        enable_ebm: bool = False,
        mode: str = "standard",  # 'edge', 'standard', 'ebm'
        max_splats: Optional[int] = None,  # auto-scale if None
        n_probe: Optional[int] = None,  # HRM2 clusters to probe
        # Orquestador: puente del gate de modulus a la API publica.
        # None = no tocar el default de M2MConfig (backward-compat exacto).
        gate_enabled: Optional[bool] = None,
        k_selection: Optional[str] = None,  # 'fixed' | 'silhouette' | 'gate'
        gate_target_docs_per_cluster: Optional[int] = None,
        _config: Optional["M2MConfig"] = None,  # internal: allows subclass to pass custom config
    ):
        if _config is not None:
            config = _config
            config.latent_dim = latent_dim
        else:
            config = M2MConfig.simple(device=device)
            config.latent_dim = latent_dim

        # Orquestador: aplicar los flags del gate sobre la config resuelta.
        if gate_enabled is not None:
            config.gate_enabled = gate_enabled
        if k_selection is not None:
            config.k_selection = k_selection
        if gate_target_docs_per_cluster is not None:
            config.gate_target_docs_per_cluster = gate_target_docs_per_cluster

        # Auto-scale max_splats if not specified (use 2x default to allow growth)
        if max_splats is not None:
            config.max_splats = max_splats

        self.engine = M2MEngine(config)

        # Configure n_probe on the HRM2 engine
        if n_probe is not None:
            self.engine.m2m.splats.engine.n_probe = n_probe

        self.latent_dim = latent_dim
        self.max_splats = config.max_splats
        self.n_probe = n_probe if n_probe is not None else getattr(self.engine.m2m.splats.engine, 'n_probe', None)

        self.enable_lsh_fallback = enable_lsh_fallback
        self.lsh_threshold = lsh_threshold
        self.lsh = None
        self._use_lsh = False
        self._lsh_id_map: List[str] = []  # maps LSH positional index -> doc_id
        # LSH fallback decision is made ONCE (first add with n>=3): recomputing
        # KMeans+silhouette on every add cost ~0.3-0.5s per batch.
        self._lsh_decided = False
        self._lsh_active_mode = False
        self._splat_id_order: List[str] = (
            []
        )  # maps splat position -> doc_id for consolidate cleanup

        # Almacenamiento interno en memoria (para compatibilidad)
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict] = {}
        self._documents: Dict[str, str] = {}
        self._deleted: set = set()
        # [P13] Cache de máscaras de filtro (ver _filter_position_mask): la clave
        # lleva _meta_version, que incrementa TODO writer de metadata / posiciones /
        # soft-deletes (add, update, delete, _restore_from_storage, consolidate).
        # Mutar un dict de metadata IN PLACE desde afuera no incrementa la versión:
        # usar update(metadata=...). Se lee/escribe sólo bajo _rw_lock.
        self._meta_version = 0
        self._filter_mask_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._meta_cols: Dict[Any, tuple] = {}
        self._meta_cols_ver: Optional[tuple] = None
        self._alive_ver: Optional[tuple] = None
        self._alive_mask: Optional[np.ndarray] = None

        # Thread safety: protect all mutations to shared dicts
        import threading as _threading
        self._rw_lock = _threading.RLock()
        # audit r1-03: search()/search_batch() (región de otro pack) entran al
        # índice por engine.search() sin tomar el lock; sin esto un lector veía
        # el índice a medias mientras un writer corría build_index (KMeans sin
        # fitear, máscaras desalineadas). Se toma el mismo RLock en la frontera
        # del engine: reentrante para update() -> add() en el mismo hilo. Cuando
        # search() tome el lock explícitamente, este wrapper sobra.
        _engine_search = self.engine.search
        _lock = self._rw_lock

        def _locked_engine_search(*args, **kwargs):
            with _lock:
                return _engine_search(*args, **kwargs)

        self.engine.search = _locked_engine_search

        # Persistencia
        self.storage = None
        if storage_path and mode != "edge":
            self.storage = M2MPersistence(storage_path, enable_wal=(enable_wal and mode != "edge"))
            self._restore_from_storage()

        # EBM
        self.ebm_enabled = enable_ebm or mode == "ebm"
        self._ebm_energy: Optional[EBMEnergy] = None
        self._ebm_exploration: Optional[EBMExploration] = None
        if self.ebm_enabled:
            self._ebm_energy = EBMEnergy()
            self._ebm_exploration = EBMExploration(self._ebm_energy)

    def _restore_from_storage(self):
        """Restaura los documentos desde storage y RECONSTRUYE el índice.

        Antes este método sólo repoblaba _vectors/_metadata/_documents: tras
        save() + reopen quedaban n_active_splats=0 y _splat_id_order=[], así que
        search()/search_batch() devolvían vacío sin error (audit r1-01). Ahora:

        1. lee TODAS las filas de metadata.db (incl. soft-deleted) en orden
           estable (shard_idx, vector_idx) = orden de inserción de add();
        2. indexa los splats por la MISMA ruta que un add() del mismo tamaño
           (engine.add_splats -> build_index con KMeans completo);
        3. _splat_id_order queda alineado 1:1 con las posiciones de splat;
        4. _deleted se restaura: un doc borrado antes del save() no reaparece;
        5. si n_restored > max_splats: indexa los primeros max_splats en ese
           orden y lo avisa con un warning (los demás siguen en storage);
        6. con enable_lsh_fallback decide LSH igual que add() y reconstruye el
           índice LSH;
        7. re-aplica el estado aprendido de los splats (α/κ/μ por id,
           ``energy/landscape.npz``) también en esta clase [audit P15].

        Idempotente (load() lo reusa): resetea el estado en memoria y el engine.
        Con M2M_HMAC_SECRET seteado verifica antes la firma del storage.
        """
        if not self.storage:
            return
        from .storage.persistence import IndexTamperedError

        # Reset del estado en memoria + engine (idempotente)
        self._meta_version += 1  # [P13] invalida el cache de máscaras de filtro
        self._vectors.clear()
        self._metadata.clear()
        self._documents.clear()
        self._deleted.clear()
        self._splat_id_order = []
        self._lsh_id_map = []
        self.lsh = None
        self._use_lsh = False
        self._lsh_decided = False
        self._lsh_active_mode = False
        store = self.engine.m2m.splats
        store.n_active = 0
        store._next_id = 0
        store._last_trained_n = 0
        store.engine.clear()
        store._m_sq_dirty = True
        store._cuda_dirty = True
        try:
            # r3a-17: sin secreto es no-op; con secreto levanta IndexTamperedError
            self.storage.verify_manifest()
            rows = self.storage.get_all_documents(include_deleted=True)
            if not rows:
                return
            # shard_idx es 1-based y corresponde directo al nombre de archivo
            # (1 -> shard_001, 2 -> shard_002, ...).
            shards = {}
            restored = []  # (doc_id, vector, metadata, document, deleted)
            missing = 0
            for row in rows:
                shard_idx = row.get("shard_idx")
                vector_idx = row.get("vector_idx")
                if shard_idx is None or vector_idx is None:
                    missing += 1
                    continue
                shard_name = f"shard_{max(1, int(shard_idx)):03d}"
                if shard_name not in shards:
                    shards[shard_name] = self.storage.load_vectors(shard_name)
                vecs = shards[shard_name]
                if vecs is None or not (0 <= int(vector_idx) < len(vecs)):
                    missing += 1
                    continue
                restored.append(
                    (
                        row["id"],
                        vecs[int(vector_idx)],
                        row.get("metadata") or {},
                        row.get("document"),
                        bool(row.get("deleted")),
                    )
                )
            n = len(restored)
            if missing:
                logger.warning(
                    "Restore: %d fila(s) de metadata sin vector localizable "
                    "(shard_idx/vector_idx inválidos); se ignoran",
                    missing,
                )
            if n == 0:
                return
            vectors = np.ascontiguousarray(
                np.stack([r[1] for r in restored]), dtype=np.float32
            )

            # LSH: misma decisión única que add() (primer lote con n>=3)
            if self.enable_lsh_fallback and n >= 3:
                self._lsh_decided = True
                silhouette = self._compute_silhouette(vectors)
                self._lsh_active_mode = silhouette < self.lsh_threshold
            if self._lsh_active_mode and self.enable_lsh_fallback:
                try:
                    from .lsh_index import CrossPolytopeLSH, LSHConfig

                    config = LSHConfig(
                        dim=self.engine.config.latent_dim,
                        n_tables=15,
                        n_bits=18,
                        n_probes=50,
                        n_candidates=500,
                    )
                    self.lsh = CrossPolytopeLSH(config)
                    self.lsh.index(vectors)
                    self._use_lsh = True
                    with self._rw_lock:
                        for doc_id, vec, meta, doc, deleted in restored:
                            self._vectors[doc_id] = np.array(vec, dtype=np.float32, copy=True)
                            self._metadata[doc_id] = meta
                            self._documents[doc_id] = doc
                            if deleted:
                                self._deleted.add(doc_id)
                        self._lsh_id_map = [r[0] for r in restored]
                        self._splat_id_order = [r[0] for r in restored]
                        self._meta_version += 1  # [P13] metadata / posiciones nuevas
                    logger.info(
                        "Restored %d documents from storage (LSH index rebuilt)", n
                    )
                    return
                except ImportError as e:
                    logger.warning("Could not load LSH module: %s.", e)
                    self._lsh_active_mode = False
                    self._use_lsh = False
                    self.lsh = None

            # Splats + IVF: misma ruta que add() (normaliza, add_splat, build_index)
            n_added = self.engine.add_splats(vectors)
            if n_added < n:
                skipped = [r[0] for r in restored[n_added:]]
                logger.warning(
                    "Restore: max_splats=%d alcanzado: %d de %d documentos restaurados NO se "
                    "indexan (siguen en storage; reabrir con max_splats mayor para "
                    "recuperarlos). Primeros ids sin indexar: %s",
                    self.max_splats,
                    n - n_added,
                    n,
                    skipped[:20],
                )
            with self._rw_lock:
                for doc_id, vec, meta, doc, deleted in restored[:n_added]:
                    self._vectors[doc_id] = np.array(vec, dtype=np.float32, copy=True)
                    self._metadata[doc_id] = meta
                    self._documents[doc_id] = doc
                    if deleted:
                        self._deleted.add(doc_id)
                self._splat_id_order = [r[0] for r in restored[:n_added]]
                self._meta_version += 1  # [P13] metadata / posiciones nuevas
            logger.info(
                "Restored %d documents from storage (%d splats indexed, %d soft-deleted)",
                n,
                n_added,
                len(self._deleted),
            )
            # Estado aprendido (α/κ/μ por id de feedback/avalanche/relax): antes sólo
            # lo re-aplicaba el override de AdvancedVectorDB; un SimpleVectorDB con
            # storage_path perdía lo aprendido en cada reopen [audit P15].
            try:
                self._restore_splat_state()
            except Exception as e:  # best-effort, igual que _write_splat_state
                logger.warning("splat state not restored on reopen: %s: %s", type(e).__name__, e)
        except IndexTamperedError:
            raise  # firma inválida: no se carga nada (audit r3a-17)
        except Exception as e:
            logger.warning("Failed to restore from storage: %s: %s", type(e).__name__, e)

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
            logger.warning("scikit-learn is required for data distribution analysis.")
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

    def _publish_added(
        self,
        ids: List[Any],
        accepted: List[int],
        vectors: np.ndarray,
        metadata: Optional[List[Dict]],
        documents: Optional[List[str]],
    ) -> None:
        """Publica un lote aceptado por add(): dicts (copy-on-write) + storage.

        Copy-on-write (audit r1-03): un lector que está iterando ``_vectors``
        (``search`` arma la lista de activos así) sigue viendo el dict viejo,
        completo y consistente, en vez de un RuntimeError por "dictionary
        changed size during iteration". Requiere el lock tomado (el storage va
        dentro: dos writers intercalando save_vectors/save_metadata rompían el
        shard_idx de la metadata).
        """
        if not accepted:
            return
        self._vectors = dict(self._vectors)
        self._metadata = dict(self._metadata)
        self._documents = dict(self._documents)
        self._deleted = set(self._deleted)
        for i in accepted:
            doc_id = ids[i]
            self._vectors[doc_id] = np.array(vectors[i], dtype=np.float32, copy=True)
            self._metadata[doc_id] = metadata[i] if metadata else {}
            self._documents[doc_id] = documents[i] if documents else None
            # Re-add de un id soft-deleted: revive (audit r1-04)
            self._deleted.discard(doc_id)
        self._meta_version += 1  # [P13] metadata / posiciones / soft-deletes cambiaron
        if self.storage:
            # Cada lote a su propio shard (shard_001, shard_002, ...); la
            # metadata referencia el shard real vía _last_shard_idx. Para un id
            # re-agregado, save_metadata (INSERT OR REPLACE, deleted=0) además
            # deshace el soft-delete persistido.
            self.storage.save_vectors(vectors[accepted], [ids[i] for i in accepted])
            shard_idx = getattr(self.storage, "_last_shard_idx", 1)
            for row, i in enumerate(accepted):
                self.storage.save_metadata(
                    ids[i],
                    shard_idx=shard_idx,
                    vector_idx=row,
                    metadata=(metadata[i] if metadata else {}),
                    document=(documents[i] if documents else None),
                )

    def _write_splat_at(self, pos: int, vec: np.ndarray) -> None:
        """Reescribe el splat de la posición ``pos`` con ``vec`` y lo reasigna a
        su cluster sin re-entrenar KMeans. Misma normalización y mutación
        in-place que add_splats (normalize_sphere) + reconstrucción del router
        (los coarse centroids quedan viejos si no). Lo usan update(vector) y el
        re-add de un id que ya tiene splat (audit r1-04/C-04b). Lock tomado.
        """
        store = self.engine.m2m.splats
        store.mu[pos] = normalize_sphere(vec)
        store.alpha[pos] = store.config.init_alpha
        store.kappa[pos] = store.config.init_kappa
        store._m_sq_dirty = True
        store._cuda_dirty = True
        # Reasignacion quirurgica al cluster correcto (O(k·d)): sacar del
        # cluster viejo, asignar al centroid mas cercano del mu nuevo.
        eng = store.engine
        # engine.embeddings es la copia del store con la que rankean
        # HRM2Engine.query (lod 0/1) y query_batch (audit r1-24).
        if getattr(eng, "embeddings", None) is not None and pos < len(eng.embeddings):
            eng.embeddings[pos] = store.mu[pos]
        assert eng.coarse_assignments is not None
        assert eng.coarse_model is not None
        old_cid = int(eng.coarse_assignments[pos])
        cent = np.asarray(eng.coarse_model.centroids_, dtype=np.float32)
        new_cid = int(np.argmin(((cent - store.mu[pos]) ** 2).sum(axis=1)))
        eng.coarse_assignments[pos] = new_cid
        if new_cid != old_cid:
            idx_old = eng.coarse_cluster_indices[old_cid]
            keep = idx_old != pos
            eng.coarse_cluster_indices[old_cid] = idx_old[keep]
            eng.coarse_cluster_embeddings[old_cid] = eng.coarse_cluster_embeddings[old_cid][keep]
            eng._cluster_masks[old_cid][pos] = False
            eng.coarse_cluster_indices[new_cid] = np.append(eng.coarse_cluster_indices[new_cid], pos)
            eng.coarse_cluster_embeddings[new_cid] = np.vstack(
                [eng.coarse_cluster_embeddings[new_cid], store.mu[pos][np.newaxis, :]]
            )
            eng._cluster_masks[new_cid][pos] = True
        else:
            # Sin cambio de cluster: refrescar la fila que ya existe (misma
            # causa raíz que r1-24: copia derivada del vector viejo).
            row = np.flatnonzero(eng.coarse_cluster_indices[old_cid] == pos)
            cemb = eng.coarse_cluster_embeddings.get(old_cid)
            if row.size == 1 and cemb is not None and row[0] < len(cemb):
                cemb[row[0]] = store.mu[pos]

    def _add_lsh(
        self,
        ids: List[Any],
        vectors: np.ndarray,
        metadata: Optional[List[Dict]],
        documents: Optional[List[str]],
    ) -> Optional[int]:
        """Camino LSH de add(). Devuelve n, o None si el módulo LSH no está
        (el llamador sigue por el camino splats). Lock tomado.

        Activación tardía (audit r1-08): si el modo LSH se decide cuando ya hay
        docs indexados por el camino splats, TODOS los almacenados entran al
        índice LSH en orden de inserción, de modo que ``_lsh_id_map`` quede
        alineado con ``_splat_id_order`` (antes los previos quedaban
        inalcanzables y search_batch devolvía ids corridos).
        """
        try:
            from .lsh_index import CrossPolytopeLSH, LSHConfig
        except ImportError as e:
            logger.warning("Could not load LSH module: %s.", e)
            self._lsh_active_mode = False
            self._use_lsh = False
            return None

        if self.lsh is None:
            config = LSHConfig(
                dim=self.engine.config.latent_dim,
                n_tables=15,
                n_bits=18,
                n_probes=50,
                n_candidates=500,
            )
            self.lsh = CrossPolytopeLSH(config)
            base_ids = [d for d in self._splat_id_order if d in self._vectors]
            base_vectors = (
                np.stack([self._vectors[d] for d in base_ids])
                if base_ids
                else np.empty((0, self.latent_dim), dtype=np.float32)
            )
        else:
            # Append-only: keep previously indexed vectors searchable
            base_ids = list(self._lsh_id_map)
            base_vectors = (
                self.lsh.vectors
                if self.lsh.vectors is not None and len(self.lsh.vectors) > 0
                else np.empty((0, self.latent_dim), dtype=np.float32)
            )

        rows = [np.asarray(r, dtype=np.float32) for r in base_vectors]
        id_map = list(base_ids)
        for i, doc_id in enumerate(ids):
            if doc_id in id_map:
                rows[id_map.index(doc_id)] = vectors[i]  # re-add: se reescribe
            else:
                rows.append(vectors[i])
                id_map.append(doc_id)
        all_vectors = (
            np.ascontiguousarray(np.stack(rows), dtype=np.float32)
            if rows
            else np.empty((0, self.latent_dim), dtype=np.float32)
        )
        self.lsh.index(all_vectors)
        self._lsh_id_map = id_map
        self._splat_id_order = list(id_map)
        self._use_lsh = True
        self._publish_added(ids, list(range(len(ids))), vectors, metadata, documents)
        return len(ids)

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
                 Un id unhashable o repetido dentro del lote → ValueError antes de
                 mutar nada (audit r1-23). Un id que ya existe se reescribe en su
                 posición (sin splat huérfano); si estaba soft-deleted, revive
                 (audit r1-04).
            vectors: Array numpy [N, D]
            metadata: Lista de dicts de metadata por documento
            documents: Lista de textos originales

        Returns:
            Número de vectores añadidos (nuevos + reescritos/revividos).
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

        # -- H-02 FIX: Dimension validation ----------------------------
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D (got {vectors.ndim}D)")
        if vectors.shape[1] != self.latent_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.latent_dim}, got {vectors.shape[1]}"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("vectors contain NaN or Inf values")
        # Vectores (casi) cero: normalize_sphere(0)=0 queda a distancia 1 de TODA
        # query unitaria y se cuela en el top-k de cualquier búsqueda (audit r1-19).
        norms = np.linalg.norm(vectors, axis=1)
        zero_rows = np.flatnonzero(norms < 1e-6)
        if zero_rows.size:
            raise ValueError(
                f"vectors contain {zero_rows.size} (near-)zero vector(s) with norm < 1e-6 "
                f"at row(s) {zero_rows[:10].tolist()}"
            )

        # Auto-generar IDs si no se proporcionan
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in range(n)]
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()

        # Validar longitudes
        if len(ids) != n:
            raise ValueError(f"len(ids)={len(ids)} != len(vectors)={n}")
        if metadata and len(metadata) != n:
            raise ValueError(f"len(metadata)={len(metadata)} != len(vectors)={n}")
        if documents and len(documents) != n:
            raise ValueError(f"len(documents)={len(documents)} != len(vectors)={n}")

        # IDs válidos ANTES de mutar nada (audit r1-23): un id unhashable
        # dejaba n_active avanzado y una entrada no-string en _splat_id_order.
        from collections.abc import Hashable as _Hashable

        seen = set()
        for i, doc_id in enumerate(ids):
            if not isinstance(doc_id, _Hashable):
                raise ValueError(
                    f"ids[{i}] is not hashable (type {type(doc_id).__name__}): "
                    "document ids must be hashable (e.g. str or int)"
                )
            if doc_id in seen:
                raise ValueError(
                    f"duplicate id {doc_id!r} in the same add() batch (ids must be unique)"
                )
            seen.add(doc_id)

        # Sección crítica ÚNICA (audit r1-03): la decisión LSH, la mutación del
        # store/índice (add_splats = add_splat + build_index), los dicts y el
        # storage corren todos bajo el lock. Antes add_splats corría fuera y dos
        # writers dejaban el índice a medias (IndexError/AssertionError en
        # build_index, n_active != len(_splat_id_order)). Los lectores entran al
        # índice por engine.search, que toma este mismo lock (ver __init__).
        with self._rw_lock:
            # Comprobar LSH fallback (decisión única: primer add con n>=3)
            if self.enable_lsh_fallback and n >= 3 and not self._lsh_decided:
                self._lsh_decided = True
                silhouette = self._compute_silhouette(vectors)
                if silhouette < self.lsh_threshold:
                    logger.info("Distribución uniforme (silhouette=%.4f)", silhouette)
                    logger.info("Activando LSH fallback...")
                    self._lsh_active_mode = True
                else:
                    logger.info(
                        "Distribución clustered (silhouette=%.4f): splats+HRM2", silhouette
                    )

            if self._lsh_active_mode and self.enable_lsh_fallback:
                n_lsh = self._add_lsh(ids, vectors, metadata, documents)
                if n_lsh is not None:
                    return n_lsh

            self._use_lsh = False
            # Ids que ya tienen splat (activos → se reescriben; en _deleted →
            # reviven en la misma posición) vs ids nuevos.
            has_splat = [doc_id in self._vectors for doc_id in ids]
            new_rows = [i for i, h in enumerate(has_splat) if not h]
            if new_rows:
                # Honestidad de capacidad (audit r1-26): lo que no entra NO es
                # buscable: no se guarda en _vectors ni en storage y se avisa.
                n_added = self.engine.add_splats(vectors[new_rows])
                if n_added < len(new_rows):
                    rejected = [str(ids[i]) for i in new_rows[n_added:]]
                    logger.warning(
                        "max_splats=%d alcanzado: %d de %d vectores rechazados (no indexados ni "
                        "persistidos). Ids descartados: %s%s",
                        self.max_splats,
                        len(new_rows) - n_added,
                        n,
                        rejected[:20],
                        " ..." if len(rejected) > 20 else "",
                    )
                    new_rows = new_rows[:n_added]
                # Track which doc_ids correspond to splat positions
                self._splat_id_order.extend(ids[i] for i in new_rows)
            accepted = set(new_rows)
            # Posiciones por dict armado UNA vez por add() (list.index es O(n) por id:
            # O(n*m) al re-insertar muchos ids existentes).
            pos_of = None
            for i, h in enumerate(has_splat):
                if h:
                    if pos_of is None:
                        pos_of = {d: p for p, d in enumerate(self._splat_id_order)}
                    self._write_splat_at(pos_of[ids[i]], vectors[i])
                    accepted.add(i)
            accepted_idx = sorted(accepted)
            self._lsh_id_map = []  # LSH no longer active after normal add
            self._publish_added(ids, accepted_idx, vectors, metadata, documents)

            # Actualizar EBM
            self._update_ebm_splats()

        return len(accepted_idx)

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
            upsert: Si True, crea el documento si no existe (o lo revive si
                estaba soft-deleted; audit r1-04)

        Returns:
            UpdateResult con éxito y delta de energía.
        """
        if id not in self._vectors and id not in self._deleted:
            if upsert:
                if vector is None:
                    raise ValueError(
                        "update(upsert=True) on a new document requires a non-zero 'vector' "
                        "(without it a zero vector — a hub in every search — was inserted)"
                    )
                vec = vector
                self.add(
                    ids=[id],
                    vectors=vec[np.newaxis, :],
                    metadata=[metadata or {}],
                    documents=[document],
                )
                return UpdateResult(success=True, message="upserted")
            return UpdateResult(success=False, message=f"Document {id!r} not found")

        if id in self._deleted:
            if upsert:
                if vector is None:
                    raise ValueError(
                        "update(upsert=True) on a deleted document requires a non-zero 'vector' "
                        "(without it a zero vector — a hub in every search — was inserted)"
                    )
                vec = vector
                # add() valida el vector: si lo rechaza, el doc sigue borrado
                # (estado intacto). add() revive el id en su posición original
                # (audit r1-04); el discard de abajo queda por si acaso.
                self.add(
                    ids=[id],
                    vectors=vec[np.newaxis, :],
                    metadata=[metadata or {}],
                    documents=[document],
                )
                self._deleted.discard(id)
                return UpdateResult(success=True, message="restored and upserted")
            return UpdateResult(success=False, message=f"Document {id!r} is deleted")

        energy_delta = 0.0

        with self._rw_lock:
            if vector is not None:
                old_vec = self._vectors.get(id)
                # Copia propia: mutar el array del llamador no debe desincronizar _vectors (audit r1-20)
                vec = np.array(vector, dtype=np.float32, copy=True)
                if vec.ndim != 1 or vec.shape[0] != self.latent_dim:
                    raise ValueError(
                        f"vector shape mismatch: expected ({self.latent_dim},), got {vec.shape}"
                    )
                # Mismo criterio que add(): NaN/Inf nunca entran al splat store
                # (audit r1-02: envenenaban mu y rompían el próximo retrain completo).
                if not np.all(np.isfinite(vec)):
                    raise ValueError("vector contains NaN or Inf values")
                norm = float(np.linalg.norm(vec))
                if norm < 1e-6:
                    raise ValueError(
                        f"vector has (near-)zero norm ({norm:.3g}) < 1e-6: a zero vector is a hub "
                        "at distance 1 from every unit query"
                    )
                self._vectors[id] = vec

                # Propagar al indice: el splat de este doc vive en la posicion
                # que le toca en _splat_id_order. Sin esto, el indice seguia
                # devolviendo el vector VIEJO para siempre (audit C-04b).
                if self._use_lsh and self.lsh is not None:
                    try:
                        pos = self._lsh_id_map.index(id)
                        # Las filas de lsh.vectors están normalizadas (index()):
                        # guardar el vector crudo dejaba la fila fuera de escala y
                        # con buckets LSH viejos -> se reindexa (audit r1-08).
                        self.lsh.vectors[pos] = normalize_sphere(vec)
                        self.lsh.index(self.lsh.vectors)
                    except ValueError:
                        logger.warning("update(vector): %s no esta en el indice LSH", id)
                elif not self._use_lsh and id in self._splat_id_order:
                    store = self.engine.m2m.splats
                    pos = self._splat_id_order.index(id)
                    if 0 <= pos < store.n_active:
                        self._write_splat_at(pos, vec)

                # Calcular delta de energía si EBM habilitado
                if self.ebm_enabled and self._ebm_energy is not None and old_vec is not None:
                    old_e = self._ebm_energy.energy(old_vec)
                    new_e = self._ebm_energy.energy(vec)
                    energy_delta = new_e - old_e

            if metadata is not None:
                self._metadata[id] = metadata
                self._meta_version += 1  # [P13] invalida el cache de máscaras de filtro

            if document is not None:
                self._documents[id] = document

        # Storage I/O outside lock
        if self.storage:
            if vector is not None:
                # El vector nuevo también tiene que llegar a disco (audit r1-09)
                self.storage.update_vector(id, vec)
            if metadata is not None:
                self.storage.update_metadata(id, metadata=metadata)
            if document is not None:
                self.storage.update_metadata(id, document=document)

        if vector is not None:
            # El splat volvió a α/κ por defecto: el estado persistido ya no le aplica [P15]
            self._refresh_splat_state_if_learned()

        return UpdateResult(success=True, energy_delta=energy_delta)

    def _remove_splats(self, doc_ids: set) -> set:
        """Saca del índice los splats de ``doc_ids`` (audit r1-06). Lock tomado.

        alpha=0 + SplatStore.compact() (misma definición de "muerto" que usa
        compact) y reconstrucción completa del router HRM2 (build_index con
        retrain: tras compactar, las posiciones cambiaron y el attach
        incremental asumiría el n viejo). ``_splat_id_order`` se reconstruye con
        la misma máscara, así sigue alineado con n_active y libera capacidad de
        max_splats. Devuelve los ids OTROS que compact() arrastró (alpha < 0.01
        o mu no finito) para que el llamador les quite sus dicts.
        """
        store = self.engine.m2m.splats
        order = self._splat_id_order
        n = store.n_active
        positions = []
        for doc_id in doc_ids:
            try:
                positions.append(order.index(doc_id))
            except ValueError:
                continue
        if not positions:
            return set()
        for p in positions:
            store.alpha[p] = 0.0
        if len(order) != n:
            # Estado ya inconsistente (p. ej. restaurado sin índice): no se
            # puede compactar sin saber qué posición es qué. Se marca muerto y
            # se documenta; el próximo add() con build_index lo normaliza.
            logger.warning(
                "hard delete: _splat_id_order (%d) != n_active (%d); los splats se marcan "
                "muertos (alpha=0) pero no se compactan ni se libera capacidad",
                len(order),
                n,
            )
            for p in positions:
                order[p] = None
            store._m_sq_dirty = True
            return set()
        keep = (store.alpha[:n] >= 0.01) & np.all(np.isfinite(store.mu[:n]), axis=1)
        extra = {order[i] for i in np.flatnonzero(~keep)} - set(doc_ids)
        if extra:
            logger.warning(
                "hard delete: compact() también retiró %d splat(s) ya muertos: %s",
                len(extra),
                sorted(str(d) for d in extra)[:20],
            )
        removed = store.compact()
        if removed:
            self._splat_id_order = [order[i] for i in np.flatnonzero(keep)]
            store.build_index(retrain=True)
            self._update_ebm_splats()
        return extra

    def _lsh_remove(self, doc_ids: set) -> None:
        """Saca filas del índice LSH (modo LSH, audit r1-06). Lock tomado."""
        if self.lsh is None or self.lsh.vectors is None:
            return
        id_map = list(self._lsh_id_map)
        positions = sorted((id_map.index(d) for d in doc_ids if d in id_map), reverse=True)
        if not positions:
            return
        for p in positions:
            id_map.pop(p)
        self._lsh_id_map = id_map
        self._splat_id_order = list(id_map)
        rows = np.delete(self.lsh.vectors, positions, axis=0)
        if len(rows):
            self.lsh.index(rows)
        else:
            self.lsh.vectors = np.empty((0, self.latent_dim), dtype=np.float32)

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
            id: ID único a eliminar (se acepta 0 y ""; antes eran no-op, audit r1-21)
            ids: Lista o np.ndarray de IDs a eliminar
            filter: Filtro de metadata para eliminar por condición
            hard: Si True, eliminación permanente: además de los dicts, el splat
                sale del índice (compact + reconstrucción HRM2, O(n) por llamada;
                en modo LSH se reindexa sin esas filas) y se libera capacidad de
                max_splats (audit r1-06).

        Returns:
            DeleteResult: ``deleted`` cuenta sólo ids que pasan de activo a
            borrado (soft) o que se borran de verdad (hard); borrar dos veces
            cuenta 1, 0 (audit r1-21).
        """
        # Resolver targets (isinstance: id=0 e ids=np.ndarray son válidos)
        to_delete = set()
        if id is not None:
            to_delete.add(id)
        if ids is not None:
            to_delete.update(ids.tolist() if isinstance(ids, np.ndarray) else ids)

        if not to_delete and not filter:
            return DeleteResult(deleted=0, energy_freed=0.0)

        energy_freed = 0.0
        deleted_count = 0
        storage_ops = []
        hard_ids: set = set()
        soft_new: set = set()

        with self._rw_lock:
            if filter and self.storage:
                to_delete.update(self.storage.filter_by_metadata(filter))
            elif filter:
                # Filtrar en memoria
                for doc_id, meta in self._metadata.items():
                    if doc_id not in self._deleted and self._match_filter(meta, filter):
                        to_delete.add(doc_id)

            for doc_id in to_delete:
                if doc_id not in self._vectors:
                    continue
                if hard:
                    hard_ids.add(doc_id)
                elif doc_id in self._deleted:
                    continue  # ya borrado: no cuenta ni se re-persiste (audit r1-21)
                else:
                    soft_new.add(doc_id)
                # Calcular energía liberada si EBM habilitado
                if self.ebm_enabled and self._ebm_energy is not None:
                    vec = self._vectors.get(doc_id)
                    if vec is not None:
                        energy_freed += float(self._ebm_energy.energy(vec))
                storage_ops.append(("hard" if hard else "soft", doc_id))
                deleted_count += 1

            if soft_new:
                self._deleted = self._deleted | soft_new
            if hard_ids:
                # El splat sale del índice; compact() puede arrastrar otros ya muertos.
                if self._use_lsh and self.lsh is not None:
                    gone = hard_ids
                    self._lsh_remove(hard_ids)
                else:
                    gone = hard_ids | self._remove_splats(hard_ids)
                # Publicación copy-on-write (lectores iterando los dicts viejos)
                self._vectors = {k: v for k, v in self._vectors.items() if k not in gone}
                self._metadata = {k: v for k, v in self._metadata.items() if k not in gone}
                self._documents = {k: v for k, v in self._documents.items() if k not in gone}
                self._deleted = self._deleted - gone
            if soft_new or hard_ids:
                self._meta_version += 1  # [P13] soft-deletes / posiciones cambiaron

            # Storage I/O (dentro del lock: serializado con add/update)
            if self.storage:
                for op, doc_id in storage_ops:
                    if op == "hard":
                        self.storage.hard_delete(doc_id)
                    else:
                        self.storage.soft_delete(doc_id)

        if hard and deleted_count:
            # El splat salió del store: su entrada del npz no debe resucitar [P15]
            self._refresh_splat_state_if_learned()

        return DeleteResult(deleted=deleted_count, energy_freed=energy_freed)

    def _match_filter(self, meta: dict, filter_dict) -> bool:
        """Evalua filtro de metadata contra un doc.

        Acepta un dict ``{campo: valor | {op: valor}}`` con ops ``$eq, $ne, $gt,
        $gte, $lt, $lte, $in, $nin`` o un callable ``metadata -> bool``. Operador
        desconocido (typos como ``$gtt``), condicion ``{}`` o tipo invalido ->
        ValueError: fail closed en vez de dejar pasar todo [audit r1-14]. Misma
        semantica con y sin storage (helper compartido ``match_metadata_filter``).
        """
        if self.storage:
            return self.storage._matches_filter(meta, filter_dict)
        return match_metadata_filter(meta, filter_dict)

    def _position_alive_mask(self, order: List[Any], n: int) -> Optional[np.ndarray]:
        """[P13] `_batch_alive_mask(n, id_map=order)` cacheado por (_meta_version, n).
        None = todas las posiciones están vivas (caso común). Lock tomado."""
        ver = (self._meta_version, n)
        if self._alive_ver != ver:
            self._alive_mask = self._batch_alive_mask(n, id_map=order)
            self._alive_ver = ver
        return self._alive_mask

    def _meta_column(self, key, order: List[Any], n: int, alive) -> tuple:
        """[P13] Columna de metadata `key` sobre las posiciones vivas: (vals, idx,
        simple). vals[j] es el valor de la clave en la posición idx[j] (sólo docs
        activos que TIENEN la clave: un doc sin la clave no cumple ninguna
        condición sobre ella); simple = todos los valores son escalares Python
        exactos. Cacheada por (_meta_version, n). Lock tomado."""
        ver = (self._meta_version, n)
        if self._meta_cols_ver != ver:
            self._meta_cols = {}
            self._meta_cols_ver = ver
        col = self._meta_cols.get(key)
        if col is None:
            if len(self._meta_cols) >= _META_COL_CACHE_MAX:
                self._meta_cols.clear()
            metadata = self._metadata
            vals, idx = [], []
            for i in range(min(n, len(order))):
                if alive is not None and not alive[i]:
                    continue
                m = metadata.get(order[i])
                if m is not None and key in m:
                    vals.append(m[key])
                    idx.append(i)
            simple = all(type(v) in _FILTER_SCALARS for v in vals)
            col = (np.array(vals, dtype=object), np.array(idx, dtype=np.int64), simple)
            self._meta_cols[key] = col
        return col

    def _vectorized_filter_mask(
        self, filter_dict, order: List[Any], n: int, alive
    ) -> Optional[np.ndarray]:
        """[P13] Máscara [n] (posición viva Y cumple el filtro) evaluada por columnas
        con operadores vectorizados, con la MISMA semántica que
        match_metadata_filter ($eq/$ne/$gt/$gte/$lt/$lte sobre escalares, $in/$nin
        sobre listas/tuplas/sets de escalares; condiciones por clave = AND). None si
        el filtro no es vectorizable (ndarray de comparación, valores no escalares,
        operador fuera de la tabla): el llamador usa el camino Python. Ante cualquier
        excepción también devuelve None: el camino Python decide si es un error real
        (p. ej. comparar str con int -> ValueError fail-closed) o si el doc que la
        provocaba nunca se evaluaba (condición anterior en AND, posición borrada)."""
        if not isinstance(filter_dict, dict):
            return None
        mask = np.ones(n, dtype=bool) if alive is None else alive.copy()
        try:
            for key, cond in filter_dict.items():
                vals, idx, simple = self._meta_column(key, order, n, alive)
                if not simple:
                    return None
                res = np.ones(len(vals), dtype=bool)
                if isinstance(cond, dict):
                    for op, comp in cond.items():
                        if op in _FILTER_IN_OPS:
                            if type(comp) not in (list, tuple, set, frozenset):
                                return None
                            if not all(type(v) in _FILTER_SCALARS for v in comp):
                                return None
                            cset = frozenset(comp)
                            if op == "$in":
                                res &= np.fromiter(
                                    (v in cset for v in vals), dtype=bool, count=len(vals)
                                )
                            else:
                                res &= np.fromiter(
                                    (v not in cset for v in vals), dtype=bool, count=len(vals)
                                )
                        else:
                            if type(comp) not in _FILTER_SCALARS:
                                return None
                            res &= np.asarray(_FILTER_VEC_OPS[op](vals, comp), dtype=bool)
                else:
                    if type(cond) not in _FILTER_SCALARS:
                        return None
                    res &= np.asarray(vals == cond, dtype=bool)
                col_mask = np.zeros(n, dtype=bool)
                col_mask[idx] = res
                mask &= col_mask
        except Exception:
            return None
        return mask

    def _filter_position_mask(self, filter, order: List[Any], n: int) -> np.ndarray:
        """[P13] Máscara [n] de posiciones admitidas por un filtro: la posición mapea
        a un doc ACTIVO (no soft ni hard-deleted) Y su metadata cumple el filtro.
        Reemplaza el bucle O(n) en Python que search() hacía por query.

        dict: máscara cacheada (LRU <= 32) por (forma canónica, _meta_version, n);
        la primera evaluación es vectorizada por columnas (_vectorized_filter_mask)
        y, si el filtro no es vectorizable, el camino Python de siempre. callable:
        nunca se cachea (puede depender de estado externo) -> camino Python.
        Lock tomado (search / search_batch_filtered). El array devuelto puede ser
        el del cache: es de sólo lectura, no lo muten.
        """
        validate_metadata_filter(filter)
        vectors, deleted, metadata, match = (
            self._vectors,
            self._deleted,
            self._metadata,
            self._match_filter,
        )

        def python_mask() -> np.ndarray:
            out = np.zeros(n, dtype=bool)
            for i in range(min(n, len(order))):
                doc_id = order[i]
                if doc_id in vectors and doc_id not in deleted and match(metadata.get(doc_id, {}), filter):
                    out[i] = True
            return out

        if callable(filter):
            return python_mask()
        key = _canonical_filter(filter)
        if key is None:
            return python_mask()
        cache_key = (key, self._meta_version, n)
        cache = self._filter_mask_cache
        hit = cache.get(cache_key)
        if hit is not None:
            cache.move_to_end(cache_key)
            return hit
        alive = self._position_alive_mask(order, n)
        allowed = self._vectorized_filter_mask(filter, order, n, alive)
        if allowed is None:
            allowed = python_mask()
        allowed.flags.writeable = False
        cache[cache_key] = allowed
        while len(cache) > _FILTER_MASK_CACHE_MAX:
            cache.popitem(last=False)
        return allowed

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter: Optional[Union[Dict, Callable]] = None,
        include_energy: bool = False,
        include_metadata: bool = False,  # Changed default to False for legacy compat
    ) -> Any:
        """
        Búsqueda de vecinos más cercanos.

        Args:
            query: Vector de consulta [D]
            k: Número de resultados (int o entero de numpy; cualquier otro tipo -> TypeError)
            filter: Filtro de metadata: dict ``{campo: valor | {op: valor}}`` (ops
                ``$eq,$ne,$gt,$gte,$lt,$lte,$in,$nin``) o callable ``metadata -> bool``.
                Operador desconocido, condición ``{}`` o tipo inválido -> ValueError.
                Se re-chequea antes de armar cada resultado: nunca se devuelven docs
                que lo violen.
            include_energy: Si True, incluye información energética (requiere ebm_enabled)
            include_metadata: Si True, retorna List[DocResult]. False retorna tupla legacy.

        Returns:
            Si include_energy=True, include_metadata=True o filter is not None:
                ``List[DocResult]`` con a lo sumo ``k`` docs activos que cumplen el
                filtro (sin ids borrados ni repetidos). Con energía el filtro se aplica
                igual (mismos ids que sin energía). ``score`` es la similitud coseno
                en [-1, 1] entre la query normalizada y el vector almacenado, IGUAL
                en todos los caminos (ver ``DocResult``) [audit r1-13]; el peso del
                splat queda en ``alpha`` y la distancia L2 en ``distance``.
            Si no: 4-tupla legacy ``(mu, alpha, kappa, idx)`` SIEMPRE (también con la
                DB vacía y en modo LSH): ``mu`` [1, m, D] (o [m, D] en LSH), ``alpha`` y
                ``kappa`` [1, m] (o [m]) e ``idx`` [1, m] (o [m]) con ``m <= k``.
                ``idx`` son posiciones en ``_splat_id_order`` de docs activos: nunca
                incluyen un doc borrado (soft o hard). DB vacía -> cuatro arrays vacíos.

        Backfill (audit r1-05): al índice se le piden ``k + #posiciones inválidas``
        (docs soft/hard-deleted o que no pasan el filtro) y se descartan al armar el
        resultado, así ``search(q, k)`` devuelve ``k`` resultados siempre que existan
        ``>= k`` docs activos que cumplan el filtro. El chequeo de colección vacía es
        O(1): no se arma una lista de activos por query (audit r1-16).
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim > 1:
            query = query.squeeze()

        # SPEC-20: Input validation for search
        if query.ndim != 1:
            raise ValueError(f"query must be 1D (got {query.ndim}D)")
        if query.shape[0] != self.latent_dim:
            raise ValueError(
                f"query dimension mismatch: expected {self.latent_dim}, got {query.shape[0]}"
            )
        if not np.all(np.isfinite(query)):
            raise ValueError("query contains NaN or Inf values")
        # r1-27: k entero (int o numpy integer); un float da TypeError claro en vez
        # del "'float' object cannot be interpreted as an integer" del interior del indice
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"k must be an integer (got {type(k).__name__}: {k!r})")
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1 (got {k})")
        # r1-14: validar el filtro ANTES de tocar el indice (fail closed, tambien con DB vacia)
        validate_metadata_filter(filter)

        # Nota del pack P6 (r1-05/r1-18): TODO el cuerpo bajo el lock de escritores.
        # engine.search lo toma como frontera, pero entre su retorno y el ensamblado del
        # DocResult / tupla legacy un delete(hard=True) puede compactar y reordenar
        # _splat_id_order y las posiciones ya no mapearian a los mismos ids. Es un RLock
        # (reentrante): engine.search lo vuelve a tomar sin deadlock.
        with self._rw_lock:

            # Retorno legacy (sin metadata, sin energia, sin filtro)
            legacy = not include_energy and not include_metadata and filter is None
            order = self._splat_id_order
            use_lsh = self._use_lsh and self.lsh is not None

            # Coleccion vacia en O(1) (r1-16; antes una lista O(n) de ids activos por query).
            # _deleted es subconjunto de _vectors: invariante de add/delete (el soft-delete
            # solo marca ids presentes en _vectors; el hard-delete los quita de ambos).
            n_active_docs = len(self._vectors) - len(self._deleted)
            if n_active_docs <= 0:
                if legacy:
                    return (
                        np.array([], dtype=np.float32),
                        np.array([], dtype=np.float32),
                        np.array([], dtype=np.float32),
                        np.array([], dtype=np.int64),
                    )
                return []

            def position_ok(pos: int) -> bool:
                """La posicion de splat mapea a un doc activo (no soft ni hard-deleted)."""
                if pos < 0 or pos >= len(order):
                    return False
                doc_id = order[pos]
                return doc_id in self._vectors and doc_id not in self._deleted

            # r1-05: en modo LSH la posicion LSH == posicion en _splat_id_order (ambas se
            # extienden en el mismo orden en add()).
            n_splats = int(self.lsh.n_vectors) if use_lsh else int(self.engine.m2m.splats.n_active)

            # Mascara por posicion (solo con filtro: doc activo Y cumple el filtro, dict o
            # callable). Posiciones invalidas = borradas o filtradas: se piden de sobra.
            allowed = None
            if filter is None:
                n_bad = max(0, n_splats - n_active_docs)
            else:
                # [P13] Mascara cacheada por filtro (clave = forma canonica +
                # _meta_version + n_splats; LRU <= 32) y vectorizada por columnas en
                # la primera evaluacion: antes un bucle O(n) EN PYTHON por query.
                allowed = self._filter_position_mask(filter, order, n_splats)
                n_bad = n_splats - int(allowed.sum())
            k_eff = min(n_splats, k + n_bad) if n_splats > 0 else 0

            if use_lsh:
                indices, distances = self._lsh_query_deterministic(query, k=k_eff)
                indices = np.asarray(indices, dtype=np.int64).reshape(-1)
                distances = np.asarray(distances, dtype=np.float32).reshape(-1)
                if allowed is None:
                    keep = np.array([position_ok(int(i)) for i in indices], dtype=bool)
                else:
                    keep = np.array(
                        [int(i) < n_splats and bool(allowed[int(i)]) for i in indices], dtype=bool
                    )
                indices, distances = indices[keep], distances[keep]
                if legacy:
                    indices = indices[:k]
                    return (
                        self.lsh.vectors[indices],
                        np.ones(len(indices), dtype=np.float32),
                        np.full(len(indices), 10.0, dtype=np.float32),
                        indices,
                    )
                # El filtro ya se aplico sobre las posiciones: el helper solo saltea borrados
                # (y con include_metadata=False evaluaria un meta={} contra el filtro).
                return self._doc_results_from_lsh(
                    indices,
                    distances,
                    k=k,
                    include_metadata=include_metadata,
                    include_energy=include_energy,
                    query=query,
                )[:k]

            # [P13] En el camino con filtro `raw` no se usa (el ranking sale del kernel
            # con `allowed`): pedirle al indice k + n_bad (= casi todos los docs
            # filtrados) era trabajo muerto (4.2 ms/query a 100k docs).
            raw = None if filter is not None else self.engine.search(query, k_eff)

            if legacy:
                if n_bad == 0:
                    return raw  # Compatibilidad legacy: sin posiciones invalidas, tal cual el indice
                mu, alpha, kappa, idx = raw
                rows = np.asarray(idx).reshape(-1)
                sel = np.zeros(len(rows), dtype=bool)
                for j, pos in enumerate(rows):
                    sel[j] = position_ok(int(pos))
                sel[np.flatnonzero(sel)[k:]] = False  # a lo sumo k posiciones validas
                return mu[:, sel], alpha[:, sel], kappa[:, sel], idx[:, sel]

            store = self.engine.m2m.splats
            # [audit r1-13] MISMA query para el ranking, el score, la distancia y la
            # energia en todos los caminos: la normalizada del camino principal
            # (M2MMemory.retrieve normaliza antes de buscar). Antes el indice rankeaba
            # con la query cruda y el score usaba otra (o el alpha del splat).
            qn = normalize_sphere(query)

            if filter is not None:
                # H8: pre-filtro ACORN-lite (con o sin energia, r1-10): el filtro corre sobre
                # la union de candidatos ANTES del top-k. `allowed` ya excluye posiciones de
                # docs borrados, asi que aca no hace falta backfill.
                results: List[DocResult] = []
                if allowed is not None and allowed.any():
                    # `allowed` ya esta calculada (y excluye borrados): se pasa directo al
                    # kernel en vez de re-entrar por search_batch_filtered, que rehace el
                    # bucle O(n) del filtro por llamada.
                    idxs = self._batch_topk_per_query(
                        qn[np.newaxis, :], k=k, allowed=allowed, expand=True
                    )[0][0]
                    seen = set()
                    for s_idx in idxs:
                        s_idx = int(s_idx)
                        if not position_ok(s_idx):
                            continue
                        doc_id = order[s_idx]
                        # Defensa en profundidad: la posicion tiene que estar en `allowed`
                        # (armada en ESTA llamada bajo el mismo lock: doc activo Y cumple
                        # el filtro). [P13] antes se re-evaluaba el filtro contra la
                        # metadata por resultado: k evaluaciones Python por query.
                        if s_idx >= len(allowed) or not allowed[s_idx] or doc_id in seen:
                            continue
                        seen.add(doc_id)
                        mu_i = store.mu[s_idx]
                        dist = float(np.sqrt(np.maximum(((mu_i - qn) ** 2).sum(), 0.0)))
                        energy_val = None
                        confidence_val = None
                        if include_energy and self.ebm_enabled and self._ebm_energy:
                            # [audit r3a-02] misma energía por resultado que el camino sin filtro
                            energy_val = _result_energy(
                                qn, mu_i, float(store.alpha[s_idx]), float(store.kappa[s_idx])
                            )
                            confidence_val = confidence_from_energy(energy_val)
                        results.append(
                            DocResult(
                                id=doc_id,
                                # [audit r1-13] antes -dist: el mismo coseno que el resto
                                score=_cosine_to_query(qn, mu_i),
                                vector=(mu_i.copy() if include_metadata else None),
                                metadata=(self._metadata.get(doc_id, {}) if include_metadata else {}),
                                document=(self._documents.get(doc_id) if include_metadata else None),
                                energy=energy_val,
                                confidence=confidence_val,
                                alpha=float(store.alpha[s_idx]),
                                distance=dist,
                            )
                        )
                        if len(results) >= k:
                            break
                return results

            # Path con metadata / energia (sin filtro): loop completo (energia per doc)
            # Construir lista de DocResults usando splat indices
            # raw = (mu, alpha, kappa, splat_indices)
            # splat_indices maps each result to a position in _splat_id_order
            mu, alpha, kappa, splat_indices = raw

            results = []
            # Handle batch dim: mu is (1, k, dim) for single query
            if mu.ndim == 3:
                mu = mu[0]       # (k, dim)
                alpha = alpha[0]  # (k,)
                kappa = kappa[0]  # (k,)
                splat_indices = splat_indices[0]  # (k,)

            seen = set()
            for i in range(min(k_eff, len(mu))):
                splat_idx = int(splat_indices[i]) if i < len(splat_indices) else -1

                # Map splat index -> document ID via _splat_id_order; skip deleted (soft/hard)
                if not position_ok(splat_idx):
                    continue
                doc_id = order[splat_idx]
                if doc_id in seen:
                    continue  # posicion huerfana + re-add del mismo id: una fila por doc
                seen.add(doc_id)

                mu_i = mu[i]
                # [audit r1-13] score = similitud coseno (query normalizada · μᵢ) en todos
                # los caminos; antes acá era alpha[i] = 1.0 constante. α queda en `alpha`.
                score = _cosine_to_query(qn, mu_i)
                dist = _l2_to_query(qn, mu_i)
                alpha_val = float(np.asarray(alpha[i]).flat[0]) if i < len(alpha) else None
                energy_val = None
                confidence_val = None

                if (
                    include_energy
                    and self.ebm_enabled
                    and self._ebm_energy
                    and alpha_val is not None
                ):
                    # [audit r3a-02] energía de la QUERY normalizada bajo el splat de ESTE
                    # resultado. Antes: el paisaje en la posición del propio documento
                    # (10.0 para todos con el vector crudo; ≈ -ln αᵢ con unitarios).
                    energy_val = _result_energy(qn, mu_i, alpha_val, float(kappa[i]))
                    confidence_val = confidence_from_energy(energy_val)

                results.append(
                    DocResult(
                        id=doc_id,
                        score=score,
                        vector=mu_i if include_metadata else None,
                        metadata=(self._metadata.get(doc_id, {}) if include_metadata else {}),
                        document=(self._documents.get(doc_id) if include_metadata else None),
                        energy=energy_val,
                        confidence=confidence_val,
                        alpha=alpha_val,
                        distance=dist,
                    )
                )
                if len(results) >= k:
                    break

            return results

    def _lsh_query_deterministic(self, query: np.ndarray, k: int):
        """lsh.query() con RNG global pineado: su fallback interno de
        candidatos usa np.random.choice, lo que hacia que dos llamadas con
        la MISMA query devolvieran resultados distintos (rompia la parity
        single vs batch). Seed fija + restauracion del estado global."""
        state = np.random.get_state()
        np.random.seed(0)
        try:
            return self.lsh.query(query, k=k)
        finally:
            np.random.set_state(state)

    def _doc_results_from_lsh(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        k: int,
        include_metadata: bool,
        include_energy: bool = False,
        filter: Optional[Dict] = None,
        query: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Ensambla List[DocResult] desde indices posicionales del LSH.

        Logica comun extraida del path LSH de search() single (T2): el mapeo
        _lsh_id_map + skip de borrados + filtro + energia. La usa search()
        single y el camino LSH de search_batch() para garantizar parity exacta.

        `query` (aditivo, requerida) es la query cruda de esta busqueda: `score` es la
        similitud coseno entre su version normalizada y el vector del LSH —la misma
        semantica que en modo splats [audit r1-13]— y `distance` la L2. En modo LSH el
        engine no tiene splats (n_active = 0), asi que NO hay energia por resultado:
        `alpha` es None y `energy` es la del paisaje EBM en la query normalizada —igual
        para todos los resultados de esa query—; con ``enable_ebm=True`` y el EBM sin
        splats es ``OOD_ENERGY`` (10) y ``confidence = sigmoid(-10) ≈ 4.5e-5``.
        """
        if query is None:
            raise ValueError("_doc_results_from_lsh requiere `query` (score = similitud coseno)")
        qn = normalize_sphere(np.asarray(query, dtype=np.float32))
        candidate_vectors = self.lsh.vectors[indices]
        results: List[Any] = []
        for i, idx in enumerate(indices):
            if i >= k:
                break
            if idx < len(self._lsh_id_map):
                doc_id = self._lsh_id_map[idx]
            else:
                continue  # stale index, skip
            if doc_id in self._deleted or doc_id not in self._vectors:
                continue
            # El filtro se evalúa contra la metadata REAL siempre; `meta` es sólo
            # lo que se devuelve (audit P15: con include_metadata=False el filtro se
            # evaluaba contra {} y descartaba todos los resultados).
            full_meta = self._metadata.get(doc_id, {})
            if filter and not self._match_filter(full_meta, filter):
                continue
            meta = full_meta if include_metadata else {}
            mu_i = candidate_vectors[i]
            score = _cosine_to_query(qn, mu_i)  # [audit r1-13] antes: la distancia del LSH
            energy_val = None
            confidence_val = None
            if include_energy and self.ebm_enabled and self._ebm_energy:
                energy_val = self._ebm_energy.energy(qn)
                confidence_val = confidence_from_energy(energy_val)
            results.append(
                DocResult(
                    id=doc_id,
                    score=score,
                    vector=mu_i if include_metadata else None,
                    metadata=meta,
                    document=(self._documents.get(doc_id) if include_metadata else None),
                    energy=energy_val,
                    confidence=confidence_val,
                    alpha=None,
                    distance=_l2_to_query(qn, mu_i),
                )
            )
        return results[:k]

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        include_metadata: bool = False,
    ) -> Any:
        """Búsqueda por lotes: amortiza todo el overhead per-query de Python.

        Estrategia (CPU, modo HRM2):
          1. Routing por cluster (HRM2) para TODO el lote.
          2. Por CADA query, distancias SOLO contra las filas de sus n_probe clusters
             sondeados + top-k (candidatos POR QUERY, misma semántica que search()
             single; nunca la unión del lote) [audit r4-07, r1-12].
          3. Construcción de DocResult (o índices globales).

        Con n<=3000 no hay poda y esto degenera en exact search batched — también
        correcto y rápido. Las posiciones de docs borrados (soft o hard) se excluyen
        del ranking y se rellena desde más candidatos propios mientras haya activos
        [audit r1-05].

        Returns:
            include_metadata=False -> np.ndarray [B, k] de posiciones (-1 si falta):
                índice en _splat_id_order (modo HRM2) o en _lsh_id_map (modo LSH).
            include_metadata=True -> List[List[DocResult]]. ``score`` es la similitud
                coseno contra la query normalizada: la MISMA semántica que ``search()``
                single [audit r1-13]; ``alpha`` y ``distance`` también se informan.
            (B, 0) / listas vacías si la DB no tiene documentos activos.
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
        if not np.all(np.isfinite(queries)):
            raise ValueError("queries contain NaN or Inf values")
        if queries.shape[1] != self.latent_dim:
            raise ValueError(
                f"queries dim mismatch: expected {self.latent_dim}, got {queries.shape[1]}"
            )
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"k must be an integer, got {type(k).__name__} ({k!r})")
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1 (got {k})")

        B = len(queries)
        if B == 0:
            return (
                np.full((B, 0), -1, dtype=np.int64)
                if not include_metadata
                else [[] for _ in range(B)]
            )

        # Lectura bajo el mismo lock que los writers (audit r1-03): entre el
        # enrutado por centroids y el armado de DocResult un writer puede
        # retrainear/compactar y desalinear las posiciones (KMeans sin fitear,
        # coarse_assignments de otra longitud).
        with self._rw_lock:
            # BUG 1 fix (T2): en modo LSH los vectores viven en el indice LSH
            # (n_active=0 en splats), por lo que el path HRM2 de abajo devolvia
            # -1/[] para toda query. Routear al LSH igual que search() single.
            if self._use_lsh and self.lsh is not None:
                out_idx, rows = self._lsh_batch_topk(queries, k=k, include_metadata=include_metadata)
                return rows if include_metadata else out_idx

            store = self.engine.m2m.splats
            n = store.n_active
            alive = self._batch_alive_mask(n)
            if n == 0 or (alive is not None and not alive.any()):
                return (
                    np.full((B, 0), -1, dtype=np.int64)
                    if not include_metadata
                    else [[] for _ in range(B)]
                )

            out_idx, out_d2 = self._batch_topk_per_query(queries, k, allowed=alive)

            if not include_metadata:
                return out_idx

            results: List[List[DocResult]] = []
            order = self._splat_id_order
            # [audit r1-13] mismo score (coseno contra la query normalizada) que search()
            # single; antes -sqrt(d²) contra la query CRUDA (dependía de su norma).
            qn = normalize_sphere(queries)
            for i in range(B):
                row: List[DocResult] = []
                for j in range(k):
                    s_idx = int(out_idx[i, j])
                    if s_idx < 0 or s_idx >= len(order):
                        continue
                    doc_id = order[s_idx]
                    if doc_id in self._deleted or doc_id not in self._vectors:
                        continue  # defensa: la máscara viva ya los excluyó del ranking
                    mu_i = store.mu[s_idx]
                    row.append(
                        DocResult(
                            id=doc_id,
                            score=_cosine_to_query(qn[i], mu_i),
                            vector=None,
                            metadata=self._metadata.get(doc_id, {}),
                            document=self._documents.get(doc_id),
                            energy=None,
                            confidence=None,
                            alpha=float(store.alpha[s_idx]),
                            distance=_l2_to_query(qn[i], mu_i),
                        )
                    )
                results.append(row[:k])
            return results

    def search_batch_filtered(
        self,
        queries: np.ndarray,
        k: int = 10,
        filter_fn=None,
        filter: Optional[Union[Dict, Callable]] = None,
    ) -> np.ndarray:
        """Búsqueda por lotes con PRE-filtro (ACORN-lite, H8).

        El filtro corre ANTES del top-k: sin filtro, el top-k del índice puede no
        contener ni un solo doc que pase el filtro (post-filter media recall 0.26 con
        selectividad 25%; medido en audit). filter_fn: callable(int posicion) -> bool,
        evaluado una vez por posición y aplicado como máscara de candidatos.
        ``filter`` (adicional [P13]): dict ``{campo: valor | {op: valor}}`` o
        callable ``metadata -> bool`` con la misma semántica que
        ``search(filter=...)``; la máscara por posición se cachea por (filtro,
        versión de metadata) y toda escritura (add/update/delete) la invalida.
        ``filter`` y ``filter_fn`` juntos -> ValueError.

        Candidatos POR QUERY [audit r4-07, r1-12]: cada query se puntúa solo contra las
        filas que pasan el filtro dentro de SUS clusters sondeados; si así quedan menos
        de 1.5*k docs permitidos, ESA query expande SUS probes siguiendo SU propio orden
        de clusters hasta cubrir el objetivo o agotar los clusters (antes la decisión se
        tomaba con la cobertura de la UNIÓN del lote y el orden de clusters de la
        query 0). Las filas que no pasan el filtro (ni las de docs borrados) entran al
        ranking, así que no hace falta sentinel [audit r1-22: BIG=1e9 no era "grande"
        frente a ||q||^2 > 1e9 y devolvía docs que violaban el filtro].

        Returns:
            np.ndarray [B, k] de posiciones (-1 si falta); (B, 0) si la DB no tiene
            documentos activos. Mismo espacio de posiciones que search_batch().
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
        if not np.all(np.isfinite(queries)):
            raise ValueError("queries contain NaN or Inf values")
        if queries.shape[1] != self.latent_dim:
            raise ValueError(
                f"queries dim mismatch: expected {self.latent_dim}, got {queries.shape[1]}"
            )
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"k must be an integer, got {type(k).__name__} ({k!r})")
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1 (got {k})")
        if filter is not None and filter_fn is not None:
            raise ValueError("pasa filter (dict) o filter_fn (callable posicion -> bool), no ambos")
        validate_metadata_filter(filter)
        if filter is None and filter_fn is None:
            return self.search_batch(queries, k=k, include_metadata=False)
        if self._use_lsh and self.lsh is not None:
            # Modo LSH: los vectores viven en el indice LSH (n_active=0 en splats); sin
            # esta rama search_batch_filtered devolvia todo -1 [audit r1-07].
            if filter is not None:
                id_map = self._lsh_id_map
                with self._rw_lock:
                    allowed = self._filter_position_mask(filter, id_map, len(id_map))
                filter_fn = lambda i: i < len(allowed) and bool(allowed[i])
            return self._lsh_batch_topk(queries, k=k, filter_fn=filter_fn)[0]

        # Mismo lock que los writers (audit r1-03): ver search_batch().
        with self._rw_lock:
            B = len(queries)
            store = self.engine.m2m.splats
            n = store.n_active
            alive = self._batch_alive_mask(n)
            if n == 0 or (alive is not None and not alive.any()):
                return np.full((B, 0), -1, dtype=np.int64)

            if filter is not None:
                # [P13] dict (o callable): mascara por posicion cacheada; ya excluye
                # las posiciones de docs borrados.
                allowed = self._filter_position_mask(filter, self._splat_id_order, n)
            else:
                allowed = np.fromiter((bool(filter_fn(i)) for i in range(n)), dtype=bool, count=n)
                if alive is not None:
                    allowed &= alive
            if not allowed.any():
                return np.full((B, k), -1, dtype=np.int64)
            return self._batch_topk_per_query(queries, k, allowed=allowed, expand=True)[0]

    def _batch_alive_mask(
        self, n: int, id_map: Optional[List[Any]] = None
    ) -> Optional[np.ndarray]:
        """Máscara [n] de posiciones VIVAS: su id sigue en _vectors y no está
        soft-deleted [audit r1-05, parte batch]. `id_map` es la lista posición -> id
        (_splat_id_order en modo HRM2, _lsh_id_map en modo LSH); posiciones fuera de
        rango cuentan como muertas. Devuelve None cuando todas lo están (caso común:
        nada borrado), para no pagar O(n) lookups por llamada."""
        if id_map is None:
            id_map = self._splat_id_order
        if not self._deleted and len(self._vectors) == n and len(id_map) >= n:
            return None
        alive = np.zeros(n, dtype=bool)
        vectors, deleted = self._vectors, self._deleted
        for pos in range(min(n, len(id_map))):
            doc_id = id_map[pos]
            if doc_id in vectors and doc_id not in deleted:
                alive[pos] = True
        return alive

    def _batch_topk_per_query(
        self,
        queries: np.ndarray,
        k: int,
        allowed: Optional[np.ndarray] = None,
        expand: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k POR QUERY en modo HRM2: candidatos = filas de los clusters sondeados
        por CADA query (los mismos que usa find_neighbors), nunca la unión del lote
        [audit r4-07, r1-12].

        allowed: máscara [n] de posiciones admitidas (filtro / docs vivos); el resto se
        saca del conjunto de candidatos antes de puntuar (sin sentinel).
        expand (ACORN-lite por query, [audit r4-07]): si tras aplicar `allowed` la query
        tiene menos de 1.5*k candidatos, agrega clusters en SU propio orden de cercanía
        hasta cubrir el objetivo o agotar los clusters.

        Devuelve (out_idx [B, k] de posiciones globales con -1 si falta, out_d2 [B, k]
        con la distancia^2 de cada posición devuelta, inf en el padding).
        """
        store = self.engine.m2m.splats
        eng = store.engine
        n = store.n_active
        index_data = store.mu[:n]
        B = len(queries)
        n_probe = int(getattr(eng, "n_probe", 5) or 5)
        m_sq = store._m_sq_cache
        if m_sq is None or len(m_sq) != n:
            m_sq = np.einsum("ij,ij->i", index_data, index_data)
        # ||q||^2 es constante por fila: no altera el orden, se suma solo al top-k.
        q_sq = np.einsum("ij,ij->i", queries, queries)

        use_hrm2 = n > 3000 and eng.coarse_model is not None
        order = None
        cluster_rows = None
        cov = None
        if use_hrm2:
            centroids = np.asarray(eng.coarse_model.centroids_, dtype=np.float32)
            diff_c = queries[:, None, :] - centroids[None, :, :]
            cdist = np.einsum("bkc,bkc->bk", diff_c, diff_c)  # distancia^2: mismo orden
            n_probe = min(n_probe, cdist.shape[1])
            order = np.argsort(cdist, axis=1)  # [B, n_coarse] orden completo (para expandir)
            probe = order[:, :n_probe]
            cluster_rows = eng.coarse_cluster_indices
            if expand:
                # docs admitidos por cluster: permite decidir la expansion por query
                # con un cumsum en vez de re-contar cluster a cluster.
                cov = np.zeros(cdist.shape[1], dtype=np.int64)
                for c, rows_c in cluster_rows.items():
                    cov[c] = int(allowed[rows_c].sum())
        need = max(1, int(k * 1.5))

        out_idx = np.full((B, k), -1, dtype=np.int64)
        out_d2 = np.full((B, k), np.inf, dtype=np.float32)
        for i in range(B):
            cand = None  # None = todas las filas (sin poda)
            if use_hrm2:
                lists = []
                for c in probe[i]:
                    rows_c = cluster_rows.get(int(c))
                    if rows_c is not None and len(rows_c) > 0:
                        lists.append(rows_c)
                if not lists:
                    continue
                if expand:
                    covered = int(cov[probe[i]].sum())
                    if covered < need:
                        rest = order[i, n_probe:]
                        hits = cov[rest]
                        if hits.any():
                            m = int(np.searchsorted(np.cumsum(hits), need - covered, side="left"))
                            for c in rest[: m + 1]:
                                rows_c = cluster_rows.get(int(c))
                                if rows_c is not None and len(rows_c) > 0:
                                    lists.append(rows_c)
                cand = lists[0] if len(lists) == 1 else np.concatenate(lists)
            if allowed is not None:
                cand = np.flatnonzero(allowed) if cand is None else cand[allowed[cand]]
                if len(cand) == 0:
                    continue
            if cand is None:
                d2 = m_sq - 2.0 * (index_data @ queries[i])
                k_eff = min(k, n)
            else:
                d2 = m_sq[cand] - 2.0 * (index_data[cand] @ queries[i])
                k_eff = min(k, len(cand))
            part = np.argpartition(d2, k_eff - 1)[:k_eff]
            part = part[np.argsort(d2[part])]
            out_d2[i, :k_eff] = np.maximum(d2[part] + q_sq[i], 0.0)
            out_idx[i, :k_eff] = part if cand is None else cand[part]
        return out_idx, out_d2

    def _lsh_batch_topk(
        self,
        queries: np.ndarray,
        k: int,
        filter_fn=None,
        include_metadata: bool = False,
    ) -> Tuple[np.ndarray, Any]:
        """Top-k POR QUERY en modo LSH; las posiciones indexan _lsh_id_map [audit r1-07].

        Candidatos = todos los que sondea el LSH para la query (lsh.query_candidates,
        sin truncar a k), para que el filtro y la exclusión de docs borrados puedan
        rellenar hasta k. Sin filtro y sin borrados devuelve exactamente lo mismo que
        search() single (mismo sondeo, mismo RNG pineado).
        Devuelve (out_idx [B, k] con -1 si falta, rows List[List[DocResult]] o None)."""
        id_map = self._lsh_id_map
        B = len(queries)
        out_idx = np.full((B, k), -1, dtype=np.int64)
        rows: List[List[DocResult]] = (
            [[] for _ in range(B)] if include_metadata else None
        )
        alive = self._batch_alive_mask(len(id_map), id_map=id_map)
        if alive is not None and not alive.any():
            return np.full((B, 0), -1, dtype=np.int64), rows  # sin activos: (B, 0)
        ok = alive
        if filter_fn is not None:
            allowed = np.fromiter(
                (bool(filter_fn(i)) for i in range(len(id_map))), dtype=bool, count=len(id_map)
            )
            ok = allowed if ok is None else (allowed & ok)
        for i in range(B):
            state = np.random.get_state()
            np.random.seed(0)  # mismo pineado que _lsh_query_deterministic (fallback RNG)
            try:
                idx, dist = self.lsh.query_candidates(queries[i], k=k)
            finally:
                np.random.set_state(state)
            if ok is not None and len(idx):
                keep = ok[idx]
                idx, dist = idx[keep], dist[keep]
            m = min(k, len(idx))
            out_idx[i, :m] = idx[:m]
            if include_metadata:
                rows[i] = self._doc_results_from_lsh(
                    idx[:m], dist[:m], k=k, include_metadata=True, query=queries[i]
                )
        return out_idx, rows

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
        return self._ebm_exploration.suggest_exploration(current_knowledge=ids, n_suggestions=n)

    def find_knowledge_gaps(self, n: int = 5) -> List[EnergyRegion]:
        """Encuentra huecos en el conocimiento (requiere ebm_enabled=True)."""
        if not self.ebm_enabled or self._ebm_exploration is None:
            raise RuntimeError("EBM features no habilitadas.")
        return self._ebm_exploration.find_knowledge_gaps(n_gaps=n)

    def save(self, path: Optional[str] = None) -> str:
        """Persiste la base en disco.

        - ``save()`` (sin argumento): hace checkpoint (flush + fsync) del WAL de
          ``storage_path``. Si la DB se creó sin ``storage_path`` levanta
          ``RuntimeError``: antes era un no-op silencioso que además llamaba a
          LOAD del engine (audit r1-17).
        - ``save(path)``: exporta un snapshot autocontenido en el directorio
          ``path`` (mismo formato que ``storage_path``: shards + metadata.db +
          soft-deletes), re-abrible con ``SimpleVectorDB(storage_path=path)`` o
          ``load(path)``. Si ``path`` es el ``storage_path`` actual equivale a
          ``save()``. Un ``path`` existente y no vacío levanta ``FileExistsError``.

        Con ``M2M_HMAC_SECRET`` seteado, firma el storage (y el snapshot) con un
        manifiesto HMAC-SHA256 que el reopen/``load()`` verifica (audit r3a-17);
        sin secreto no se firma ni se verifica.

        Args:
            path: Directorio destino del snapshot (opcional).

        Returns:
            Ruta absoluta persistida (``storage_path`` o el snapshot).
        """
        import shutil
        from pathlib import Path

        def _new_snapshot(target: str) -> M2MPersistence:
            dest = Path(target)
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError(
                    f"save(path): {dest} already exists and is not empty (not overwriting)"
                )
            return M2MPersistence(str(dest), enable_wal=False)

        target = str(path).strip() if path is not None else ""
        if self.storage is not None:
            if not target or Path(target).resolve() == self.storage.storage_path:
                # save() ya no re-escribe el shard completo: cada add() persistió
                # su propio shard (append-only), la metadata SQLite mapea cada doc
                # a su (shard_idx, vector_idx). checkpoint() hace fsync del WAL.
                self.storage.checkpoint()
                self.storage.write_manifest()
                return str(self.storage.storage_path)
            self.storage.checkpoint()  # el snapshot copia el WAL ya flusheado
            snap = _new_snapshot(target)
            try:
                shutil.copytree(
                    str(self.storage.storage_path), str(snap.storage_path), dirs_exist_ok=True
                )
                snap.write_manifest()
            finally:
                snap.close()
            return str(snap.storage_path)

        if not target:
            raise RuntimeError(
                "save() without storage_path cannot persist in place: pass a path to export "
                "a snapshot (save(path)) or create the DB with storage_path=..."
            )
        # DB sólo en memoria: escribir un snapshot real (antes: no-op silencioso)
        snap = _new_snapshot(target)
        try:
            ids = [i for i in self._splat_id_order if i in self._vectors]
            seen = set(ids)
            ids += [i for i in self._vectors if i not in seen]
            if ids:
                vectors = np.stack(
                    [np.asarray(self._vectors[i], dtype=np.float32).reshape(-1) for i in ids]
                )
                snap.save_vectors(vectors, ids)
                shard_idx = getattr(snap, "_last_shard_idx", 1)
                for i, doc_id in enumerate(ids):
                    snap.save_metadata(
                        doc_id,
                        shard_idx=shard_idx,
                        vector_idx=i,
                        metadata=self._metadata.get(doc_id) or {},
                        document=self._documents.get(doc_id),
                    )
                for doc_id in ids:
                    if doc_id in self._deleted:
                        snap.soft_delete(doc_id)
            snap.write_manifest()
        finally:
            snap.close()
        return str(snap.storage_path)

    def load(self, path: str) -> "SimpleVectorDB":
        """Carga la base desde el directorio de storage ``path``.

        ``path`` es un directorio escrito por ``storage_path`` o ``save(path)``;
        reemplaza el estado en memoria por el de ``path`` (índice reconstruido
        por ``_restore_from_storage``) y pasa a persistir ahí. Errores explícitos
        (antes ``load()`` devolvía self con un warning ante cualquier falla,
        audit r1-17): ``FileNotFoundError`` si ``path`` no existe,
        ``NotADirectoryError`` si es un archivo, ``RuntimeError`` si la firma
        HMAC del storage no verifica y ``ValueError`` (path traversal).
        """
        from pathlib import Path

        target = str(path).strip() if path is not None else ""
        if not target:
            raise ValueError("load() requires a storage directory path")
        p = Path(target)
        if not p.exists():
            raise FileNotFoundError(f"load(): path does not exist: {p}")
        if not p.is_dir():
            raise NotADirectoryError(
                f"load(): {p} is a file; expected a storage directory written by "
                "storage_path/save(path) (pre-computed .bin datasets go through "
                "engine.load_optimized)"
            )
        old = self.storage
        enable_wal = old.enable_wal if old is not None else True
        if old is not None and old.storage_path == p.resolve():
            old.close()  # mismo directorio: cerrar el WAL antes de reabrirlo
            old = None
        self.storage = M2MPersistence(str(p), enable_wal=enable_wal)
        if old is not None:
            old.close()
        self._restore_from_storage()
        if self.ebm_enabled:
            self._update_ebm_splats()
        return self

    def get_stats(self) -> Dict:
        """Estadísticas del sistema."""
        base = {
            "total_documents": len(self._vectors),
            "active_documents": len([d for d in self._vectors if d not in self._deleted]),
            "deleted_documents": len(self._deleted),
            "lsh_active": self._use_lsh,
            "ebm_enabled": self.ebm_enabled,
            # backend EFECTIVO de búsqueda (cpu/cuda/vulkan) según el selector
            # (m2m.engine), vs. el dispositivo que se pidió [audit P15].
            "backend": self.engine.backend,
            "requested_device": self.engine.get_backend_info().get("requested"),
        }
        if self.storage:
            storage_stats = self.storage.get_stats()
            base.update({"storage": storage_stats})
        if self.ebm_enabled and self._ebm_energy:
            base["free_energy"] = self._ebm_energy.free_energy()
        return base


    # ------------------------------------------------------------------
    # Online learning por ids de documento [audit r3a-08]
    # ------------------------------------------------------------------

    def feedback(
        self,
        query: np.ndarray,
        relevant_ids: Optional[List[Any]] = None,
        irrelevant_ids: Optional[List[Any]] = None,
    ) -> "FeedbackResult":
        """
        Aprende de feedback del usuario tras una búsqueda (reglas Hebbianas).

        ``relevant_ids`` / ``irrelevant_ids`` son IDS DE DOCUMENTO (los que
        devuelve ``DocResult.id``), NO posiciones de splat: ``DocResult`` no
        expone la posición a propósito porque es inestable (el hard-delete y
        ``consolidate()`` compactan el store y reordenan ``_splat_id_order``).
        Las posiciones se resuelven en el momento de la llamada, bajo
        ``_rw_lock`` (el mismo de search/writers).

        Reglas (``OnlineUpdater``): relevante → α += η_α, κ += η_κ y μ deriva
        hacia la query re-proyectado a la esfera; irrelevante → α decae y κ baja.
        α y κ quedan acotados. Con EBM, el paisaje (y el SOC de
        ``AdvancedVectorDB``) se refresca al terminar.

        Args:
            query: Vector de consulta [D] (finito).
            relevant_ids: ids de documentos confirmados relevantes.
            irrelevant_ids: ids de documentos confirmados NO relevantes.

        Returns:
            ``FeedbackResult`` con cuántos ids se aplicaron y cuáles se
            ignoraron por no existir (o estar borrados).

        Raises:
            ValueError: dimensión incorrecta o query con NaN/Inf.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.latent_dim:
            raise ValueError(
                f"query dimension mismatch: expected {self.latent_dim}, got {q.shape[0]}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("query contains NaN or Inf values")

        with self._rw_lock:
            # ids -> posiciones AHORA (nunca se guardan entre llamadas)
            pos_of: Dict[Any, int] = {}
            for pos, doc_id in enumerate(self._splat_id_order):
                if (
                    doc_id is not None
                    and doc_id not in pos_of
                    and doc_id in self._vectors
                    and doc_id not in self._deleted
                ):
                    pos_of[doc_id] = pos

            def _resolve(doc_ids) -> Tuple[List[int], List[Any]]:
                positions: List[int] = []
                unknown: List[Any] = []
                for doc_id in doc_ids or []:
                    pos = pos_of.get(doc_id)
                    if pos is None:
                        unknown.append(doc_id)
                    elif pos not in positions:
                        positions.append(pos)
                return positions, unknown

            rel, unk_rel = _resolve(relevant_ids)
            irr, unk_irr = _resolve(irrelevant_ids)
            unknown = sorted({str(d) for d in unk_rel + unk_irr})
            summary: Dict[str, int] = {}
            if rel or irr:
                summary = self.engine.m2m.splats.feedback(q, rel, irr)
                # el drift Hebbiano movió mu[:n] in-place; el índice HRM2 guarda
                # una copia -> se sincronizan esas filas (n > 3000 rankea por la copia)
                self._sync_index_mu(rel)
                self._update_ebm_splats()
                self._write_splat_state()

        return FeedbackResult(
            n_relevant=len(rel),
            n_irrelevant=len(irr),
            updated_positions=rel + irr,
            unknown_ids=unknown,
            summary=summary,
        )

    def _sync_index_mu(self, positions: List[int]) -> None:
        """Refresca en el índice HRM2 las filas de ``positions`` cuyo μ cambió
        in-place (drift Hebbiano). ``engine.embeddings`` y
        ``coarse_cluster_embeddings`` son COPIAS de μ: sin esto, con
        n > 3000 (camino IVF) ``search()`` seguía rankeando con el μ viejo.
        La asignación a clúster no cambia (drift chico): attach in-place,
        O(len(positions)). Requiere el lock tomado. [audit r3a-08]
        """
        store = self.engine.m2m.splats
        eng = store.engine
        n = store.n_active
        emb = getattr(eng, "embeddings", None)
        if not positions or emb is None or len(emb) != n or not getattr(eng, "_is_indexed", False):
            return
        masks = getattr(eng, "_cluster_masks", None) or {}
        blocks = getattr(eng, "coarse_cluster_embeddings", None) or {}
        for p in positions:
            if not 0 <= p < n:
                continue
            emb[p] = store.mu[p]
            for cid, mask in masks.items():
                if len(mask) == n and mask[p]:
                    rows = np.flatnonzero(mask)
                    j = int(np.searchsorted(rows, p))
                    block = blocks.get(cid)
                    if block is not None and j < len(block) and rows[j] == p:
                        block[j] = store.mu[p]
                    break

    # ------------------------------------------------------------------
    # Estado aprendido de los splats (α, κ, μ por id) [audit r3a-07]
    # ------------------------------------------------------------------
    # El storage persiste vectores + metadata; α/κ/μ aprendidos (feedback,
    # avalanche, relax) vivían sólo en memoria y al reabrir volvían a
    # init_alpha/init_kappa. Se guardan en <storage>/energy/landscape.npz
    # (API existente ``save_energy_state``/``load_energy_state``, sin pickle),
    # junto con el vector normalizado del que nació cada splat (``base``): al
    # restaurar, un doc cuyo vector cambió después (``update(vector=)``, que
    # resetea α/κ) NO recibe el estado viejo.

    def _splat_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Snapshot (α, κ, μ, base) por id en el orden actual de splats, o None si
        ``_splat_id_order`` no está alineado con el store (no se persiste un
        estado que no se pueda volver a aplicar). Lock tomado."""
        order = self._splat_id_order
        store = self.engine.m2m.splats
        n = store.n_active
        if len(order) != n:
            return None
        base = np.zeros((n, self.latent_dim), dtype=np.float32)
        for p, doc_id in enumerate(order):
            v = self._vectors.get(doc_id)
            if v is not None:
                base[p] = normalize_sphere(np.asarray(v, dtype=np.float32).reshape(-1))
        return {
            "version": np.array([1], dtype=np.int64),
            "ids": np.array([str(d) if d is not None else "" for d in order], dtype=np.str_),
            "alpha": np.array(store.alpha[:n], dtype=np.float32),
            "kappa": np.array(store.kappa[:n], dtype=np.float32),
            "mu": np.array(store.mu[:n], dtype=np.float32),
            "base": base,
        }

    def _write_splat_state(self, target_dir: Optional[str] = None) -> None:
        """Persiste el estado aprendido en ``<target_dir|storage>/energy/landscape.npz``.
        ``target_dir`` se usa para el snapshot de una DB sólo en memoria
        (``save(path)``). Fallas de escritura se loguean, no rompen."""
        state = self._splat_state()
        if state is None:
            return
        from pathlib import Path

        try:
            if target_dir is not None:
                edir = Path(target_dir) / "energy"
                edir.mkdir(parents=True, exist_ok=True)
                np.savez(str(edir / "landscape.npz"), **state)
            elif self.storage is not None:
                Path(self.storage.storage_path, "energy").mkdir(parents=True, exist_ok=True)
                self.storage.save_energy_state(state)
        except Exception as e:  # defensa: la persistencia del estado es best-effort
            logger.warning("splat state not persisted: %s: %s", type(e).__name__, e)

    def _has_persisted_splat_state(self) -> bool:
        """True si el storage ya guarda estado aprendido (``energy/landscape.npz``)."""
        if self.storage is None:
            return False
        from pathlib import Path

        return (Path(self.storage.storage_path) / "energy" / "landscape.npz").is_file()

    def _refresh_splat_state_if_learned(self) -> None:
        """Re-escribe ``energy/landscape.npz`` si el storage ya guarda estado
        aprendido (α/κ/μ por id) [audit P15].

        Lo llaman los writers que dejan entradas viejas en ese archivo:
        ``update(vector=)`` (el splat vuelve a α/κ por defecto) y el hard-delete.
        Sin esto, al reabrir, un doc re-agregado o vuelto a su vector anterior
        resucitaba su α/κ/μ aprendidos. Sin estado persistido no hace nada.
        """
        if not self._has_persisted_splat_state():
            return
        with self._rw_lock:
            self._write_splat_state()

    def _restore_splat_state(self) -> bool:
        """Re-aplica el estado aprendido (α, κ, μ) por id tras reconstruir el
        store desde storage. Devuelve True si aplicó al menos un splat."""
        if self.storage is None:
            return False
        try:
            state = self.storage.load_energy_state()
        except Exception as e:
            logger.warning("splat state not restored: %s: %s", type(e).__name__, e)
            return False
        if not state or not {"ids", "alpha", "kappa", "mu", "base"} <= set(state):
            return False  # sin estado (storage viejo) u otro formato: nada que aplicar
        ids = np.asarray(state["ids"]).reshape(-1).tolist()
        alpha = np.asarray(state["alpha"], dtype=np.float64).reshape(-1)
        kappa = np.asarray(state["kappa"], dtype=np.float64).reshape(-1)
        mu = np.asarray(state["mu"], dtype=np.float32)
        base = np.asarray(state["base"], dtype=np.float32)
        n_st = len(ids)
        if (
            len(alpha) != n_st
            or len(kappa) != n_st
            or mu.shape != (n_st, self.latent_dim)
            or base.shape != (n_st, self.latent_dim)
        ):
            logger.warning("splat state ignored: inconsistent shapes")
            return False

        store = self.engine.m2m.splats
        order = self._splat_id_order
        if len(order) != store.n_active:
            return False
        pos_of = {d: p for p, d in enumerate(order)}
        applied = 0
        drifted: List[int] = []
        for j, doc_id in enumerate(ids):
            p = pos_of.get(doc_id)
            v = self._vectors.get(doc_id) if p is not None else None
            if p is None or v is None:
                continue
            cur = normalize_sphere(np.asarray(v, dtype=np.float32).reshape(-1))
            if not np.allclose(cur, base[j], atol=1e-5):
                continue  # el doc cambió de vector después: el estado aprendido ya no aplica
            a, k = float(alpha[j]), float(kappa[j])
            if np.isfinite(a) and a > 0:
                store.alpha[p] = a
            if np.isfinite(k) and k > 0:
                store.kappa[p] = k
            m = mu[j]
            nrm = float(np.linalg.norm(m))
            if np.isfinite(m).all() and abs(nrm - 1.0) < 1e-3:
                store.mu[p] = m
                drifted.append(p)
            applied += 1
        if applied:
            store._m_sq_dirty = True
            self._sync_index_mu(drifted)
            # En el constructor el EBM todavía no existe (ebm_enabled se asigna
            # después de _restore_from_storage): ahí no hay nada que actualizar
            # [audit P15]; AdvancedVectorDB lo hace tras crear el SOC.
            if getattr(self, "ebm_enabled", False):
                self._update_ebm_splats()
        return applied > 0


from dataclasses import dataclass, field as _dataclass_field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # sólo para anotaciones: ConsolidationDecision se importa dentro del método
    from .ebm.soc import ConsolidationDecision


@dataclass
class FeedbackResult:
    """Resultado de ``SimpleVectorDB.feedback()`` [audit r3a-08].

    - ``n_relevant`` / ``n_irrelevant``: ids resueltos a un splat activo y
      aplicados (ids repetidos cuentan una vez).
    - ``updated_positions``: posiciones de splat tocadas EN ESA llamada. Son
      informativas: las posiciones son inestables (hard-delete y
      ``consolidate()`` compactan el store). Para seguir operando, usá ids.
    - ``unknown_ids``: ids ignorados por no existir o estar borrados.
    - ``summary``: contadores acumulados del ``OnlineUpdater``.
    """

    n_relevant: int
    n_irrelevant: int
    updated_positions: List[int] = _dataclass_field(default_factory=list)
    unknown_ids: List[str] = _dataclass_field(default_factory=list)
    summary: Dict[str, int] = _dataclass_field(default_factory=dict)

    def __repr__(self):
        return (
            f"FeedbackResult(n_relevant={self.n_relevant}, n_irrelevant={self.n_irrelevant}, "
            f"updated_positions={self.updated_positions}, unknown_ids={self.unknown_ids})"
        )


# ---------------------------------------------------------------------------
# AdvancedVectorDB - CRUD + SOC + EBM completo
# ---------------------------------------------------------------------------


class AdvancedVectorDB(SimpleVectorDB):
    """
    AdvancedVectorDB: Suite completa para agentes autónomos.

    Hereda de SimpleVectorDB y añade:
    1. Exploración (Langevin Dynamics): SOC + exploración del espacio latente.
    2. Context Management (3-tier memory): Hot VRAM, Warm RAM, Cold SSD.
    3. SOC Engine: Self-Organized Criticality con avalanches.
    4. EBM completo: Energía, exploración, confianza.
    5. Online learning: ``feedback()`` (heredado) y ``maybe_consolidate()``.

    Uso:
        db = AdvancedVectorDB(latent_dim=768)
        db.add(['doc1'], vectors, metadata=[{'cat': 'ai'}])
        result = db.search_with_energy(query, k=10)
        state = db.check_criticality()
        db.relax(iterations=10)
        decision = db.maybe_consolidate()   # consolida SÓLO en SUPERCRITICAL

    SOC sobre el store real [audit r3a-07]: ``check_criticality`` /
    ``trigger_avalanche`` / ``relax`` leen y escriben ``SplatStore`` (el
    ``SOCEngine`` recibe vistas ``[:n_active]`` sin copia) bajo ``_rw_lock``;
    avalanche, relax y feedback persisten el estado aprendido (α/κ/μ por id) y
    sobreviven a ``save()`` + reopen. Nada es automático por defecto: la
    consolidación guiada por criticalidad es explícita (``maybe_consolidate``) y
    la automática es opt-in (``auto_consolidate=True``).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        latent_dim: int = 640,
        storage_path: Optional[str] = None,
        enable_soc: bool = True,
        enable_energy_features: bool = True,
        auto_consolidate: bool = False,
    ):
        # SPEC-12: Pass advanced config directly to parent to avoid double engine creation
        config = M2MConfig.advanced(device=device)
        config.latent_dim = latent_dim

        super().__init__(
            device=device,
            latent_dim=latent_dim,
            storage_path=storage_path,
            enable_ebm=enable_energy_features,
            mode="ebm" if enable_energy_features else "standard",
            _config=config,
        )

        # SOC Engine
        self.soc_enabled = enable_soc
        # [audit r3a-07] automático SÓLO opt-in: con True, cada cambio del store
        # (_update_ebm_splats) evalúa la criticality y consolida si está SUPERCRITICAL.
        self.auto_consolidate = bool(auto_consolidate)
        # Guardia de reentrancia: consolidate() -> _update_ebm_splats() -> auto -> consolidate()
        self._soc_busy = False
        self._soc: Optional[SOCEngine] = None
        if enable_soc and self.ebm_enabled:
            self._soc = SOCEngine(self._ebm_energy)

        if self.ebm_enabled:
            # El estado aprendido (α/κ/μ) persistido: _restore_from_storage corre dentro
            # de super().__init__ ANTES de que existan ebm_enabled/_soc, así que se aplica acá.
            with self._rw_lock:
                self._restore_splat_state()

    def _update_ebm_splats(self):
        """Actualiza splats en EBM y SOC.

        Al SOC le pasan VISTAS del store (``SplatStore.mu/alpha/kappa[:n]``), no
        copias: así avalanche y relax escriben en el store real [audit r3a-07].
        Los clústeres del SOC son los coarse clusters del índice HRM2 cuando
        cubren exactamente los n splats activos.
        """
        super()._update_ebm_splats()
        soc = getattr(self, "_soc", None)
        if soc is None or not self.ebm_enabled:
            return
        store = self.engine.m2m.splats
        n = store.n_active
        if n > 0:
            labels = getattr(store.engine, "coarse_assignments", None)
            if labels is None or len(labels) != n:
                labels = None  # índice vencido (compact): el SOC agrupa por geometría
            soc.update_splats(
                store.mu[:n], store.alpha[:n], store.kappa[:n], cluster_labels=labels
            )
        else:
            soc.update_splats(
                np.zeros((0, self.latent_dim), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        if getattr(self, "auto_consolidate", False) and not getattr(self, "_soc_busy", False):
            self.maybe_consolidate()

    def consolidate(self, threshold: float = None) -> int:
        """Run Self-Organized Criticality memory consolidation.

        Podar deja el store consistente [audit r3a-07]: ``_splat_id_order`` se
        compacta con la misma máscara, los docs que pierden su único splat
        salen de ``_vectors``/``_metadata``/``_documents`` Y del storage (antes
        sólo salían de memoria y resucitaban al reabrir), y EBM/SOC dejan de
        apuntar a las filas podadas.
        """
        with self._rw_lock:
            mem = self.engine.m2m
            if threshold is None:
                threshold = mem.soc_threshold

            # Identify which splats will be removed before consolidation
            splats_to_remove = set()
            if mem.splats.n_active > 0:
                removed_indices = np.where(mem.splats.alpha[: mem.splats.n_active] < threshold)[0]
                splats_to_remove = set(int(i) for i in removed_indices)

            self._soc_busy = True
            try:
                # Call the underlying consolidate
                n_removed = mem.consolidate(threshold)

                # Clean up orphaned doc_ids from _vectors, _metadata, _documents
                if n_removed > 0 and self._splat_id_order:
                    # Build set of doc_ids to remove based on splat positions
                    orphans = {
                        self._splat_id_order[idx]
                        for idx in splats_to_remove
                        if idx < len(self._splat_id_order)
                    }
                    # Rebuild _splat_id_order keeping only surviving splats
                    kept = [
                        self._splat_id_order[i]
                        for i in range(len(self._splat_id_order))
                        if i not in splats_to_remove
                    ]
                    self._splat_id_order = kept
                    # Remove orphaned entries
                    for doc_id in orphans:
                        self._vectors.pop(doc_id, None)
                        self._metadata.pop(doc_id, None)
                        self._documents.pop(doc_id, None)
                        self._deleted.discard(doc_id)
                        if self.storage is not None:
                            self.storage.hard_delete(doc_id)
                    self._meta_version += 1  # [P13] invalida el cache de máscaras de filtro
                    logger.info("SOC cleanup: removed %d orphaned doc_ids", len(orphans))

                # EBM/SOC con los splats que quedan (antes seguían con los arrays previos)
                self._update_ebm_splats()
                self._write_splat_state()
            finally:
                self._soc_busy = False

        return n_removed

    def check_criticality(self) -> CriticalityReport:
        """
        Verifica el estado de criticalidad del sistema.

        Returns:
            CriticalityReport con estado SUBCRITICAL/CRITICAL/SUPERCRITICAL.
        """
        if self._soc is None:
            raise RuntimeError("SOC no habilitado. Usa enable_soc=True.")
        with self._rw_lock:
            return self._soc.check_criticality()

    def trigger_avalanche(self, seed_point: Optional[np.ndarray] = None) -> AvalancheResult:
        """
        Dispara una avalanche de reorganización del sistema (sobre el store real:
        baja α de los splats de los clústeres que se desbordan).

        Args:
            seed_point: Punto semilla. Si None, usa el clúster de mayor altura.

        Returns:
            AvalancheResult con estadísticas.
        """
        if self._soc is None:
            raise RuntimeError("SOC no habilitado. Usa enable_soc=True.")
        with self._rw_lock:
            self._soc_busy = True
            try:
                result = self._soc.trigger_avalanche(seed_point)
                self._after_soc_write()
            finally:
                self._soc_busy = False
        return result

    def relax(self, iterations: int = 10) -> RelaxationResult:
        """
        Relaja el sistema sin subir su energía libre (baja el exceso de
        concentración κ de los splats sobre-afilados; ver ``SOCEngine.relax``).

        Args:
            iterations: Iteraciones de relajación.

        Returns:
            RelaxationResult con energía libre inicial y final.
        """
        if self._soc is None:
            raise RuntimeError("SOC no habilitado. Usa enable_soc=True.")
        with self._rw_lock:
            self._soc_busy = True
            try:
                result = self._soc.relax(iterations)
                self._after_soc_write()
            finally:
                self._soc_busy = False
        return result

    def _after_soc_write(self) -> None:
        """Tras avalanche/relax (que escribieron en el store por referencia):
        invalida caches derivados de μ, refresca EBM/SOC y persiste el estado
        aprendido para que sobreviva a ``save()`` + reopen. Lock tomado."""
        self.engine.m2m.splats._m_sq_dirty = True
        self._update_ebm_splats()
        self._write_splat_state()

    def maybe_consolidate(self, threshold: float = None) -> "ConsolidationDecision":
        """Consolidación guiada por criticality: SÓLO si el índice está en SUPERCRITICAL.

        Explícita e idempotente [audit r3a-07]: si el estado no es SUPERCRITICAL
        no hace nada (``triggered=False``, ``removed=0``); tras consolidar, el
        peso muerto podado deja de alimentar el índice. Con
        ``auto_consolidate=True`` se invoca sola tras cada cambio del store.

        ``threshold`` por defecto = ``decay_threshold`` del SOC (α < 0.1) y NO
        ``soc_threshold`` (0.8): con 0.8 una vía semiautomática podaría también
        los splats que una avalanche acaba de dejar en 0.7.

        Args:
            threshold: α por debajo del cual un splat se poda (None -> decay_threshold).

        Returns:
            ConsolidationDecision con el estado/índice DESPUÉS de la decisión.
        """
        from .ebm.soc import ConsolidationDecision, CriticalityState

        if self._soc is None:
            raise RuntimeError("SOC no habilitado. Usa enable_soc=True.")
        with self._rw_lock:
            report = self._soc.check_criticality()
            thr = float(self._soc.decay_threshold if threshold is None else threshold)
            if report.state == CriticalityState.SUPERCRITICAL:
                removed = self.consolidate(thr)
                after = self._soc.check_criticality()
                return ConsolidationDecision(True, after.state, after.index, removed, thr)
            return ConsolidationDecision(False, report.state, report.index, 0, thr)

    def save(self, path: Optional[str] = None) -> str:
        """``SimpleVectorDB.save`` + el estado aprendido de los splats (α/κ/μ por id,
        ``energy/landscape.npz`` dentro del storage/snapshot) para que avalanche,
        relax y feedback sobrevivan al reopen [audit r3a-07]."""
        with self._rw_lock:
            self._write_splat_state()  # antes: save(path) copia el directorio entero
            out = super().save(path)
            if self.storage is None:
                # DB sólo en memoria: el snapshot se escribió sin nuestro estado
                self._write_splat_state(target_dir=out)
        return out

    def _restore_from_storage(self):
        """Restore base + estado aprendido de los splats.

        Durante ``super().__init__`` todavía no existe ``ebm_enabled`` (el
        restore base corre antes del bloque EBM): en ese caso lo aplica
        ``__init__`` de esta clase cuando el SOC ya está creado.
        """
        super()._restore_from_storage()
        if getattr(self, "ebm_enabled", False):
            with self._rw_lock:
                self._restore_splat_state()

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
            logger.warning("requests not installed. Install with: pip install requests")

    def _headers(self) -> Dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, path: str) -> Dict:
        if self._requests is None:
            raise RuntimeError("requests no instalado")
        r = self._requests.get(
            f"{self.base_url}{path}", headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: Dict) -> Dict:
        if self._requests is None:
            raise RuntimeError("requests no instalado")
        r = self._requests.post(
            f"{self.base_url}{path}",
            json=data,
            headers=self._headers(),
            timeout=self.timeout,
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
        return self._client._post(f"/v1/collections/{self.name}/explore", {"n_suggestions": n})

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
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "vulkan"],
        help="Device to use",
    )
    parser.add_argument("--n-splats", type=int, default=10000)
    parser.add_argument("--max-splats", type=int, default=100000)
    parser.add_argument("--knn-k", type=int, default=64)
    parser.add_argument("--enable-vulkan", action="store_true")
    parser.add_argument("--disable-vulkan", action="store_true")
    parser.add_argument(
        "--transform-dataset",
        nargs=2,
        metavar=("INPUT.npy", "OUTPUT.bin"),
        help="Transform embeddings dataset to M2M hierarchical splats",
    )
    parser.add_argument(
        "--load-optimized",
        type=str,
        metavar="INPUT.bin",
        help="Start M2M from a pre-transformed dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
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

        from .dataset_transformer import M2MDatasetTransformer

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
