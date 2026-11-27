"""
SOCEngine - Self-Organized Criticality para M2M.

El SOC permite que el sistema se auto-organice naturalmente.
Cuando acumula demasiada energía (supercritical), una "avalanche"
redistribuye la energía de manera natural, como arena en una pila.

Estados del sistema:
  SUBCRITICAL:   Sistema estable, sin reorganización necesaria
  CRITICAL:      Punto de transición, avalanche inminente
  SUPERCRITICAL: Sistema inestable, necesita reorganización

Cómo funciona sobre el store real [audit r2-16, r3a-07]
--------------------------------------------------------
- ``update_splats()`` REFERENCIA los arrays que le pasan (``np.asarray`` sin
  copia si ya son float32): ``AdvancedVectorDB`` le pasa vistas de
  ``SplatStore.mu/alpha/kappa[:n_active]``, así avalanche y relax escriben en
  el store directamente. Si pasás arrays float64, ``asarray`` los convierte y
  los cambios NO llegan a vuestros arrays: pasá float32.
- Los clústeres son reales: los coarse clusters del índice HRM2 cuando el
  llamador los pasa (``cluster_labels``), o un cuantizador geométrico
  determinista sobre μ (anclas unitarias + re-centrado tipo Lloyd) si no.
  Antes había un clúster por splat: ``size_variance`` era estructuralmente 0
  (40% del índice en cero) y el máximo alcanzable era 0.6 < 0.7, así que
  SUPERCRITICAL era inalcanzable. Y el constructor documentado no llamaba a
  ``_rebuild_clusters``: 0 clústeres e índice 0 siempre.
- El índice de criticalidad combina cuatro señales reales de μ/κ/α, cada una
  en [0, 1], con una OR probabilística ``1 - Π(1 - cᵢ)`` (índice = "cuánta
  inestabilidad hay, al menos en un eje"; ninguna fracción es estructuralmente
  cero):
    * ``decayed``  (α): fracción de splats con α < ``decay_threshold`` — peso
      muerto que una consolidación podaría. Es la señal de SOBRECARGA: con el
      80% de los splats de α bajo el índice ya es ≥ 0.8.
    * ``energy``   (α, μ): desbalance de energía media por clúster
      (``cv(mean α_c)``): clústeres "calientes" vs "fríos" — lo que la
      avalanche redistribuye.
    * ``size``     (μ): desbalance de tamaños de clúster (``cv(n_c)``): un
      clúster gigante es un hot-spot del IVF.
    * ``kappa``    (κ): dispersión de concentración (``cv(κ)``): agujas
      sobre-afiladas junto a splats anchos — lo que ``relax()`` suaviza.
  Para las tres señales de desbalance, cv por debajo de
  ``imbalance_cv_low`` vale 0 y desde ``imbalance_cv_high`` vale 1 (un store
  fresco con clústeres de tamaños moderadamente distintos queda SUBCRITICAL).
- ``relax()`` ya no sube la energía libre F = -log Z: Z = Σ αᵢ(π/κᵢ)^(D/2)
  no depende de μ, y decaer α u homogeneizar κ (f convexa, Jensen) sólo
  reducen Z (ΔF ≥ 0  siempre: la sonda del auditor daba +0.001 sin mejorar
  nunca). Ahora relax libera el exceso de CONCENTRACIÓN: cada splat cuyo κ
  supera la concentración de equilibrio de su vecindario (κ* = 4/r², con r la
  distancia al splat más cercano: dos splats vecinos se cubren hasta la mitad
  del camino) baja κ hacia κ*. Bajar κ aumenta Z término a término (log-sum-exp
  monótona), así F es no-creciente por construcción y estrictamente decreciente
  cuando hay splats sobre-afilados. No toca μ (es el embedding del documento:
  moverlo cambiaba el ranking de search()) ni α (es la energía del SOC: la
  manejan feedback/avalanche/consolidate).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .energy_api import EBMEnergy


class CriticalityState(Enum):
    """Estado de criticalidad del sistema."""

    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"


@dataclass
class CriticalityReport:
    """Reporte del estado de criticalidad.

    - ``index``: 0-1, mayor = más inestable (ver módulo).
    - ``energy_variance`` / ``size_variance``: varianza (sin normalizar) de la
      energía total (Σα) y del tamaño entre clústeres REALES.
    - ``decayed_fraction``: fracción de splats con α < ``decay_threshold``.
    - ``kappa_dispersion``: coeficiente de variación de κ (σ/μ).
    - ``components``: las cuatro señales normalizadas en [0, 1].
    """

    state: CriticalityState
    index: float  # 0-1, mayor = más inestable
    energy_variance: float
    size_variance: float
    n_clusters: int = 0
    decayed_fraction: float = 0.0
    kappa_dispersion: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    def needs_relaxation(self) -> bool:
        """True si el sistema necesita relajación."""
        return self.state == CriticalityState.SUPERCRITICAL

    def needs_monitoring(self) -> bool:
        """True si el sistema está en estado crítico."""
        return self.state in (CriticalityState.CRITICAL, CriticalityState.SUPERCRITICAL)


@dataclass
class AvalancheResult:
    """Resultado de una avalanche de reorganización."""

    affected_clusters: int
    energy_released: float
    duration_ms: float
    new_equilibrium: float

    def summary(self) -> str:
        return (
            f"Avalanche: {self.affected_clusters} clusters afectados, "
            f"energía liberada={self.energy_released:.4f}, "
            f"duración={self.duration_ms:.1f}ms"
        )


@dataclass
class RelaxationResult:
    """Resultado de relajación del sistema.

    ``initial_energy`` / ``final_energy`` son la energía libre F = -log Z del
    ``energy_api`` antes y después: ``final_energy <= initial_energy`` por
    construcción (audit r2-16).
    """

    initial_energy: float
    final_energy: float
    energy_delta: float
    iterations: int

    @property
    def improved(self) -> bool:
        """True si la relajación mejoró el sistema."""
        return self.final_energy < self.initial_energy


@dataclass
class ConsolidationDecision:
    """Resultado de ``AdvancedVectorDB.maybe_consolidate()`` [audit r3a-07].

    ``triggered`` es True sólo si el índice de criticalidad estaba en
    SUPERCRITICAL; en ese caso ``removed`` es lo que podó la consolidación.
    ``state`` / ``index`` son los de DESPUÉS de la decisión.
    """

    triggered: bool
    state: CriticalityState
    index: float
    removed: int
    threshold: float

    def __repr__(self):
        return (
            f"ConsolidationDecision(triggered={self.triggered}, state={self.state.value}, "
            f"index={self.index:.4f}, removed={self.removed}, threshold={self.threshold})"
        )


@dataclass
class _Cluster:
    """Cluster interno para SOC.

    ``energy`` es la amplitud total Σα de sus splats (cantidad extensiva, la
    que se "libera"); la altura media por splat (``energy / size``) es lo que
    se compara con ``critical_threshold`` en la avalanche — con clústeres de
    un splat coincide con el criterio original (α > umbral).
    """

    center: np.ndarray
    energy: float
    size: int
    splat_indices: List[int]

    def reorganize(self):
        """Marca el cluster como reorganizado, reduciendo su energía."""
        self.energy *= 0.7  # Libera el 30%

    def receive_energy(self, amount: float):
        """Recibe energía de clusters vecinos."""
        self.energy += amount


def _cv(values: np.ndarray) -> float:
    """Coeficiente de variación σ/μ (0 si μ <= 0 o sin datos)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    m = float(x.mean())
    return float(x.std() / m) if m > 0.0 else 0.0


def _ramp(cv: float, low: float, high: float) -> float:
    """cv <= low -> 0; cv >= high -> 1; lineal en el medio."""
    return float(np.clip((cv - low) / max(high - low, 1e-12), 0.0, 1.0))


def _geometric_labels(mu: np.ndarray, max_points_for_lloyd: int = 4096) -> np.ndarray:
    """Clusters geométricos baratos y deterministas sobre μ [N, D] (esfera unidad).

    Cuantizador esférico: ``K = round(sqrt(N))`` anclas unitarias aleatorias
    (seed fija) y asignación por argmax(μ·ancla); con N acotado, unas
    iteraciones de re-centrado tipo Lloyd. O(N·K·D), vectorizado, sin
    dependencias. Devuelve labels [N] en [0, K).
    """
    n = int(len(mu))
    k = int(min(n, max(2, round(np.sqrt(n)))))
    rng = np.random.default_rng(12345)
    x = np.asarray(mu, dtype=np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    anchors = rng.normal(size=(k, x.shape[1])).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-12
    labels = np.argmax(x @ anchors.T, axis=1)
    for _ in range(5 if n <= max_points_for_lloyd else 0):
        sums = np.zeros_like(anchors)
        np.add.at(sums, labels, x)
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        new_anchors = np.where(norms > 1e-12, sums / (norms + 1e-12), anchors)
        new_labels = np.argmax(x @ new_anchors.T, axis=1)
        anchors = new_anchors
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels


def _nearest_neighbor_distance(mu: np.ndarray, sample_if_larger: int = 4096) -> np.ndarray:
    """Distancia de cada splat a su vecino más cercano (otro splat) [N].

    Gram-trick por bloques de 1024 filas (memoria acotada). Con N mayor que
    ``sample_if_larger`` se mide contra una muestra fija de ``sample_if_larger``
    splats (aproximación por arriba de r, O(N·S·D)).
    """
    x = np.asarray(mu, dtype=np.float32)
    n = len(x)
    if n < 2:
        return np.full(n, np.inf, dtype=np.float32)
    if n > sample_if_larger:
        ref_idx = np.random.default_rng(777).choice(n, size=sample_if_larger, replace=False)
        ref = x[ref_idx]
    else:
        ref_idx = None
        ref = x
    sq = np.einsum("ij,ij->i", ref, ref)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, 1024):
        block = x[start : start + 1024]
        d2 = (
            np.einsum("ij,ij->i", block, block)[:, None]
            - 2.0 * (block @ ref.T)
            + sq[None, :]
        )
        # descartar la comparación consigo mismo (misma fila si la muestra lo incluye)
        if ref_idx is None:
            rows = np.arange(start, start + len(block))
            d2[np.arange(len(block)), rows] = np.inf
        else:
            self_hits = ref_idx[None, :] == np.arange(start, start + len(block))[:, None]
            d2[self_hits] = np.inf
        out[start : start + len(block)] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


class SOCEngine:
    """
    Motor de Self-Organized Criticality para M2M.

    Detecta cuando el sistema se vuelve inestable (supercritical)
    y desencadena avalanches para redistribuir la energía naturalmente.

    Uso:
        soc = SOCEngine(energy_api, splat_mu, splat_alpha, splat_kappa)
        report = soc.check_criticality()
        if report.needs_relaxation():
            result = soc.trigger_avalanche()
        soc.relax(iterations=10)

    ``splat_mu/alpha/kappa`` (float32) se REFERENCIAN, no se copian: avalanche y
    relax los modifican in-place (ver módulo). El motor no dispara nada por sí
    mismo: ``check_criticality``/``trigger_avalanche``/``relax`` son explícitos;
    en ``AdvancedVectorDB`` la vía automática es opt-in (``auto_consolidate``).

    Args:
        energy_api: EBMEnergy del sistema (leerá F y recibirá los splats).
        splat_mu: Centroides de splats [N, D]
        splat_alpha: Amplitudes [N]
        splat_kappa: Concentraciones [N]
        critical_threshold: Altura media de α por splat (energía del clúster /
            tamaño) a partir de la cual un clúster se desborda en una avalanche.
        decay_threshold: α por debajo del cual un splat cuenta como decaído
            (peso muerto). Es también el umbral por defecto de
            ``AdvancedVectorDB.maybe_consolidate()``.
        imbalance_cv_low / imbalance_cv_high: mapeo de los cv de desbalance
            (energía, tamaño, κ) a [0, 1] (ver módulo).
        alpha_floor: la avalanche nunca baja α por debajo de este valor.
    """

    def __init__(
        self,
        energy_api: EBMEnergy,
        splat_mu: Optional[np.ndarray] = None,
        splat_alpha: Optional[np.ndarray] = None,
        splat_kappa: Optional[np.ndarray] = None,
        critical_threshold: float = 0.8,
        decay_threshold: float = 0.1,
        imbalance_cv_low: float = 0.5,
        imbalance_cv_high: float = 2.0,
        alpha_floor: float = 0.01,
    ):
        self.energy_api = energy_api
        self.critical_threshold = critical_threshold
        self.decay_threshold = decay_threshold
        self.imbalance_cv_low = imbalance_cv_low
        self.imbalance_cv_high = imbalance_cv_high
        self.alpha_floor = alpha_floor
        self.avalanche_history: List[AvalancheResult] = []

        self.splat_mu = None
        self.splat_alpha = None
        self.splat_kappa = None
        self.cluster_labels: Optional[np.ndarray] = None
        self._clusters: List[_Cluster] = []
        self._clusters_dirty = False

        if splat_mu is not None:
            # uso documentado: los clústeres se arman desde el arranque (audit r2-16)
            self.update_splats(splat_mu, splat_alpha, splat_kappa)

    # ------------------------------------------------------------------
    # Splats y clústeres
    # ------------------------------------------------------------------

    def update_splats(
        self,
        splat_mu: np.ndarray,
        splat_alpha: np.ndarray,
        splat_kappa: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
    ):
        """Actualiza los splats (por referencia si son float32) y marca los
        clústeres para reconstruir.

        Args:
            splat_mu, splat_alpha, splat_kappa: arrays del store [N, D] / [N] / [N].
            cluster_labels: etiqueta de clúster coarse por splat [N] (p. ej.
                ``HRM2Engine.coarse_assignments``). None o longitud distinta de
                N -> clústeres geométricos sobre μ.
        """
        mu = np.asarray(splat_mu, dtype=np.float32)
        alpha = np.asarray(splat_alpha, dtype=np.float32)
        kappa = np.asarray(splat_kappa, dtype=np.float32)
        if mu.ndim != 2 or len(alpha) != len(mu) or len(kappa) != len(mu):
            raise ValueError(
                "splat_mu must be [N, D] and splat_alpha/splat_kappa must have length N "
                f"(got {mu.shape}, {alpha.shape}, {kappa.shape})"
            )
        self.splat_mu = mu
        self.splat_alpha = alpha
        self.splat_kappa = kappa
        if cluster_labels is None or len(cluster_labels) != len(mu):
            self.cluster_labels = None
        else:
            self.cluster_labels = np.asarray(cluster_labels).reshape(-1)
        self._clusters_dirty = True

    def _ensure_clusters(self) -> List[_Cluster]:
        """Reconstruye los clústeres si los splats cambiaron (diferido: update_splats
        se llama tras cada add, y armar clústeres ahí sería costo en el hot path)."""
        if not self._clusters_dirty:
            return self._clusters
        self._clusters_dirty = False
        mu = self.splat_mu
        if mu is None or len(mu) == 0:
            self._clusters = []
            return self._clusters
        n = len(mu)
        labels = self.cluster_labels
        if labels is None:
            labels = _geometric_labels(mu)
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        bounds = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
        clusters: List[_Cluster] = []
        for idx in np.split(order, bounds):
            idx = np.asarray(idx, dtype=np.int64)
            members = mu[idx]
            center = members.mean(axis=0)
            nrm = float(np.linalg.norm(center))
            center = center / nrm if nrm > 1e-12 else members[0]
            clusters.append(
                _Cluster(
                    center=center,
                    energy=float(self.splat_alpha[idx].sum()),
                    size=int(len(idx)),
                    splat_indices=idx.tolist(),
                )
            )
        self._clusters = clusters
        return clusters

    def _rebuild_clusters(self):
        """Compatibilidad: fuerza la reconstrucción de los clústeres en el próximo uso."""
        self._clusters_dirty = True

    # ------------------------------------------------------------------
    # Criticalidad
    # ------------------------------------------------------------------

    def _compute_energy_variance(self) -> float:
        """Varianza de energía (Σα) entre clústeres."""
        clusters = self._ensure_clusters()
        if not clusters:
            return 0.0
        return float(np.var([c.energy for c in clusters]))

    def _compute_criticality_index(self, energy_variance: float, size_variance: float) -> float:
        """Índice de criticalidad combinado [audit r2-16].

        (Mantiene la firma histórica; los dos argumentos son informativos.)
        Combina cuatro señales reales de μ/κ/α con una OR probabilística
        ``1 - Π(1 - cᵢ)``: ver el módulo. Ninguna es estructuralmente cero.
        """
        c = self._components()
        index = 1.0
        for value in c.values():
            index *= 1.0 - float(value)
        return float(1.0 - index)

    def _components(self) -> Dict[str, float]:
        clusters = self._ensure_clusters()
        if not clusters or self.splat_alpha is None or self.splat_kappa is None:
            return {"decayed": 0.0, "energy": 0.0, "size": 0.0, "kappa": 0.0}
        sizes = np.array([c.size for c in clusters], dtype=np.float64)
        mean_alpha = np.array([c.energy for c in clusters], dtype=np.float64) / np.maximum(
            sizes, 1.0
        )
        lo, hi = self.imbalance_cv_low, self.imbalance_cv_high
        return {
            "decayed": float(np.mean(self.splat_alpha < self.decay_threshold)),
            "energy": _ramp(_cv(mean_alpha), lo, hi),
            "size": _ramp(_cv(sizes), lo, hi),
            "kappa": _ramp(_cv(self.splat_kappa), lo, hi),
        }

    def check_criticality(self) -> CriticalityReport:
        """
        Verifica si el sistema está en estado crítico.

        Returns:
            CriticalityReport con estado y métricas.
        """
        clusters = self._ensure_clusters()
        energy_variance = self._compute_energy_variance()
        cluster_sizes = [len(c.splat_indices) for c in clusters] if clusters else [0]
        size_variance = float(np.var(cluster_sizes))

        components = self._components()
        index = self._compute_criticality_index(energy_variance, size_variance)

        if index < 0.3:
            state = CriticalityState.SUBCRITICAL
        elif index < 0.7:
            state = CriticalityState.CRITICAL
        else:
            state = CriticalityState.SUPERCRITICAL

        return CriticalityReport(
            state=state,
            index=index,
            energy_variance=energy_variance,
            size_variance=size_variance,
            n_clusters=len(clusters),
            decayed_fraction=components["decayed"],
            kappa_dispersion=_cv(self.splat_kappa) if self.splat_kappa is not None else 0.0,
            components=components,
        )

    # ------------------------------------------------------------------
    # Avalanche
    # ------------------------------------------------------------------

    def trigger_avalanche(self, seed_point: Optional[np.ndarray] = None) -> AvalancheResult:
        """
        Dispara una avalanche de reorganización SOBRE LOS SPLATS REALES.

        Un clúster se desborda si su altura media por splat (Σα / tamaño)
        supera ``critical_threshold``; entonces cada uno de sus splats libera
        el 30% de su α (piso ``alpha_floor``) y la cascada continúa por sus
        vecinos (los 10 clústeres más cercanos, radio <= 2 en la esfera) que
        también estén por encima del umbral. Antes la avalanche trabajaba sobre
        una copia (``get_splats_arrays()`` devuelve ``.copy()``): el store no
        cambiaba nunca [audit r3a-07].

        Puede ser:
        - Automática: cuando se llama después de check_criticality()
        - Manual: disparada explícitamente por el usuario

        Args:
            seed_point: Punto semilla. Si None, usa el clúster de mayor altura.

        Returns:
            AvalancheResult con estadísticas de la avalanche.
        """
        import time

        start_time = time.time()
        clusters = self._ensure_clusters()
        new_eq = self._compute_equilibrium_state()
        if not clusters or self.splat_alpha is None:
            return AvalancheResult(
                affected_clusters=0,
                energy_released=0.0,
                duration_ms=0.0,
                new_equilibrium=new_eq,
            )

        # Punto semilla: clúster más cercano a seed_point, o el de mayor altura media
        centers = np.stack([c.center for c in clusters])
        if seed_point is not None and self.splat_mu is not None:
            seed = np.asarray(seed_point, dtype=np.float32).reshape(-1)
            start_cluster_idx = int(np.argmin(np.linalg.norm(centers - seed, axis=1)))
        else:
            heights = np.array([c.energy / max(c.size, 1) for c in clusters])
            start_cluster_idx = int(np.argmax(heights))

        # Vecindario por clúster, vectorizado (Gram entre centros; esfera unidad -> d² = 2 - 2c·c')
        neighbors = self._neighbor_table(centers)

        # BFS de avalanche
        affected_clusters: List[_Cluster] = []
        queue = [start_cluster_idx]
        visited = set()
        total_energy_released = 0.0
        max_cascade = min(1000, len(clusters))
        alpha = self.splat_alpha

        while queue and len(affected_clusters) < max_cascade:
            idx = queue.pop(0)
            if idx in visited:
                continue
            visited.add(idx)
            cluster = clusters[idx]
            if cluster.size == 0 or cluster.energy / cluster.size <= self.critical_threshold:
                continue
            affected_clusters.append(cluster)
            energy_released = cluster.energy * 0.3  # Libera 30%
            total_energy_released += energy_released
            idxs = np.asarray(cluster.splat_indices, dtype=np.int64)
            idxs = idxs[idxs < len(alpha)]
            if len(idxs):
                # nunca por debajo del piso (alpha < 0.01 es "muerto" para compact())
                alpha[idxs] = np.maximum(alpha[idxs] * 0.7, np.minimum(alpha[idxs], self.alpha_floor))
            cluster.reorganize()
            # La energía liberada empuja a los vecinos: los que están por encima del
            # umbral se desbordan a su vez (cascada).
            for n_idx in neighbors[idx]:
                n_cluster = clusters[n_idx]
                if n_idx not in visited and n_cluster.energy / max(n_cluster.size, 1) > self.critical_threshold:
                    queue.append(n_idx)

        if affected_clusters:
            # los _Cluster ya reflejan el α nuevo (reorganize()); el energy_api también
            self.energy_api.update_splats(self.splat_mu, self.splat_alpha, self.splat_kappa)

        new_eq = self._compute_equilibrium_state()
        duration_ms = (time.time() - start_time) * 1000

        result = AvalancheResult(
            affected_clusters=len(affected_clusters),
            energy_released=total_energy_released,
            duration_ms=duration_ms,
            new_equilibrium=new_eq,
        )
        self.avalanche_history.append(result)
        return result

    def _neighbor_table(self, centers: np.ndarray, radius: float = 2.0, max_neighbors: int = 10):
        """Vecinos de cada clúster (índices) dentro de ``radius``, hasta los
        ``max_neighbors`` más cercanos. Devuelve lista de arrays."""
        k = len(centers)
        if k <= 1:
            return [np.zeros(0, dtype=np.int64) for _ in range(k)]
        d2 = (
            np.einsum("ij,ij->i", centers, centers)[:, None]
            - 2.0 * (centers @ centers.T)
            + np.einsum("ij,ij->i", centers, centers)[None, :]
        )
        np.maximum(d2, 0.0, out=d2)
        np.fill_diagonal(d2, np.inf)
        out = []
        for i in range(k):
            row = d2[i]
            cand = np.flatnonzero(row <= radius * radius)
            if len(cand) > max_neighbors:
                cand = cand[np.argsort(row[cand])[:max_neighbors]]
            else:
                cand = cand[np.argsort(row[cand])]
            out.append(cand.astype(np.int64))
        return out

    def _get_neighbor_clusters(self, cluster_idx: int, radius: float = 2.0) -> List[int]:
        """Obtiene índices de clústeres vecinos dentro de un radio (compatibilidad)."""
        clusters = self._ensure_clusters()
        if not clusters or cluster_idx >= len(clusters):
            return []
        centers = np.stack([c.center for c in clusters])
        return self._neighbor_table(centers, radius=radius)[cluster_idx].tolist()

    def _compute_equilibrium_state(self) -> float:
        """Altura media de α por splat (con clústeres de un splat = media de α)."""
        if self.splat_alpha is None or len(self.splat_alpha) == 0:
            return 0.0
        return float(np.mean(self.splat_alpha))

    # ------------------------------------------------------------------
    # Relax
    # ------------------------------------------------------------------

    def relax(self, iterations: int = 10, lr: float = 0.01) -> RelaxationResult:
        """
        Relaja el sistema hacia un estado de MENOR energía libre sin subirla jamás.

        Cada iteración, para todo splat con κ por encima de la concentración de
        equilibrio de su vecindario (κ* = 4 / r², r = distancia al splat más
        cercano; tope [``kappa_min``, ``kappa_max``]):
            κ ← κ + lr·(κ* − κ)      (sólo baja: nunca afila)

        Por qué así [audit r2-16]: F = -log Z con Z = Σ αᵢ(π/κᵢ)^(D/2) no
        depende de μ, y decaer α u homogeneizar κ sólo reducen Z (ΔF >= 0
        siempre; la sonda del auditor medía +0.001). Bajar κ aumenta Z término a
        término, así F es no-creciente por construcción y baja de forma estricta
        mientras haya splats sobre-afilados (p. ej. κ saturada en kappa_max por
        feedback relevante). μ (el embedding del documento) y α (la energía del
        SOC) no se tocan: mover μ contraía los splats hacia el centro de masa
        global, los sacaba de la esfera y cambiaba el ranking de search().

        Args:
            iterations: Número de pasos.
            lr: Paso relativo hacia κ* por iteración (0 < lr <= 1).

        Returns:
            RelaxationResult con la energía libre antes/después.
        """
        initial_energy = self.energy_api.free_energy()

        if self.splat_mu is None or self.splat_alpha is None or self.splat_kappa is None:
            return RelaxationResult(
                initial_energy=initial_energy,
                final_energy=initial_energy,
                energy_delta=0.0,
                iterations=0,
            )
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0 (got {iterations})")
        if not 0.0 < lr <= 1.0:
            raise ValueError(f"lr must be in (0, 1] (got {lr})")

        iterations = int(iterations)
        kappa = self.splat_kappa
        if iterations > 0 and len(self.splat_mu) > 1:
            r = _nearest_neighbor_distance(self.splat_mu)
            with np.errstate(divide="ignore", invalid="ignore"):
                kappa_star = 4.0 / np.maximum(r.astype(np.float64) ** 2, 1e-12)
            kappa_star = np.clip(kappa_star, 0.5, 100.0)
            done = 0
            for _ in range(iterations):
                over = (kappa > kappa_star) & (kappa > 0.5)
                if not np.any(over):
                    break
                kappa[over] = (kappa[over] + lr * (kappa_star[over] - kappa[over])).astype(np.float32)
                done += 1
            iterations = done if done else iterations

        # Actualizar energy_api con splats relajados
        self.energy_api.update_splats(self.splat_mu, self.splat_alpha, self.splat_kappa)

        final_energy = self.energy_api.free_energy()

        return RelaxationResult(
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_delta=final_energy - initial_energy,
            iterations=iterations,
        )

    def get_statistics(self) -> dict:
        """Estadísticas del SOC engine."""
        report = self.check_criticality()
        return {
            "criticality_state": report.state.value,
            "criticality_index": report.index,
            "n_clusters": len(self._ensure_clusters()),
            "n_avalanches": len(self.avalanche_history),
            "last_avalanche_clusters": (
                self.avalanche_history[-1].affected_clusters if self.avalanche_history else 0
            ),
            "equilibrium_state": self._compute_equilibrium_state(),
            "decayed_fraction": report.decayed_fraction,
            "kappa_dispersion": report.kappa_dispersion,
            "components": dict(report.components),
        }
