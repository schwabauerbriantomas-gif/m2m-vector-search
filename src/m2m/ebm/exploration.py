"""
EBMExploration - Exploración del paisaje energético.

Permite a agentes y LLMs descubrir regiones de alta incertidumbre
y sugerir áreas para expandir el conocimiento.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .energy_api import EBMEnergy


@dataclass
class EnergyRegion:
    """Una región del espacio con energía específica."""

    center: np.ndarray  # Centro de la región
    energy: float  # Energía promedio de la región
    radius: float  # Radio estimado
    n_points: int  # Puntos en la región


@dataclass
class ExplorationSuggestion:
    """Sugerencia de exploración para un agente."""

    region: EnergyRegion
    description: str
    suggested_queries: List[str]
    potential_value: float  # Alta energía = alto potencial de descubrimiento


class EBMExploration:
    """
    API de Exploración del paisaje energético.

    Permite:
    - Encontrar zonas de alta incertidumbre (alta energía)
    - Muestrear puntos en regiones inexploradas
    - Generar sugerencias de exploración para agentes

    Uso:
        exploration = EBMExploration(energy_api, all_vectors)
        regions = exploration.find_high_energy_regions(min_energy=0.7)
        uncertain = exploration.sample_uncertain(k=5)
        suggestions = exploration.suggest_exploration()
    """

    def __init__(
        self,
        energy_api: EBMEnergy,
        all_vectors: Optional[np.ndarray] = None,
        all_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            energy_api: EBMEnergy con splats del sistema
            all_vectors: Todos los vectores indexados [N, D]
            all_ids: IDs correspondientes
        """
        self.energy_api = energy_api
        self.all_vectors = all_vectors
        self.all_ids = all_ids

    def update_vectors(self, vectors: np.ndarray, ids: List[str]):
        """Actualiza los vectores de referencia."""
        self.all_vectors = vectors.astype(np.float32)
        self.all_ids = ids

    def find_high_energy_regions(
        self,
        topic_vector: Optional[np.ndarray] = None,
        min_energy: float = 0.7,
        n_regions: int = 5,
        n_samples: int = 500,
    ) -> List[EnergyRegion]:
        """
        Encuentra regiones de alta energía (incertidumbre).

        Args:
            topic_vector: Si se proporciona, busca en la vecindad.
                         Si no, busca globalmente.
            min_energy: Umbral mínimo de energía
            n_regions: Número de regiones a retornar
            n_samples: Puntos de muestra para exploración

        Returns:
            Lista de regiones ordenadas por energía descendente.
        """
        if self.all_vectors is None or len(self.all_vectors) == 0:
            return []

        dim = self.all_vectors.shape[1]

        # Muestrear puntos candidatos
        if topic_vector is not None:
            candidates = self._sample_around(
                np.asarray(topic_vector, dtype=np.float32),
                n=n_samples,
                radius=2.0,
                dim=dim,
            )
        else:
            candidates = self._sample_global(n=n_samples, dim=dim)

        # Calcular energía y filtrar
        high_energy_points = []
        for point in candidates:
            e = self.energy_api.energy(point)
            if e >= min_energy:
                high_energy_points.append((point, e))

        if not high_energy_points:
            return []

        # Clustering simple by greedy grouping
        return self._cluster_high_energy_points(high_energy_points, n_regions)

    def _sample_around(
        self, center: np.ndarray, n: int, radius: float, dim: int
    ) -> List[np.ndarray]:
        """Muestrea puntos alrededor de un centro."""
        noise = np.random.randn(n, dim).astype(np.float32) * radius
        return [center + noise[i] for i in range(n)]

    def _sample_global(self, n: int, dim: int) -> List[np.ndarray]:
        """Muestrea puntos globalmente, interpolando entre vectores conocidos."""
        if self.all_vectors is None or len(self.all_vectors) == 0:
            return [np.random.randn(dim).astype(np.float32) for _ in range(n)]

        # Mezcla de vectores conocidos con ruido
        indices = np.random.randint(0, len(self.all_vectors), n)
        base = self.all_vectors[indices].astype(np.float32)
        noise = np.random.randn(n, dim).astype(np.float32) * 0.5
        return [base[i] + noise[i] for i in range(n)]

    def _cluster_high_energy_points(
        self,
        points_energies: List[Tuple[np.ndarray, float]],
        n_clusters: int,
    ) -> List[EnergyRegion]:
        """Agrupa puntos de alta energía en regiones usando distancia greedy."""
        if not points_energies:
            return []

        # Ordenar por energía descendente
        points_energies.sort(key=lambda x: x[1], reverse=True)

        regions = []
        assigned = [False] * len(points_energies)
        min_radius = 0.5

        for i, (pt, e) in enumerate(points_energies):
            if assigned[i]:
                continue
            # Crear región con este punto como centro
            cluster_pts = [pt]
            for j, (pt2, e2) in enumerate(points_energies):
                if i != j and not assigned[j]:
                    dist = np.linalg.norm(pt - pt2)
                    if dist < min_radius:
                        cluster_pts.append(pt2)
                        assigned[j] = True
            assigned[i] = True

            center = np.mean(cluster_pts, axis=0).astype(np.float32)
            region = EnergyRegion(
                center=center,
                energy=e,
                radius=min_radius,
                n_points=len(cluster_pts),
            )
            regions.append(region)

            if len(regions) >= n_clusters:
                break

        return regions

    def sample_uncertain(
        self,
        k: int = 5,
        temperature: float = 1.0,
        from_region: Optional[EnergyRegion] = None,
    ) -> List[np.ndarray]:
        """
        Muestrea puntos de regiones inciertas (alta energía).

        Usa muestreo de Boltzmann donde la temperatura controla exploración:
        - T baja: puntos de energía más alta (más inciertos)
        - T alta: exploración más uniforme

        Args:
            k: Número de puntos a muestrear
            temperature: T ≥ 0 del muestreo de Boltzmann (pesos ∝ exp(E / T)).
                T < 0 o no finita (NaN/inf) -> ValueError (antes una T negativa se
                reemplazaba en silencio por 1e-10). Cualquier T > 0, por chica que sea,
                es válida. T = 0 -> devuelve los k candidatos de MAYOR energía (el
                límite determinista de la distribución: este sampler busca regiones
                inciertas) sin muestrear y sin lanzar [audit r2-17].
            from_region: Si se proporciona, muestrea de esa región específica

        Returns:
            Lista de k vectores en regiones inciertas.
        """
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError(
                f"temperature must be a finite number >= 0 (got {temperature!r})"
            )

        if self.all_vectors is None or len(self.all_vectors) == 0:
            return []

        dim = self.all_vectors.shape[1]

        if from_region is not None:
            candidates = self._sample_around(
                from_region.center, n=k * 10, radius=from_region.radius, dim=dim
            )
        else:
            # Muestrear globalmente con preferencia a regiones no conocidas
            candidates = self._sample_global(n=k * 10, dim=dim)

        # Energías de cada candidato (vectorizado)
        cand_array = np.stack(candidates)
        energies = self.energy_api.energy_batch(cand_array)

        # Muestreo de Boltzmann: ∝ exp(E / T) (mayor energía = mayor prob).
        # Gumbel-top-k sobre los logits E/T ≡ muestreo SIN reemplazo de esa
        # distribución, pero no falla cuando (T chica) casi todos los pesos hacen
        # underflow a 0: np.random.choice(p=...) lanzaba
        # 'Fewer non-zero entries in p than size' para T <= 0.01 [audit r2-17].
        # Estabilidad: se resta max(E) ANTES de dividir por T (E - max(E) <= 0 es
        # finito, así (E - max)/T nunca produce inf - inf).
        n_cand = len(candidates)
        k_actual = min(k, n_cand)
        if temperature == 0.0:
            order = np.argsort(-energies, kind="stable")[:k_actual]
            return [candidates[int(i)] for i in order]
        e64 = energies.astype(np.float64)
        with np.errstate(over="ignore", divide="ignore"):
            logits = (e64 - float(np.max(e64))) / temperature
        noise = np.random.gumbel(size=n_cand)
        gumbel = logits + noise
        if np.isinf(logits).any():
            # T tan chica que (E - max)/T hace overflow: esos logits quedan en -inf
            # empatados y el ruido ya no puede ordenarlos. Su orden real es el de la
            # energía (mismo T): se desempata por E y después por el ruido, que es
            # exactamente el límite T -> 0 (top-k por energía) sin perder los empates
            # exactos de E, que sí deben sortearse.
            order = sorted(
                range(n_cand), key=lambda i: (gumbel[i], float(e64[i]), noise[i]), reverse=True
            )
            pick = order[:k_actual]
        elif k_actual < n_cand:
            pick = np.argpartition(-gumbel, k_actual - 1)[:k_actual]
        else:
            pick = range(n_cand)
        return [candidates[int(i)] for i in pick]

    def suggest_exploration(
        self,
        current_knowledge: Optional[List[str]] = None,
        n_suggestions: int = 3,
    ) -> List[ExplorationSuggestion]:
        """
        Sugiere áreas a explorar basándose en gaps en el paisaje energético.

        Args:
            current_knowledge: IDs de documentos conocidos
            n_suggestions: Número de sugerencias

        Returns:
            Lista de ExplorationSuggestion con regiones prioritarias.
        """
        if self.all_vectors is None or len(self.all_vectors) == 0:
            return []

        # Encontrar regiones de alta incertidumbre
        regions = self.find_high_energy_regions(n_regions=n_suggestions * 2)

        suggestions = []
        for region in regions[:n_suggestions]:
            # Encontrar vectores cercanos a la región
            if self.all_vectors is not None:
                dists = np.linalg.norm(self.all_vectors - region.center[np.newaxis, :], axis=1)
                nearby_idx = np.argsort(dists)[:5]
                nearby_ids = (
                    [self.all_ids[i] for i in nearby_idx]
                    if self.all_ids
                    else [str(i) for i in nearby_idx]
                )
            else:
                nearby_ids = []

            description = (
                f"Región de alta incertidumbre (energía={region.energy:.3f}). "
                f"Hay {region.n_points} puntos muestreados en esta zona. "
                f"Documentos cercanos: {', '.join(nearby_ids[:3])}."
            )

            suggestion = ExplorationSuggestion(
                region=region,
                description=description,
                suggested_queries=[
                    f"Explorar zona cerca de {nearby_ids[0] if nearby_ids else 'centro'}",
                    f"Buscar información complementaria en región de energía {region.energy:.2f}",
                ],
                potential_value=region.energy,
            )
            suggestions.append(suggestion)

        # Ordenar por valor potencial descendente
        suggestions.sort(key=lambda s: s.potential_value, reverse=True)
        return suggestions

    def find_knowledge_gaps(self, n_gaps: int = 5) -> List[EnergyRegion]:
        """
        Encuentra 'huecos' en el conocimiento del sistema.

        Los huecos son regiones entre clusters conocidos donde
        la energía es alta (poco representadas por splats).

        Returns:
            Lista de regiones con gaps de conocimiento.
        """
        return self.find_high_energy_regions(min_energy=0.5, n_regions=n_gaps, n_samples=1000)
