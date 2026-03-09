"""
EnergyRouter - Router Energético para M2M Cluster.

OPCIONAL: Si no se habilita, el sistema funciona normalmente
sin distribución de carga basada en energía.

Estrategias disponibles:
  energy_balanced  - Menor energía = mayor probabilidad de selección
  round_robin      - Distribución secuencial uniforme
  least_loaded     - Nodo con menor carga activa
  locality_aware   - Prioriza nodos con datos relevantes
  hybrid           - Combina energía + carga + localidad + latencia
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .balancer import LoadBalancer
from .health import EdgeNodeInfo, LoadMetrics


@dataclass
class NodeEnergy:
    """Estado energético de un nodo del cluster."""

    node_id: str
    last_energy: float = 1.0
    updated_at: float = field(default_factory=time.time)
    weight: float = 1.0


class EnergyRouter:
    """
    Router Energético para M2M Cluster.

    Distribuye queries entre múltiples nodos basándose en el
    paisaje energético de cada uno. Es OPCIONAL: si se
    deshabilita o hay menos de 2 nodos, usa round_robin simple.

    Uso:
        router = EnergyRouter({
            'enabled': True,
            'strategy': 'energy_balanced',
            'cache_energy': True,
            'cache_ttl_seconds': 60,
            'min_nodes': 2,
        })
        node = router.route(query_vector, online_nodes)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Configuración del router. Keys:
              - enabled: bool (default False)
              - strategy: str (default 'energy_balanced')
              - fallback_strategy: str (default 'round_robin')
              - cache_energy: bool (default True)
              - cache_ttl_seconds: int (default 60)
              - min_nodes: int (default 2)
        """
        config = config or {}
        self.enabled: bool = config.get("enabled", False)
        self.strategy: str = config.get("strategy", "energy_balanced")
        self.fallback_strategy: str = config.get("fallback_strategy", "round_robin")
        self.cache_energy: bool = config.get("cache_energy", True)
        self.cache_ttl: float = float(config.get("cache_ttl_seconds", 60))
        self.min_nodes: int = config.get("min_nodes", 2)

        # Cache de energía por (query_hash, node_id) -> energy
        self._energy_cache: Dict[str, float] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Estado de nodos
        self._node_energies: Dict[str, NodeEnergy] = {}
        self._rr_index: int = 0

        # Métricas de routing
        self._route_counts: Dict[str, int] = {}
        self._total_routes: int = 0

        # Balancer interno (para modos non-energy)
        self._balancer = LoadBalancer()

        # Determinar si usar routing simple
        self._use_simple = not self.enabled

    def register_node(self, node_id: str, weight: float = 1.0):
        """Registra un nodo en el router."""
        self._node_energies[node_id] = NodeEnergy(
            node_id=node_id, weight=weight
        )
        self._route_counts[node_id] = 0

    def remove_node(self, node_id: str):
        """Elimina un nodo del router."""
        self._node_energies.pop(node_id, None)
        self._route_counts.pop(node_id, None)

    def update_node_energy(self, node_id: str, energy: float):
        """Actualiza el estado energético de un nodo."""
        if node_id in self._node_energies:
            self._node_energies[node_id].last_energy = energy
            self._node_energies[node_id].updated_at = time.time()

    def route(
        self,
        query: np.ndarray,
        online_nodes: List[str],
        load_metrics: Optional[Dict[str, "LoadMetrics"]] = None,
        operation: str = "search",
    ) -> List[str]:
        """
        Determina el/los nodo/s más apropiados para la query.

        Args:
            query: Vector de consulta
            online_nodes: Lista de nodos online disponibles
            load_metrics: Métricas de carga por nodo
            operation: Tipo de operación ('search', 'write')

        Returns:
            Lista de node_ids ordenados por prioridad de routing.
        """
        if not online_nodes:
            return []

        # Router deshabilitado o insuficientes nodos
        if self._use_simple or len(online_nodes) < self.min_nodes:
            return self._simple_route(online_nodes)

        if self.strategy == "energy_balanced":
            return self._energy_route(query, online_nodes)
        elif self.strategy == "round_robin":
            return self._round_robin_route(online_nodes)
        elif self.strategy == "least_loaded":
            return self._least_loaded_route(online_nodes, load_metrics)
        elif self.strategy == "locality_aware":
            return self._locality_route(query, online_nodes)
        elif self.strategy == "hybrid":
            return self._hybrid_route(query, online_nodes, load_metrics)
        else:
            return self._simple_route(online_nodes)

    def _simple_route(self, nodes: List[str]) -> List[str]:
        """Routing simple: retorna todos los nodos disponibles."""
        return list(nodes)

    def _round_robin_route(self, nodes: List[str]) -> List[str]:
        """Routing round-robin: distribución secuencial uniforme."""
        if not nodes:
            return []
        idx = self._rr_index % len(nodes)
        self._rr_index += 1
        selected = nodes[idx]
        self._record_route(selected)
        return [selected]

    def _energy_route(self, query: np.ndarray, nodes: List[str]) -> List[str]:
        """
        Routing basado en energía del paisaje de cada nodo.

        Lógica:
        - Cada nodo tiene un estado energético local
        - La query se envía al nodo con MENOR energía
        - Menor energía = mayor confianza = mejor resultado potencial
        - Componente probabilística para balanceo
        """
        node_energies = []
        for node_id in nodes:
            # Obtener energía del nodo (con cache)
            energy = self._get_cached_energy(query, node_id)
            ne = self._node_energies.get(node_id)
            weight = ne.weight if ne else 1.0
            weighted_energy = energy * weight
            node_energies.append((node_id, weighted_energy))

        # Convertir energía a probabilidad (menor energía = mayor prob)
        energies = [e for _, e in node_energies]
        max_energy = max(energies) if energies else 1.0
        probs = np.array([(max_energy - e + 0.1) for e in energies])
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(nodes)) / len(nodes)

        # Selección probabilística
        selected_idx = np.random.choice(len(nodes), p=probs)
        selected = nodes[selected_idx]
        self._record_route(selected)
        return [selected]

    def _least_loaded_route(
        self, nodes: List[str], load_metrics: Optional[Dict[str, "LoadMetrics"]]
    ) -> List[str]:
        """Routing al nodo con menor carga."""
        if not load_metrics:
            return self._round_robin_route(nodes)

        loads = []
        for node_id in nodes:
            m = load_metrics.get(node_id)
            load = m.active_queries if m else 0
            loads.append((node_id, load))

        loads.sort(key=lambda x: x[1])
        selected = loads[0][0]
        self._record_route(selected)
        return [selected]

    def _locality_route(self, query: np.ndarray, nodes: List[str]) -> List[str]:
        """Routing por localidad: prioriza nodos con datos relevantes."""
        # Verificar qué nodos tienen datos cerca de la query
        locality_scores = []
        for node_id in nodes:
            ne = self._node_energies.get(node_id)
            # Menor energía = mayor localidad (el nodo conoce bien esta región)
            energy = ne.last_energy if ne else 1.0
            locality_scores.append((node_id, 1.0 - min(energy, 1.0)))

        locality_scores.sort(key=lambda x: x[1], reverse=True)
        selected = locality_scores[0][0]
        self._record_route(selected)
        return [selected]

    def _hybrid_route(
        self,
        query: np.ndarray,
        nodes: List[str],
        load_metrics: Optional[Dict[str, "LoadMetrics"]],
    ) -> List[str]:
        """
        Routing híbrido: combina energía + carga + localidad.

        Score = 0.4*energía + 0.2*carga + 0.3*localidad + 0.1*latencia
        Menor score = mejor nodo.
        """
        scores = []
        for node_id in nodes:
            ne = self._node_energies.get(node_id)

            # Energía (menor es mejor)
            energy_score = ne.last_energy if ne else 1.0

            # Carga (menor es mejor)
            m = (load_metrics or {}).get(node_id)
            load_score = min(m.active_queries / 100.0, 1.0) if m else 0.0

            # Localidad (menor energía = mayor localidad local → menor score)
            locality_score = energy_score

            # Latencia proxy (0 por defecto, se actualizaría con pings reales)
            latency_score = 0.0

            combined = (
                0.4 * energy_score
                + 0.2 * load_score
                + 0.3 * locality_score
                + 0.1 * latency_score
            )
            scores.append((node_id, combined))

        scores.sort(key=lambda x: x[1])
        selected = scores[0][0]
        self._record_route(selected)
        return [selected]

    def _get_cached_energy(self, query: np.ndarray, node_id: str) -> float:
        """Obtiene la energía de un nodo (con cache por TTL)."""
        if not self.cache_energy:
            ne = self._node_energies.get(node_id)
            return ne.last_energy if ne else 1.0

        cache_key = f"{hash(query.tobytes())}_{node_id}"
        now = time.time()

        if cache_key in self._energy_cache:
            if now - self._cache_timestamps.get(cache_key, 0) < self.cache_ttl:
                return self._energy_cache[cache_key]

        # Cache miss: usar última energía conocida del nodo
        ne = self._node_energies.get(node_id)
        energy = ne.last_energy if ne else 1.0
        self._energy_cache[cache_key] = energy
        self._cache_timestamps[cache_key] = now
        return energy

    def _record_route(self, node_id: str):
        """Registra una decisión de routing para estadísticas."""
        self._route_counts[node_id] = self._route_counts.get(node_id, 0) + 1
        self._total_routes += 1

    def get_routing_stats(self) -> Dict[str, Any]:
        """Estadísticas de routing para monitoreo."""
        return {
            "enabled": self.enabled,
            "strategy": self.strategy,
            "nodes_count": len(self._node_energies),
            "cache_size": len(self._energy_cache),
            "total_routes": self._total_routes,
            "node_distribution": dict(self._route_counts),
            "node_energies": {
                nid: ne.last_energy for nid, ne in self._node_energies.items()
            },
        }

    def clear_cache(self):
        """Limpia el cache de energía."""
        self._energy_cache.clear()
        self._cache_timestamps.clear()


class ClusterRouter:
    """
    Router de Cluster M2M con soporte de routing energético.

    Envuelve al EnergyRouter y al balancer de carga existente,
    exponiendo una interfaz unificada de routing.
    """

    def __init__(
        self,
        energy_router_config: Optional[Dict[str, Any]] = None,
    ):
        self.edge_nodes: Dict[str, EdgeNodeInfo] = {}
        self.metadata_index: Dict[str, Set[str]] = {}
        self.load_metrics: Dict[str, LoadMetrics] = {}
        self.centroids: Dict[str, np.ndarray] = {}
        self.balancer = LoadBalancer()

        # EnergyRouter opcional
        self.energy_router = EnergyRouter(energy_router_config or {"enabled": False})

    def register_edge(self, edge_id: str, url: str, weight: float = 1.0) -> EdgeNodeInfo:
        """Registra un edge node."""
        info = EdgeNodeInfo(
            edge_id=edge_id, url=url, status="online", last_heartbeat=time.time()
        )
        self.edge_nodes[edge_id] = info
        self.load_metrics[edge_id] = LoadMetrics()
        self.energy_router.register_node(edge_id, weight=weight)
        return info

    def remove_edge(self, edge_id: str):
        """Elimina un edge node."""
        if edge_id in self.edge_nodes:
            del self.edge_nodes[edge_id]
        if edge_id in self.load_metrics:
            del self.load_metrics[edge_id]
        self.energy_router.remove_node(edge_id)

        for doc_id, edges in list(self.metadata_index.items()):
            if edge_id in edges:
                edges.remove(edge_id)
                if not edges:
                    del self.metadata_index[doc_id]

    def heartbeat(self, edge_id: str, metrics: LoadMetrics):
        """Actualiza métricas de un edge node."""
        if edge_id in self.edge_nodes:
            self.edge_nodes[edge_id].last_heartbeat = time.time()
            self.edge_nodes[edge_id].status = "online"
            self.load_metrics[edge_id] = metrics

    def get_online_edges(self) -> List[str]:
        """Retorna edge_ids de nodos online y responsivos."""
        current_time = time.time()
        online = []
        for edge_id, node in self.edge_nodes.items():
            if node.status == "online":
                if current_time - node.last_heartbeat < 30.0:
                    online.append(edge_id)
                else:
                    node.status = "offline"
        return online

    def route_query(
        self,
        query: np.ndarray,
        k: int,
        strategy: str = "broadcast",
    ) -> List[str]:
        """
        Determina qué edge nodes deben manejar esta query.

        Si el EnergyRouter está habilitado, usa routing energético.
        Si no, usa la estrategia legacy.
        """
        online_edges = self.get_online_edges()
        if not online_edges:
            return []

        if self.energy_router.enabled:
            return self.energy_router.route(
                query, online_edges, self.load_metrics, operation="search"
            )

        # Fallback al balancer legacy
        return self.balancer.select_best_edges(
            online_edges, self.load_metrics, strategy
        )

    def register_document(self, doc_id: str, edge_id: str):
        """Registra dónde vive un documento."""
        if doc_id not in self.metadata_index:
            self.metadata_index[doc_id] = set()
        self.metadata_index[doc_id].add(edge_id)

    def get_routing_stats(self) -> Dict:
        """Estadísticas de routing."""
        return {
            "online_nodes": len(self.get_online_edges()),
            "total_nodes": len(self.edge_nodes),
            "energy_router": self.energy_router.get_routing_stats(),
        }
