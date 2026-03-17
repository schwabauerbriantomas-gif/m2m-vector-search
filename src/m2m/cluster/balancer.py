from typing import Dict, List

from m2m.cluster.health import LoadMetrics


class LoadBalancer:
    """
    Handles distribution strategies for M2M Cluster queries.
    """

    def __init__(self):
        self._current_rr_index = 0

    def route_round_robin(self, online_edges: List[str]) -> str:
        """
        Returns a single edge sequentially.
        """
        if not online_edges:
            raise ValueError("No online edges available for routing.")

        selected = online_edges[self._current_rr_index % len(online_edges)]
        self._current_rr_index = (self._current_rr_index + 1) % len(online_edges)
        return selected

    def route_least_loaded(
        self, online_edges: List[str], load_metrics: Dict[str, LoadMetrics]
    ) -> List[str]:
        """
        Returns edges sorted by the least active queries and lowest latency.
        """
        if not online_edges:
            return []

        def _score_edge(eid: str) -> float:
            # We want to minimize score. Active queries factor heavily, latency is a tie-breaker
            metrics = load_metrics.get(eid, LoadMetrics())
            return (metrics.active_queries * 100.0) + metrics.query_latency_ms

        return sorted(online_edges, key=_score_edge)

    def select_best_edges(
        self,
        online_edges: List[str],
        load_metrics: Dict[str, LoadMetrics],
        strategy: str = "least_loaded",
    ) -> List[str]:
        """
        Main entrypoint for router to map load balancing strategy names.
        """
        if strategy == "broadcast":
            return online_edges

        if strategy == "round_robin":
            if not online_edges:
                return []
            return [self.route_round_robin(online_edges)]

        if strategy == "least_loaded":
            return self.route_least_loaded(online_edges, load_metrics)

        # Default back to broadcast if unknown
        return online_edges
