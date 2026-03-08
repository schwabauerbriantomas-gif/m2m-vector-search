from typing import List, Tuple, Optional
import numpy as np
import time

from .. import M2MConfig, SimpleVectorDB
from .health import LoadMetrics
from .sync import SyncQueue


class EdgeNode:
    """
    M2M instance running on edge device.

    Features:
    - Local HRM2 index via SimpleVectorDB
    - Can operate offline
    - Sycs with coordinator when available
    """

    def __init__(
        self, edge_id: str, config: M2MConfig, coordinator_url: Optional[str] = None
    ):
        self.edge_id = edge_id
        self.local_store = SimpleVectorDB(config.device, latent_dim=config.latent_dim)
        self.coordinator_url = coordinator_url
        self.sync_queue = SyncQueue(flush_interval_seconds=5.0)

        # Metrics tracking
        self.active_queries = 0
        self.total_queries = 0
        self.total_search_time = 0.0

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Local search using HRM2."""
        self.active_queries += 1
        start_time = time.time()
        try:
            # SimpleVectorDB returns (indices, alpha, kappa) or similar.
            # We want (doc_id, distance/score). Let's adapt based on standard return
            neighbors_mu, neighbors_alpha, neighbors_kappa = self.local_store.search(
                query, k
            )

            # Simple wrapper to parse results into distance format
            # In real system, document indices would map to actual doc IDs.
            # Here we fake doc IDs based on positional index and use kappa/alpha to derive a score
            results = []
            for i in range(len(neighbors_mu)):
                # Fake score logic: lower is better, usually.
                # Assuming neighbors_mu already sorted by similarity in engine
                distance = float(
                    i
                )  # Placeholder for actual distance metrics from engine
                doc_index = int(i)  # Placeholder for actual doc ID lookup
                results.append((doc_index, distance))

            return results
        finally:
            self.active_queries -= 1
            self.total_queries += 1
            self.total_search_time += time.time() - start_time

    def get_metrics(self) -> LoadMetrics:
        """Return current performance metrics."""
        avg_latency = 0.0
        if self.total_queries > 0:
            avg_latency = (self.total_search_time / self.total_queries) * 1000

        return LoadMetrics(
            active_queries=self.active_queries, query_latency_ms=avg_latency
        )

    def sync_with_coordinator(self):
        """Sync metadata and health status with coordinator."""
        if not self.coordinator_url:
            return

        # In a real implementation this would make a network call to the coordinator
        # e.g., requests.post(f"{self.coordinator_url}/heartbeat", json=self.get_metrics())
        pass

    def ingest(self, vectors: np.ndarray, doc_ids: List[str] = None):
        """Ingest documents locally and queue notification to coordinator."""
        added = self.local_store.add(vectors)

        if self.coordinator_url and doc_ids:
            # Notify coordinator that we own these documents
            for doc_id in doc_ids:
                self.sync_queue.add_action(
                    {"action": "register", "doc_id": doc_id, "edge_id": self.edge_id}
                )
            self.sync_with_coordinator()
        return added
