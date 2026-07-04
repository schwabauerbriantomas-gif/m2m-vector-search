import time
from typing import List, Optional, Tuple

import numpy as np

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

    def __init__(self, edge_id: str, config: M2MConfig, coordinator_url: Optional[str] = None):
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
            # Call search without metadata to get legacy tuple (mu, alpha, kappa)
            result = self.local_store.search(query, k)

            # Handle both legacy tuple format and new DocResult list
            if isinstance(result, tuple):
                # find_neighbors returns (mu, alpha, kappa, splat_indices)
                # local_store.search may return 3 or 4 element tuples
                if len(result) == 4:
                    neighbors_mu, neighbors_alpha, neighbors_kappa, splat_indices = result
                else:
                    neighbors_mu, neighbors_alpha, neighbors_kappa = result
                    splat_indices = np.arange(len(neighbors_mu))
                # splat_indices may be multi-dimensional; flatten to 1D for indexing
                splat_flat = np.asarray(splat_indices).flatten()
                results = []
                for i in range(len(splat_flat)):
                    distance = float(i)  # Placeholder
                    doc_index = int(splat_flat[i])
                    results.append((doc_index, distance))
            else:
                # DocResult list format
                results = [(r.id, r.score) for r in result]

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

        return LoadMetrics(active_queries=self.active_queries, query_latency_ms=avg_latency)

    def sync_with_coordinator(self):
        """Sync metadata and health status with coordinator."""
        if not self.coordinator_url:
            return

        try:
            import requests

            metrics = self.get_metrics()
            requests.post(
                f"{self.coordinator_url}/heartbeat",
                json={
                    "edge_id": self.edge_id,
                    "vector_count": metrics.vector_count,
                    "cpu_load": metrics.cpu_load,
                    "memory_mb": metrics.memory_mb,
                },
                timeout=5,
            )
        except Exception:
            # Coordinator unreachable — non-fatal, will retry on next sync
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
