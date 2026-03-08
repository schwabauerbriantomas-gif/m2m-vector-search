import time
from typing import Dict, List, Set

import numpy as np

from .balancer import LoadBalancer
from .health import EdgeNodeInfo, LoadMetrics


class ClusterRouter:
    """
    Routes queries to optimal edge nodes based on:
    - Data locality (which edge has relevant data)
    - Load balancing (distribute queries evenly)
    - Latency (prefer nearby edges if location available)
    """

    def __init__(self):
        self.edge_nodes: Dict[str, EdgeNodeInfo] = {}
        self.metadata_index: Dict[str, Set[str]] = {}  # doc_id -> edge_ids
        # Maintain health metrics to avoid degraded nodes
        self.load_metrics: Dict[str, LoadMetrics] = {}
        # Keep track of semantic sharding boundaries if applicable
        self.centroids: Dict[str, np.ndarray] = {}
        self.balancer = LoadBalancer()

    def register_edge(self, edge_id: str, url: str) -> EdgeNodeInfo:
        """Register a new edge node."""
        info = EdgeNodeInfo(
            edge_id=edge_id, url=url, status="online", last_heartbeat=time.time()
        )
        self.edge_nodes[edge_id] = info
        self.load_metrics[edge_id] = LoadMetrics()
        return info

    def remove_edge(self, edge_id: str):
        """Remove a disconnected edge node."""
        if edge_id in self.edge_nodes:
            del self.edge_nodes[edge_id]
        if edge_id in self.load_metrics:
            del self.load_metrics[edge_id]

        # Clean up metadata index mapping
        for doc_id, edges in list(self.metadata_index.items()):
            if edge_id in edges:
                edges.remove(edge_id)
                if not edges:
                    del self.metadata_index[doc_id]

    def heartbeat(self, edge_id: str, metrics: LoadMetrics):
        """Update edge node health and load."""
        if edge_id in self.edge_nodes:
            self.edge_nodes[edge_id].last_heartbeat = time.time()
            self.edge_nodes[edge_id].status = "online"
            self.load_metrics[edge_id] = metrics

    def get_online_edges(self) -> List[str]:
        """Return list of edge_ids that are responsive and not overloaded."""
        current_time = time.time()
        online = []
        for edge_id, node in self.edge_nodes.items():
            if node.status == "online":
                # Check for dead nodes that haven't heartbeated in 30s
                if current_time - node.last_heartbeat < 30.0:
                    online.append(edge_id)
                else:
                    node.status = "offline"
        return online

    def route_query(
        self, query: np.ndarray, k: int, strategy: str = "broadcast"
    ) -> List[str]:
        """
        Determine which edge nodes should handle this query.
        Returns list of optimal edge node IDs to query.
        """
        online_edges = self.get_online_edges()
        if not online_edges:
            return []

        # Use balancer for routing
        return self.balancer.select_best_edges(
            online_edges, self.load_metrics, strategy
        )

    def register_document(self, doc_id: str, edge_id: str):
        """Record what document lives where for exact lookup"""
        if doc_id not in self.metadata_index:
            self.metadata_index[doc_id] = set()
        self.metadata_index[doc_id].add(edge_id)
