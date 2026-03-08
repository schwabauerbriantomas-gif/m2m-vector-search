import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
import requests

from .router import ClusterRouter
from .aggregator import ResultAggregator
from .sharding import shard_by_hash
from .edge_node import EdgeNode
from .. import M2MConfig

class CoordinatorUnavailable(Exception):
    pass

class CoordinatorClient:
    """Wrapper to handle network requests to coordinate router."""
    def __init__(self, url: str):
         self.url = url
         self.session = requests.Session()
         
    def route_query(self, query: np.ndarray, k: int) -> List[str]:
        # For a real implementation:
        # response = self.session.post(f"{self.url}/route", json={"query": query.tolist(), "k": k})
        # return response.json()["edge_ids"]
        raise NotImplementedError("Network API not yet implemented")

class M2MClusterClient:
    """
    Client for applications to interact with M2M cluster.
    
    Handles:
    - Query routing
    - Result aggregation
    - Failover (if coordinator down, query edges directly)
    """
    
    def __init__(self, coordinator_url: Optional[str] = None, fallback_edges: List[str] = None, in_memory_router: Optional[ClusterRouter] = None):
        self.coordinator_url = coordinator_url
        self.fallback_edges = fallback_edges or []
        self.in_memory_router = in_memory_router
        self.aggregator = ResultAggregator(rrf_k=60)
        
        # Local mock references for Phase 1 testing
        self.local_edges: Dict[str, EdgeNode] = {}
        
    def register_local_edge(self, edge: EdgeNode):
        """For testing: directly register a local EdgeNode instance."""
        self.local_edges[edge.edge_id] = edge
        if self.in_memory_router:
            self.in_memory_router.register_edge(edge.edge_id, f"http://localhost/{edge.edge_id}")

    def _query_edge(self, edge_id: str, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Mock network call to Edge Node."""
        if edge_id in self.local_edges:
            return self.local_edges[edge_id].search(query, k)
            
        # Real implementation:
        # edge_url = get_edge_url(edge_id)
        # response = requests.post(f"{edge_url}/search", json={"query": query.tolist(), "k": k})
        # return response.json()["results"]
        return []
        
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Distributed search across cluster.
        """
        try:
            # Phase 1 logic: uses in-memory router if provided, else tries coordinator API
            if self.in_memory_router:
                 edge_ids = self.in_memory_router.route_query(query, k)
            elif self.coordinator_url:
                 # Real network routing
                 coordinator = CoordinatorClient(self.coordinator_url)
                 edge_ids = coordinator.route_query(query, k)
            else:
                 raise CoordinatorUnavailable("No coordinator or router provided")
                 
            results = {}
            for edge_id in edge_ids:
                edge_results = self._query_edge(edge_id, query, k)
                if edge_results:
                    results[edge_id] = edge_results
                    
            if not results:
                return []
                
            return self.aggregator.merge_results(results, k, strategy="rrf")
            
        except (CoordinatorUnavailable, requests.exceptions.RequestException):
            # Fallback: Query all known fallback edges directly
            print("Coordinator unavailable. Falling back to direct edge queries.")
            return self._fallback_search(query, k)

    def _fallback_search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Query directly known edges without coordinator."""
        results = {}
        # Try local edges first if present
        target_edges = self.fallback_edges if self.fallback_edges else list(self.local_edges.keys())
        
        for edge in target_edges:
            try:
                edge_results = self._query_edge(edge, query, k)
                if edge_results:
                     results[edge] = edge_results
            except Exception as e:
                print(f"Failed to query fallback edge {edge}: {e}")
                
        if not results:
             return []
             
        # Merge whatever partial results we got
        return self.aggregator.merge_results(results, k, strategy="rrf")
        
    def ingest(self, vectors: np.ndarray, doc_ids: List[str] = None, strategy: str = "shard"):
        """
        Ingest documents to cluster.
        In Phase 1 testing, uses the local edges and in-memory router.
        """
        added_count = 0
        if strategy == "shard":
             # Distribute across edges evenly (simulated hash sharding)
             edge_list = list(self.local_edges.keys())
             if not edge_list:
                 return 0
                 
             if doc_ids is None:
                 doc_ids = [f"doc_{i}" for i in range(len(vectors))]
                 
             for i, (vector, doc_id) in enumerate(zip(vectors, doc_ids)):
                 target_edge = shard_by_hash(doc_id, len(edge_list))
                 # Hash returns exactly "edge-X". Re-map to actual edge IDs
                 edge_idx = int(target_edge.split("-")[1])
                 mapped_edge_id = edge_list[edge_idx]
                 
                 # Direct local index update for Phase 1 MVP
                 # Reshape vector for ingestion (N, D)
                 vec_reshaped = vector.reshape(1, -1)
                 self.local_edges[mapped_edge_id].ingest(vec_reshaped, [doc_id])
                 
                 if self.in_memory_router:
                     self.in_memory_router.register_document(doc_id, mapped_edge_id)
                     
                 added_count += 1
        else:
             print(f"Ingest strategy '{strategy}' not fully supported in Phase 1.")
             
        return added_count
