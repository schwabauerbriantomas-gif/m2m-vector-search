# M2M Edge-Cloud Cluster Architecture

## Problem Statement

M2M Vector Search actual está diseñado para **single-node**, lo que limita:

1. **Escalabilidad**: Máximo ~100K documentos por instancia
2. **Alta disponibilidad**: Punto único de fallo
3. **Balanceo de carga**: Sin distribución de queries
4. **Edge-Cloud hybrid**: No hay coordinación entre instancias

---

## Proposed Architecture: M2M Cluster

```
┌─────────────────────────────────────────────────────────────────────┐
│                         M2M CLUSTER                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                    COORDINATOR NODE                         │     │
│  │  (Lightweight, can run on cloud or powerful edge)           │     │
│  ├────────────────────────────────────────────────────────────┤     │
│  │  • ClusterRouter: Route queries to optimal edge nodes      │     │
│  │  • Aggregator: Merge results from multiple edges           │     │
│  │  • MetadataStore: Global index of which data is where      │     │
│  │  • HealthMonitor: Track edge node status                   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │
│  │  EDGE #1    │      │  EDGE #2    │      │  EDGE #3    │         │
│  ├─────────────┤      ├─────────────┤      ├─────────────┤         │
│  │ M2M Local   │      │ M2M Local   │      │ M2M Local   │         │
│  │ HRM2 Index  │      │ HRM2 Index  │      │ HRM2 Index  │         │
│  │ Vulkan/CPU  │      │ Vulkan/CPU  │      │ CPU only    │         │
│  │             │      │             │      │             │         │
│  │ Data: A-M   │      │ Data: N-Z   │      │ Data: 0-9   │         │
│  │ ~30K docs   │      │ ~25K docs   │      │ ~15K docs   │         │
│  └─────────────┘      └─────────────┘      └─────────────┘         │
│                                                                      │
│  EDGE NODES can operate INDEPENDENTLY (offline mode)                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. ClusterRouter (Coordinator)

```python
class ClusterRouter:
    """
    Routes queries to optimal edge nodes based on:
    - Data locality (which edge has relevant data)
    - Load balancing (distribute queries evenly)
    - Latency (prefer nearby edges)
    """
    
    def __init__(self):
        self.edge_nodes: Dict[str, EdgeNodeInfo] = {}
        self.metadata_index: Dict[str, Set[str]] = {}  # doc_id -> edge_ids
        self.load_metrics: Dict[str, LoadMetrics] = {}
    
    def route_query(self, query: np.ndarray, k: int) -> List[str]:
        """
        Determine which edge nodes should handle this query.
        
        Returns list of edge node IDs to query.
        """
        pass
    
    def register_edge(self, edge_id: str, metadata: EdgeNodeInfo):
        """Register a new edge node."""
        pass
    
    def heartbeat(self, edge_id: str, metrics: LoadMetrics):
        """Update edge node health and load."""
        pass
```

### 2. Aggregator (Coordinator)

```python
class ResultAggregator:
    """
    Merges results from multiple edge nodes.
    
    Strategies:
    - RRF (Reciprocal Rank Fusion)
    - Score-based merge
    - Distance-based merge
    """
    
    def merge_results(
        self,
        results: Dict[str, List[Tuple[int, float]]],  # edge_id -> results
        k: int,
        strategy: str = "rrf"
    ) -> List[Tuple[int, float]]:
        """
        Merge and re-rank results from multiple edges.
        """
        if strategy == "rrf":
            return self._rrf_merge(results, k)
        elif strategy == "score":
            return self._score_merge(results, k)
```

### 3. EdgeNode (Edge Layer)

```python
class EdgeNode:
    """
    M2M instance running on edge device.
    
    Features:
    - Local HRM2 index
    - Vulkan acceleration (if available)
    - Can operate offline
    - Syncs with coordinator when available
    """
    
    def __init__(self, edge_id: str, config: M2MConfig, coordinator_url: str = None):
        self.edge_id = edge_id
        self.local_store = SimpleVectorDB(config)
        self.coordinator_url = coordinator_url
        self.offline_queue: List[Query] = []  # Queue for offline mode
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Local search using HRM2."""
        return self.local_store.search(query, k)
    
    def sync_with_coordinator(self):
        """Sync metadata and health status with coordinator."""
        pass
    
    def ingest(self, documents: List[Document]):
        """Ingest documents locally and notify coordinator."""
        for doc in documents:
            self.local_store.add(doc.embedding)
        self._notify_coordinator_ingest(documents)
```

### 4. ClusterClient (Application Layer)

```python
class M2MClusterClient:
    """
    Client for applications to interact with M2M cluster.
    
    Handles:
    - Query routing
    - Result aggregation
    - Failover (if coordinator down, query edges directly)
    """
    
    def __init__(self, coordinator_url: str, fallback_edges: List[str] = None):
        self.coordinator = CoordinatorClient(coordinator_url)
        self.fallback_edges = fallback_edges or []
    
    def search(self, query: np.ndarray, k: int = 10) -> List[SearchResult]:
        """
        Distributed search across cluster.
        """
        try:
            # Normal: Use coordinator
            edge_ids = self.coordinator.route_query(query, k)
            results = {}
            for edge_id in edge_ids:
                results[edge_id] = self._query_edge(edge_id, query, k)
            return self.coordinator.aggregate(results, k)
        except CoordinatorUnavailable:
            # Fallback: Query all edges directly
            return self._fallback_search(query, k)
    
    def ingest(self, documents: List[Document], strategy: str = "shard"):
        """
        Ingest documents to cluster.
        
        Strategies:
        - shard: Distribute across edges
        - replicate: Copy to all edges
        - locality: Route based on data affinity
        """
        pass
```

---

## Data Distribution Strategies

### Strategy 1: Sharding by Hash

```python
def shard_by_hash(doc_id: str, num_edges: int) -> str:
    """Deterministic sharding by document ID hash."""
    hash_val = hash(doc_id)
    edge_index = hash_val % num_edges
    return f"edge-{edge_index}"
```

### Strategy 2: Sharding by Semantic Cluster

```python
def shard_by_cluster(embedding: np.ndarray, centroids: np.ndarray) -> str:
    """
    Route document to edge with nearest centroid.
    
    Advantage: Queries only need to hit relevant edges.
    """
    distances = np.linalg.norm(centroids - embedding, axis=1)
    nearest_edge = np.argmin(distances)
    return f"edge-{nearest_edge}"
```

### Strategy 3: Geo-Aware Sharding

```python
def shard_by_geo(doc_metadata: Dict, edge_locations: Dict[str, GeoLocation]) -> str:
    """
    Route document to nearest edge by geographic location.
    
    Advantage: Low latency for regional queries.
    """
    doc_location = doc_metadata.get("geo_location")
    if doc_location:
        nearest = min(edge_locations.items(), 
                      key=lambda x: geo_distance(doc_location, x[1]))
        return nearest[0]
    return "edge-default"
```

---

## Query Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    QUERY FLOW                                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. CLIENT sends query to Coordinator                        │
│     │                                                         │
│     ▼                                                         │
│  2. COORDINATOR determines which edges to query              │
│     │  - Based on metadata index                             │
│     │  - Load balancing                                      │
│     │  - Edge health                                         │
│     ▼                                                         │
│  3. COORDINATOR fans out queries to selected edges           │
│     │                                                         │
│     ├──────────────┬──────────────┐                          │
│     ▼              ▼              ▼                          │
│  Edge #1       Edge #2       Edge #3                         │
│  (parallel)    (parallel)    (parallel)                      │
│     │              │              │                          │
│     └──────────────┴──────────────┘                          │
│                    │                                          │
│                    ▼                                          │
│  4. COORDINATOR aggregates results                           │
│     │  - RRF merge                                           │
│     │  - Re-rank by distance                                 │
│     ▼                                                         │
│  5. CLIENT receives unified results                          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Failover Modes

### Mode 1: Coordinator Down

```python
# Client directly queries known edges
client = M2MClusterClient(
    coordinator_url="http://coordinator:8080",
    fallback_edges=["http://edge1:8080", "http://edge2:8080", "http://edge3:8080"]
)

# If coordinator unavailable, queries all fallback edges
results = client.search(query, k=10)  # Auto-fallback
```

### Mode 2: Edge Node Down

```python
# Coordinator excludes unhealthy edges from routing
# Results may be partial but system remains available
```

### Mode 3: Network Partition (Edge Offline)

```python
# Edge continues to operate locally
# Queues updates for sync when network restored
edge = EdgeNode(edge_id="edge-1", config=config)
edge.ingest(new_documents)  # Stored locally
edge.sync_with_coordinator()  # Syncs when available
```

---

## Implementation Phases

### Phase 1: Core Cluster Infrastructure (Week 1-2)

| Task | Files | Priority |
|------|-------|----------|
| ClusterRouter implementation | `cluster/router.py` | High |
| ResultAggregator with RRF | `cluster/aggregator.py` | High |
| EdgeNode wrapper | `cluster/edge_node.py` | High |
| Basic health monitoring | `cluster/health.py` | Medium |

### Phase 2: Communication Layer (Week 2-3)

| Task | Files | Priority |
|------|-------|----------|
| gRPC/HTTP API for coordinator | `cluster/api/` | High |
| Edge-coordinator protocol | `cluster/protocol.py` | High |
| Client SDK | `cluster/client.py` | Medium |

### Phase 3: Advanced Features (Week 3-4)

| Task | Files | Priority |
|------|-------|----------|
| Semantic sharding | `cluster/sharding.py` | Medium |
| Load balancing strategies | `cluster/balancer.py` | Medium |
| Offline sync queue | `cluster/sync.py` | Low |

### Phase 4: Operations (Week 4-5)

| Task | Files | Priority |
|------|-------|----------|
| Docker/K8s deployment | `deploy/` | Medium |
| Monitoring dashboards | `monitoring/` | Low |
| Documentation | `docs/cluster.md` | Medium |

---

## Performance Targets

| Metric | Single Node | Cluster (3 edges) | Cluster (10 edges) |
|--------|-------------|-------------------|-------------------|
| Max documents | 100K | 300K | 1M |
| Query latency (p50) | 50ms | 80ms | 120ms |
| Query latency (p99) | 150ms | 200ms | 300ms |
| Throughput (QPS) | 100 | 300 | 1000 |
| Availability | 95% | 99% | 99.9% |
| Failure recovery | N/A | 5s | 5s |

---

## Comparison: M2M Cluster vs Alternatives

| Feature | M2M Cluster | Pinecone | Milvus | Weaviate |
|---------|-------------|----------|--------|----------|
| **Edge-first** | ✅ Native | ❌ Cloud only | ❌ Server-based | ❌ Server-based |
| **Offline capable** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **GPU acceleration** | ✅ Vulkan | ✅ Proprietary | ✅ CUDA | ❌ Limited |
| **Open source** | ✅ AGPL | ❌ No | ✅ Apache | ✅ BSD |
| **Setup complexity** | Medium | Low | High | Medium |
| **Cost at scale** | Low (commodity hardware) | High | Medium | Medium |

---

## Key Advantages of M2M Cluster

### 1. Edge-Native Design
- Other vector DBs assume reliable network
- M2M Cluster designed for intermittent connectivity
- Each edge is fully functional standalone

### 2. Low Hardware Requirements
- Edges can be cheap devices (Raspberry Pi class)
- No need for expensive GPU servers
- Vulkan works on any GPU (AMD, Intel, NVIDIA)

### 3. Privacy-First
- Data stays on edge by default
- Coordinator only sees metadata, not embeddings
- Suitable for sensitive data (healthcare, finance)

### 4. Zero Vendor Lock-in
- No proprietary cloud service
- Can run on any infrastructure
- Full control over data and compute

---

## File Structure

```
m2m-vector-search/
├── src/m2m/
│   ├── cluster/
│   │   ├── __init__.py
│   │   ├── router.py          # ClusterRouter
│   │   ├── aggregator.py      # ResultAggregator
│   │   ├── edge_node.py       # EdgeNode
│   │   ├── client.py          # M2MClusterClient
│   │   ├── sharding.py        # Distribution strategies
│   │   ├── health.py          # Health monitoring
│   │   └── protocol.py        # Communication protocol
│   ├── api/
│   │   ├── coordinator_api.py # REST/gRPC for coordinator
│   │   └── edge_api.py        # REST/gRPC for edges
│   └── ... (existing files)
├── deploy/
│   ├── docker-compose.yml     # Local cluster setup
│   ├── k8s/                   # Kubernetes manifests
│   └── terraform/             # Cloud deployment
└── tests/
    ├── test_cluster.py
    ├── test_sharding.py
    └── test_failover.py
```

---

## Next Steps

1. **Validate architecture**: Review this proposal
2. **MVP implementation**: Router + Aggregator + EdgeNode
3. **Local testing**: 3-node cluster on single machine
4. **Deployment guide**: Docker Compose for easy setup
5. **Benchmark**: Compare with single-node M2M

---

**Conclusion**: M2M Cluster addresses the fundamental scalability limitation while preserving M2M's core advantages: edge-first, offline-capable, privacy-preserving, and open source.
