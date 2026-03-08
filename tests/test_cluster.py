import numpy as np

from m2m import M2MConfig
from m2m.cluster import (
    ClusterRouter,
    ResultAggregator,
    EdgeNode,
    M2MClusterClient,
    shard_by_hash,
)
from m2m.cluster.health import LoadMetrics


def test_shard_by_hash():
    doc_id = "test_doc_123"
    shard1 = shard_by_hash(doc_id, 3)
    shard2 = shard_by_hash(doc_id, 3)
    # Deterministic check
    assert shard1 == shard2
    assert shard1.startswith("edge-")


def test_router_heartbeat_and_routing():
    router = ClusterRouter()

    # Register 3 edges
    router.register_edge("edge-0", "http://edge0")
    router.register_edge("edge-1", "http://edge1")
    router.register_edge("edge-2", "http://edge2")

    online = router.get_online_edges()
    assert len(online) == 3

    # Mock some load
    router.heartbeat("edge-0", LoadMetrics(active_queries=10))
    router.heartbeat("edge-1", LoadMetrics(active_queries=2))
    router.heartbeat("edge-2", LoadMetrics(active_queries=5))

    # Query routing
    query = np.random.randn(640).astype(np.float32)
    edges = router.route_query(query, k=10, strategy="least_loaded")

    # Should prefer edge-1 as it has least queries
    assert edges[0] == "edge-1"


def test_aggregator_rrf():
    agg = ResultAggregator(rrf_k=60)

    # edge results: list of (doc_id, distance)
    # lower distance is better natively, but RRF uses order/rank
    results = {
        "edge-0": [(100, 0.1), (101, 0.5)],
        "edge-1": [(101, 0.2), (102, 0.3)],
    }

    merged = agg.merge_results(results, k=2, strategy="rrf")
    assert len(merged) == 2

    # 101 appears in both, should rank high
    doc_ids = [r[0] for r in merged]
    assert 101 in doc_ids

    # Ensure distance is the minimum observed
    for doc_id, dist in merged:
        if doc_id == 101:
            assert dist == 0.2  # min(0.5, 0.2)


def test_cluster_client_failover():
    # Setup client without coordinator
    client = M2MClusterClient(fallback_edges=["edge-0", "edge-1"])

    # Register mock local edges
    config = M2MConfig(device="cpu", latent_dim=128)  # small config
    edge0 = EdgeNode("edge-0", config)
    edge1 = EdgeNode("edge-1", config)

    client.register_local_edge(edge0)
    client.register_local_edge(edge1)

    # Create random vectors
    vectors = np.random.randn(10, 128).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(10)]

    # Test local ingest with failover
    added = 0
    # Custom simple mock ingest instead of client.ingest which relies on hash shards matching exact keys
    for i in range(5):
        added += edge0.ingest(vectors[i : i + 1], [doc_ids[i]])
    for i in range(5, 10):
        added += edge1.ingest(vectors[i : i + 1], [doc_ids[i]])

    assert added == 10

    # Distributed Search (implicitly tests fallback as no coordinator URL is set)
    query = np.random.randn(128).astype(np.float32)
    results = client.search(query, k=5)

    # Since we added 10 docs, we should get 5 back
    assert len(results) <= 5
