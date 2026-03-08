import pytest
import numpy as np
import asyncio

from m2m.cluster.health import GeoLocation, LoadMetrics
from m2m.cluster.sharding import shard_by_cluster, shard_by_geo
from m2m.cluster.balancer import LoadBalancer
from m2m.cluster.sync import SyncQueue


def test_semantic_sharding():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    edge_ids = ["edge-0", "edge-1"]

    # Should map to edge-0
    e1 = shard_by_cluster(np.array([1.0, 1.0]), centroids, edge_ids)
    assert e1 == "edge-0"

    # Should map to edge-1
    e2 = shard_by_cluster(np.array([9.0, 9.0]), centroids, edge_ids)
    assert e2 == "edge-1"


def test_geo_sharding():
    # NY vs LA
    edge_locations = {
        "edge-ny": GeoLocation(latitude=40.7128, longitude=-74.0060),
        "edge-la": GeoLocation(latitude=34.0522, longitude=-118.2437),
    }

    # Boston (closer to NY)
    doc_meta_boston = {"geo_location": {"lat": 42.3601, "lon": -71.0589}}
    res1 = shard_by_geo(doc_meta_boston, edge_locations)
    assert res1 == "edge-ny"

    # SF (closer to LA)
    doc_meta_sf = {"geo_location": {"lat": 37.7749, "lon": -122.4194}}
    res2 = shard_by_geo(doc_meta_sf, edge_locations)
    assert res2 == "edge-la"


def test_load_balancer():
    balancer = LoadBalancer()
    edges = ["edge-1", "edge-2", "edge-3"]
    metrics = {
        "edge-1": LoadMetrics(active_queries=10, query_latency_ms=50.0),
        "edge-2": LoadMetrics(active_queries=2, query_latency_ms=20.0),  # Best
        "edge-3": LoadMetrics(active_queries=5, query_latency_ms=30.0),
    }

    # Least Loaded
    res = balancer.select_best_edges(edges, metrics, strategy="least_loaded")
    assert res[0] == "edge-2"
    assert res[1] == "edge-3"
    assert res[2] == "edge-1"

    # Round Robin
    rr1 = balancer.select_best_edges(edges, metrics, strategy="round_robin")
    assert rr1 == ["edge-1"]
    rr2 = balancer.select_best_edges(edges, metrics, strategy="round_robin")
    assert rr2 == ["edge-2"]


@pytest.mark.asyncio
async def test_offline_sync_queue():
    queue = SyncQueue(flush_interval_seconds=0.1)

    # Add actions
    queue.add_action({"action": "register", "doc_id": "doc1"})
    queue.add_action({"action": "register", "doc_id": "doc2"})

    assert len(queue.get_pending()) == 2

    call_count = 0

    async def mock_dispatch(actions) -> bool:
        nonlocal call_count
        call_count += 1
        # Pretend it fails the first time
        if call_count == 1:
            return False
        return True

    await queue.start_background_sync(mock_dispatch)

    # First flush will fail, second flush should succeed
    await asyncio.sleep(0.3)
    await queue.stop()

    # Should have cleared queue on the second attempt
    assert len(queue.get_pending()) == 0
    assert call_count >= 2
