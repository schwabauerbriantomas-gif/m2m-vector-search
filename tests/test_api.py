import numpy as np
from fastapi.testclient import TestClient

from m2m.api.coordinator_api import app as coordinator_app
from m2m.api.edge_api import app as edge_app

coord_client = TestClient(coordinator_app)
edge_client = TestClient(edge_app)


def test_edge_health():
    response = edge_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_edge_ingest_and_search():
    # Ingest
    v = np.random.randn(5, 128).astype(np.float32).tolist()
    resp = edge_client.post(
        "/ingest", json={"vectors": v, "doc_ids": ["d1", "d2", "d3", "d4", "d5"]}
    )
    assert resp.status_code == 200
    assert resp.json()["added"] == 5

    # Search
    q = np.random.randn(128).astype(np.float32).tolist()
    s_resp = edge_client.post("/search", json={"query": q, "k": 2})
    assert s_resp.status_code == 200
    data = s_resp.json()
    assert len(data["results"]) == 2


def test_coordinator_route_and_register():
    # Register
    req = {"edge_id": "edge-test-1", "url": "http://edge-test-1:8000"}
    r = coord_client.post("/register", json=req)
    assert r.status_code == 200

    # Heartbeat
    metrics = {
        "cpu_usage": 10.0,
        "memory_usage": 20.0,
        "active_queries": 1,
        "query_latency_ms": 5.0,
    }
    h_req = {"edge_id": "edge-test-1", "metrics": metrics}
    h = coord_client.post("/heartbeat", json=h_req)
    assert h.status_code == 200

    # Route
    q = np.random.randn(128).astype(np.float32).tolist()
    rt_req = {"query": q, "k": 10, "strategy": "least_loaded"}
    rt = coord_client.post("/route", json=rt_req)
    assert rt.status_code == 200
    data = rt.json()
    assert data["edge_ids"] == ["edge-test-1"]
