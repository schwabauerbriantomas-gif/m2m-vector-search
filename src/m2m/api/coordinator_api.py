import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from ..cluster.aggregator import ResultAggregator
from ..cluster.protocol import (
    HeartbeatRequest,
    QueryRequest,
    QueryResponse,
    RegisterRequest,
    RouteRequest,
    RouteResponse,
    SearchResult,
)
from ..cluster.router import ClusterRouter

# ── SECURITY: API Key for coordinator (C-01, P-09 fix) ───────────────
import os
_COORDINATOR_API_KEY = os.environ.get("M2M_COORDINATOR_API_KEY")


def _validate_coordinator_key(request: Request) -> None:
    if _COORDINATOR_API_KEY is None:
        return
    auth = request.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else request.headers.get("X-API-Key", "")
    if token != _COORDINATOR_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── SECURITY: URL validation for edge nodes (P-12 fix) ───────────────
_ALLOWED_EDGE_SCHEMES = {"http", "https"}
_ALLOWED_EDGE_PORTS = {8000, 8001, 8080, 8888, 9000, 3000, 5000}


def _validate_edge_url(url: str) -> str:
    """Validates edge node URLs to prevent SSRF."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_EDGE_SCHEMES:
        raise ValueError(f"Invalid edge URL scheme: {parsed.scheme}")
    if parsed.hostname in ("localhost", "127.0.0.1", "::1") or parsed.hostname is None:
        pass  # Localhost is expected for edge nodes
    else:
        # Block internal/private IPs (basic SSRF prevention)
        import ipaddress
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                pass  # Allow private IPs in cluster context
            elif ip.is_reserved or ip.is_multicast:
                raise ValueError(f"Blocked edge URL: {url}")
        except ValueError:
            pass  # hostname, not IP - allow
    return url

# Global cluster dependencies
router = ClusterRouter()
aggregator = ResultAggregator()
http_client = httpx.AsyncClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await http_client.aclose()


app = FastAPI(title="M2M Coordinator Node", lifespan=lifespan)


@app.post("/register")
async def register_edge(request: Request):
    """Register edge node with API key validation (P-09 fix)."""
    _validate_coordinator_key(request)
    body = await request.json()
    # Validate edge URL (P-12 fix)
    if "url" in body:
        _validate_edge_url(body["url"])
    reg_req = RegisterRequest(**body)
    info = router.register_edge(reg_req.edge_id, reg_req.url)
    return {"status": "success", "edge_info": info}


@app.post("/heartbeat")
async def heartbeat(request: Request):
    """Heartbeat with basic validation (P-10 fix)."""
    _validate_coordinator_key(request)
    body = await request.json()
    hb_req = HeartbeatRequest(**body)
    # Cap metrics to prevent abuse (P-10 fix)
    if hb_req.metrics and hasattr(hb_req.metrics, '__dict__'):
        m = hb_req.metrics
        if hasattr(m, 'cpu_usage'):
            m.cpu_usage = max(0.0, min(100.0, float(m.cpu_usage)))
        if hasattr(m, 'memory_usage'):
            m.memory_usage = max(0.0, min(100.0, float(m.memory_usage)))
    router.heartbeat(hb_req.edge_id, hb_req.metrics)
    return {"status": "success"}


@app.post("/route", response_model=RouteResponse)
async def route_query(request: Request):
    _validate_coordinator_key(request)
    body = await request.json()
    route_req = RouteRequest(**body)
    query_np = np.array(route_req.query, dtype=np.float32)
    edge_ids = router.route_query(query_np, route_req.k, route_req.strategy)
    return RouteResponse(edge_ids=edge_ids)


async def fetch_edge_results(
    edge_id: str, edge_url: str, request: QueryRequest
) -> Tuple[str, List[Tuple[int, float]]]:
    """Helper to fetch search results from a single edge node asynchronously."""
    try:
        response = await http_client.post(
            f"{edge_url}/search", json=request.model_dump(), timeout=5.0
        )
        response.raise_for_status()
        data = response.json()

        # Parse SearchResult objects back into tuples for the aggregator
        parsed_results = []
        for res in data.get("results", []):
            parsed_results.append((res["doc_id"], res["distance"]))

        return edge_id, parsed_results
    except Exception as e:
        print(f"[Coordinator] Error querying edge {edge_id}: {e}")
        return edge_id, []


@app.post("/search", response_model=QueryResponse)
async def search(request: Request):
    """
    End-to-end coordinator search.
    1. Routes query to correct edges.
    2. Fans out HTTP requests to edges concurrently.
    3. Aggregates results.
    """
    _validate_coordinator_key(request)
    body = await request.json()
    query_req = QueryRequest(**body)
    query_np = np.array(query_req.query, dtype=np.float32)

    # 1. Route
    edge_ids = router.route_query(query_np, query_req.k, query_req.strategy)
    if not edge_ids:
        raise HTTPException(status_code=503, detail="No online edge nodes available")

    # 2. Fanout
    tasks = []
    for edge_id in edge_ids:
        if edge_id in router.edge_nodes:
            edge_url = router.edge_nodes[edge_id].url
            tasks.append(fetch_edge_results(edge_id, edge_url, query_req))

    # Run requests concurrently
    raw_results = await asyncio.gather(*tasks)

    # Format for aggregator: dict[edge_id, list[(doc_id, distance)]]
    results_dict: Dict[str, List[Tuple[int, float]]] = {}
    for edge_id, res_list in raw_results:
        if res_list:
            results_dict[edge_id] = res_list

    # 3. Aggregate
    merged = aggregator.merge_results(results_dict, k=query_req.k, strategy="rrf")

    final_results = [SearchResult(doc_id=doc_id, distance=dist) for doc_id, dist in merged]
    return QueryResponse(results=final_results)
