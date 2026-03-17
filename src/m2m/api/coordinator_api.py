import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException

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
async def register_edge(request: RegisterRequest):
    info = router.register_edge(request.edge_id, request.url)
    return {"status": "success", "edge_info": info}


@app.post("/heartbeat")
async def heartbeat(request: HeartbeatRequest):
    router.heartbeat(request.edge_id, request.metrics)
    return {"status": "success"}


@app.post("/route", response_model=RouteResponse)
async def route_query(request: RouteRequest):
    query_np = np.array(request.query, dtype=np.float32)
    edge_ids = router.route_query(query_np, request.k, request.strategy)
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
async def search(request: QueryRequest):
    """
    End-to-end coordinator search.
    1. Routes query to correct edges.
    2. Fans out HTTP requests to edges concurrently.
    3. Aggregates results.
    """
    query_np = np.array(request.query, dtype=np.float32)

    # 1. Route
    edge_ids = router.route_query(query_np, request.k, request.strategy)
    if not edge_ids:
        raise HTTPException(status_code=503, detail="No online edge nodes available")

    # 2. Fanout
    tasks = []
    for edge_id in edge_ids:
        if edge_id in router.edge_nodes:
            edge_url = router.edge_nodes[edge_id].url
            tasks.append(fetch_edge_results(edge_id, edge_url, request))

    # Run requests concurrently
    raw_results = await asyncio.gather(*tasks)

    # Format for aggregator: dict[edge_id, list[(doc_id, distance)]]
    results_dict: Dict[str, List[Tuple[int, float]]] = {}
    for edge_id, res_list in raw_results:
        if res_list:
            results_dict[edge_id] = res_list

    # 3. Aggregate
    merged = aggregator.merge_results(results_dict, k=request.k, strategy="rrf")

    final_results = [SearchResult(doc_id=doc_id, distance=dist) for doc_id, dist in merged]
    return QueryResponse(results=final_results)
