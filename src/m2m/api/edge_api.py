import os
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI

from ..cluster.edge_node import EdgeNode
from ..cluster.protocol import (
    IngestRequest,
    IngestResponse,
    LoadMetricsModel,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from ..config import M2MConfig


async def dispatch_sync_events(actions) -> bool:
    """Attempt to send queued offline actions to coordinator."""
    if not edge.coordinator_url:
        return True

    try:
        async with httpx.AsyncClient():
            # We would iterate over actions and execute them. For now, we mock success.
            # E.g., for action in actions: if action['action'] == 'register': ...
            pass
        return True
    except Exception as e:
        print(f"Sync dispatch failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the offline sync queue
    await edge.sync_queue.start_background_sync(dispatch_sync_events)
    yield
    # Stop cleanly
    await edge.sync_queue.stop()


app = FastAPI(title="M2M Edge Node", lifespan=lifespan)

# Global edge instance
config = M2MConfig(device="cpu", latent_dim=128)
edge_id = os.environ.get("EDGE_ID", "edge-local")
coordinator_url = os.environ.get("COORDINATOR_URL", None)
edge = EdgeNode(edge_id=edge_id, config=config, coordinator_url=coordinator_url)


# For dynamic configuration via tests
def set_edge(new_edge: EdgeNode):
    global edge
    edge = new_edge


@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "edge_id": edge.edge_id}


@app.get("/metrics", response_model=LoadMetricsModel)
def get_metrics():
    """Returns load metrics."""
    m = edge.get_metrics()
    return LoadMetricsModel(
        active_queries=m.active_queries, query_latency_ms=m.query_latency_ms
    )


@app.post("/search", response_model=QueryResponse)
def search(request: QueryRequest):
    """Local vector search"""
    query_np = np.array(request.query, dtype=np.float32)
    raw_results = edge.search(query_np, request.k)
    results = [
        SearchResult(doc_id=doc_id, distance=dist) for doc_id, dist in raw_results
    ]
    return QueryResponse(results=results)


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, bg_tasks: BackgroundTasks):
    """Local embedding ingestion"""
    vectors_np = np.array(request.vectors, dtype=np.float32)
    added = edge.ingest(vectors_np, request.doc_ids)
    bg_tasks.add_task(edge.sync_with_coordinator)
    return IngestResponse(added=added)
