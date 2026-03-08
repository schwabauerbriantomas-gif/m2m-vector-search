from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class GeoLocationModel(BaseModel):
    latitude: float
    longitude: float


class EdgeNodeInfoModel(BaseModel):
    edge_id: str
    url: str
    status: str = "online"
    last_heartbeat: float = 0.0
    document_count: int = 0
    location: Optional[GeoLocationModel] = None
    capabilities: Dict[str, bool] = Field(default_factory=dict)


class LoadMetricsModel(BaseModel):
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_queries: int = 0
    query_latency_ms: float = 0.0


class HeartbeatRequest(BaseModel):
    edge_id: str
    metrics: LoadMetricsModel


class RegisterRequest(BaseModel):
    edge_id: str
    url: str


class QueryRequest(BaseModel):
    query: List[float]
    k: int = 10
    strategy: str = "broadcast"


class SearchResult(BaseModel):
    doc_id: int
    distance: float


class QueryResponse(BaseModel):
    results: List[SearchResult]


class RouteRequest(BaseModel):
    query: List[float]
    k: int = 10
    strategy: str = "broadcast"


class RouteResponse(BaseModel):
    edge_ids: List[str]


class IngestRequest(BaseModel):
    vectors: List[List[float]]
    doc_ids: Optional[List[str]] = None


class IngestResponse(BaseModel):
    added: int
