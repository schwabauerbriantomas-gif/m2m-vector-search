from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LoadMetrics:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_queries: int = 0
    query_latency_ms: float = 0.0


@dataclass
class GeoLocation:
    latitude: float
    longitude: float


@dataclass
class EdgeNodeInfo:
    edge_id: str
    url: str
    status: str = "offline"  # online, offline, degraded
    last_heartbeat: float = 0.0
    document_count: int = 0
    location: Optional[GeoLocation] = None
    capabilities: Dict[str, bool] = field(default_factory=dict)
