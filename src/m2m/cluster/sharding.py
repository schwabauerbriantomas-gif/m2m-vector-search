import hashlib
import numpy as np
from typing import Dict
from .health import GeoLocation


def shard_by_hash(doc_id: str, num_edges: int) -> str:
    """
    Deterministic sharding by document ID hash.
    Distributes documents evenly across the available edge nodes.
    """
    if num_edges <= 0:
        raise ValueError("Number of edges must be greater than 0")

    # Create a deterministic integer from the doc_id string
    hash_obj = hashlib.md5(doc_id.encode("utf-8"))
    hash_val = int(hash_obj.hexdigest(), 16)

    edge_index = hash_val % num_edges
    return f"edge-{edge_index}"


def shard_by_cluster(
    embedding: np.ndarray, centroids: np.ndarray, edge_ids: list[str]
) -> str:
    """
    Route document to edge with nearest centroid.
    Advantage: Queries only need to hit relevant edges.
    """
    if len(centroids) == 0 or len(edge_ids) == 0:
        raise ValueError("Centroids and edge_ids lists cannot be empty")

    distances = np.linalg.norm(centroids - embedding, axis=1)
    nearest_idx = int(np.argmin(distances))

    # Ensure index bounds
    nearest_idx = nearest_idx % len(edge_ids)
    return edge_ids[nearest_idx]


def _haversine_distance(loc1: GeoLocation, loc2: GeoLocation) -> float:
    """Calculate the great circle distance in kilometers between two points on the earth."""
    # Convert decimal degrees to radians
    from math import radians, cos, sin, asin, sqrt

    lon1, lat1, lon2, lat2 = map(
        radians, [loc1.longitude, loc1.latitude, loc2.longitude, loc2.latitude]
    )

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def shard_by_geo(doc_metadata: Dict, edge_locations: Dict[str, GeoLocation]) -> str:
    """
    Route document to nearest edge by geographic location.
    Advantage: Low latency for regional queries.
    """
    if not edge_locations:
        return "edge-default"

    loc_data = doc_metadata.get("geo_location")
    if not loc_data:
        # Fallback if no location data
        return list(edge_locations.keys())[0] if edge_locations else "edge-default"

    doc_location = GeoLocation(
        latitude=loc_data.get("lat", 0), longitude=loc_data.get("lon", 0)
    )

    nearest_edge = min(
        edge_locations.items(), key=lambda x: _haversine_distance(doc_location, x[1])
    )

    return nearest_edge[0]
