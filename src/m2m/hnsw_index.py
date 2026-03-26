"""
HNSW Index - Pure Python implementation for M2M

Implements the Hierarchical Navigable Small World graph algorithm
for approximate nearest neighbor search. Falls back gracefully when
hnswlib C extension is not available.

Reference: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs" (2016).
"""

import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .interfaces import IndexSearchResult, VectorIndex


class HNSWNode:
    """A node in the HNSW graph."""
    __slots__ = ('idx', 'level', 'neighbors')

    def __init__(self, idx: int, level: int):
        self.idx = idx
        self.level = level
        self.neighbors: List[List[int]] = [[] for _ in range(level + 1)]


class HNSWIndex(VectorIndex):
    """Pure-Python HNSW index for approximate nearest neighbor search.

    Parameters:
        dim: Vector dimensionality
        M: Max number of connections per node per layer
        ef_construction: Size of dynamic candidate list during construction
        ef_search: Size of dynamic candidate list during search
        max_level_mult: Controls the max layer (log(M) * max_level_mult)
        metric: 'cosine' or 'euclidean'
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_level_mult: float = 1.0 / math.log(2),
        metric: str = "cosine",
        seed: int = 42,
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M  # max connections at layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_level_mult = max_level_mult
        self.metric = metric
        self._rng = random.Random(seed)

        self._vectors: List[np.ndarray] = []
        self._nodes: List[HNSWNode] = []
        self._entry_point: int = -1
        self._max_level: int = -1
        self._removed: Set[int] = set()

    def build(self, vectors: np.ndarray) -> None:
        """Build index from vectors [N, D]."""
        self._vectors.clear()
        self._nodes.clear()
        self._removed.clear()
        self._entry_point = -1
        self._max_level = -1

        for i in range(len(vectors)):
            self._insert(vectors[i], i)

    def search(self, query: np.ndarray, k: int = 10) -> IndexSearchResult:
        """Search for k nearest neighbors."""
        if len(self._vectors) == 0:
            return IndexSearchResult(
                indices=np.array([], dtype=np.int64),
                distances=np.array([], dtype=np.float32),
            )

        ef = max(self.ef_search, k)

        # Phase 1: Greedy descent from top layer to layer 1 to find entry point
        ep = self._entry_point
        for level in range(self._max_level, 0, -1):
            ep = self._greedy_closest(query, ep, level)

        # Phase 2: Search at layer 0 with full ef
        candidates = self._search_layer(query, ep, ef, 0)

        # Filter removed and take top-k
        results = [(dist, idx) for dist, idx in candidates if idx not in self._removed]
        results.sort()
        results = results[:k]

        if not results:
            return IndexSearchResult(
                indices=np.array([], dtype=np.int64),
                distances=np.array([], dtype=np.float32),
            )

        indices = np.array([idx for _, idx in results], dtype=np.int64)
        distances = np.array([dist for dist, _ in results], dtype=np.float32)
        return IndexSearchResult(indices=indices, distances=distances)

    def add(self, vectors: np.ndarray) -> None:
        """Add new vectors to the index."""
        vectors = np.asarray(vectors, dtype=np.float32)
        for i in range(len(vectors)):
            base_idx = len(self._vectors)
            self._insert(vectors[i], base_idx)

    def remove(self, indices: np.ndarray) -> None:
        """Mark indices as removed (lazy deletion)."""
        for idx in indices:
            self._removed.add(int(idx))

    @property
    def n_items(self) -> int:
        return len(self._vectors) - len(self._removed)

    @property
    def supports_remove(self) -> bool:
        return True

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.max_level_mult)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            dot = float(np.dot(a, b))
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            denom = norm_a * norm_b
            if denom < 1e-10:
                return 1.0
            sim = dot / denom
            return 1.0 - sim
        else:
            return float(np.linalg.norm(a - b))

    def _insert(self, vector: np.ndarray, idx: int) -> None:
        level = self._random_level()
        node = HNSWNode(idx, level)
        self._vectors.append(vector.copy())
        self._nodes.append(node)

        if self._entry_point == -1:
            self._entry_point = idx
            self._max_level = level
            return

        query = vector
        ep = self._entry_point

        # Traverse from top to above insertion level
        for lev in range(self._max_level, level, -1):
            if lev <= self._max_level:
                ep = self._greedy_closest(query, ep, lev)

        # Insert into each layer from min(level, max_level) down to 0
        for lev in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(query, ep, self.ef_construction, lev)
            M_max = self.M_max0 if lev == 0 else self.M
            neighbors = self._select_neighbors(query, candidates, self.M, lev)

            # Set bidirectional connections
            node.neighbors[lev].extend(neighbors)
            for neighbor_idx in neighbors:
                neighbor = self._nodes[neighbor_idx]
                # Add back-link
                if idx not in neighbor.neighbors[lev]:
                    neighbor.neighbors[lev].append(idx)
                if len(neighbor.neighbors[lev]) > M_max:
                    # Prune: keep only M_max closest
                    neighbor_vec = self._vectors[neighbor_idx]
                    scored = [(self._distance(neighbor_vec, self._vectors[ni]), ni)
                              for ni in neighbor.neighbors[lev]]
                    scored.sort()
                    neighbor.neighbors[lev] = [ni for _, ni in scored[:M_max]]

            if candidates:
                ep = candidates[0][1]

        if level > self._max_level:
            self._max_level = level
            self._entry_point = idx

    def _greedy_closest(self, query: np.ndarray, entry: int, level: int) -> int:
        """Greedy walk to find closest node at a given level."""
        current = entry
        current_dist = self._distance(query, self._vectors[current])

        changed = True
        while changed:
            changed = False
            node = self._nodes[current]
            if level >= len(node.neighbors):
                break
            for neighbor_idx in node.neighbors[level]:
                if neighbor_idx in self._removed:
                    continue
                d = self._distance(query, self._vectors[neighbor_idx])
                if d < current_dist:
                    current_dist = d
                    current = neighbor_idx
                    changed = True

        return current

    def _search_layer(
        self, query: np.ndarray, entry: int, ef: int, level: int
    ) -> List[Tuple[float, int]]:
        """Search a single layer, returning (distance, index) pairs sorted by distance."""
        visited: Set[int] = {entry}
        dist_ep = self._distance(query, self._vectors[entry])

        # Min-heap for candidates (closest first), max-heap for results (farthest first for pruning)
        candidates = [(dist_ep, entry)]  # min-heap
        results = [(-dist_ep, entry)]    # max-heap (negate for max behavior)

        while candidates:
            dist_c, c = heapq.heappop(candidates)

            # If closest candidate is farther than farthest result, stop
            farthest_result = -results[0][0]
            if dist_c > farthest_result:
                break

            node = self._nodes[c]
            if level >= len(node.neighbors):
                continue

            for neighbor_idx in node.neighbors[level]:
                if neighbor_idx in visited or neighbor_idx in self._removed:
                    continue
                visited.add(neighbor_idx)

                dist_n = self._distance(query, self._vectors[neighbor_idx])
                farthest_result = -results[0][0]

                if dist_n < farthest_result or len(results) < ef:
                    heapq.heappush(candidates, (dist_n, neighbor_idx))
                    heapq.heappush(results, (-dist_n, neighbor_idx))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(abs(d), idx) for d, idx in sorted(results)]

    def _select_neighbors(
        self, query: np.ndarray, candidates: List[Tuple[float, int]], M: int, level: int
    ) -> List[int]:
        """Simple selection: take M closest from candidates."""
        sorted_cands = sorted(candidates)[:M]
        return [idx for _, idx in sorted_cands]
