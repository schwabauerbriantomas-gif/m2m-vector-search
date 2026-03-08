from typing import Dict, List, Tuple


class ResultAggregator:
    """
    Merges results from multiple edge nodes.
    Strategies:
    - rrf: Reciprocal Rank Fusion
    - score: Score-based merge
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def merge_results(
        self,
        results: Dict[
            str, List[Tuple[int, float]]
        ],  # edge_id -> list of (doc_id, score/distance)
        k: int,
        strategy: str = "rrf",
    ) -> List[Tuple[int, float]]:
        """
        Merge and re-rank results from multiple edges.
        Note: The score in M2M usually represents geodesic distance (lower is better).
        We might need to invert it for RRF or just rank by distance.
        """
        if not results:
            return []

        if strategy == "rrf":
            return self._rrf_merge(results, k)
        elif strategy == "score":
            return self._score_merge(results, k)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

    def _rrf_merge(
        self, results: Dict[str, List[Tuple[int, float]]], k: int
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion.
        RRF_score = sum(1 / (k + rank_in_list))
        Since original returns distances, we just sort them first and assign ranks.
        """
        rrf_scores: Dict[int, float] = {}

        # Original distances mapped back for consistency if needed, but RRF produces a score
        # where higher is better. We will return RRF scores instead of distances,
        # or we could keep the best distance. Let's keep best distance for return compatibility.
        best_distances: Dict[int, float] = {}

        for edge_id, edge_results in results.items():
            # Sort by distance (ascending) to get ranks
            # Assuming edge_results are already sorted by the EdgeNode
            for rank, (doc_id, distance) in enumerate(edge_results):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                    best_distances[doc_id] = distance

                # RRF calculation
                rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)

                # Keep the minimum distance seen across edges
                if distance < best_distances[doc_id]:
                    best_distances[doc_id] = distance

        # Sort by RRF score descending (higher is better)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top k, but format as (doc_id, distance) to match single-node interface
        return [(doc_id, best_distances[doc_id]) for doc_id, _ in sorted_docs[:k]]

    def _score_merge(
        self, results: Dict[str, List[Tuple[int, float]]], k: int
    ) -> List[Tuple[int, float]]:
        """
        Simple score/distance-based merge.
        Combines all results and sorts by distance.
        """
        all_results = []

        # Flatten and filter duplicates (keeping best distance)
        best_distances: Dict[int, float] = {}

        for edge_id, edge_results in results.items():
            for doc_id, distance in edge_results:
                if doc_id not in best_distances or distance < best_distances[doc_id]:
                    best_distances[doc_id] = distance

        # Convert back to list
        for doc_id, dist in best_distances.items():
            all_results.append((doc_id, dist))

        # Sort by distance ascending
        all_results.sort(key=lambda x: x[1])

        return all_results[:k]
