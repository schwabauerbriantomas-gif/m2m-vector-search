import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
import itertools

@dataclass
class LSHConfig:
    """Configuración del índice LSH"""
    dim: int = 640                    # Dimensión de embeddings
    n_tables: int = 15                # Número de tablas hash
    n_bits: int = 18                  # Bits por hash (concatenación de k funciones)
    n_probes: int = 50                # Probes para multi-probe
    n_candidates: int = 500           # Candidatos a re-rankear
    seed: int = 42

class CrossPolytopeLSH:
    def __init__(self, config: LSHConfig):
        self.config = config
        self.dim = config.dim
        self.n_tables = config.n_tables
        self.n_bits = config.n_bits

        self.k = max(1, int(np.ceil(config.n_bits / np.log2(2 * self.dim))))

        np.random.seed(config.seed)
        
        self.rotations = []
        for _ in range(self.n_tables):
            table_rots = []
            for _ in range(self.k):
                table_rots.append(self._random_rotation(self.dim))
            self.rotations.append(table_rots)

        self.tables: List[dict] = [{} for _ in range(self.n_tables)]
        self.vectors: Optional[np.ndarray] = None
        self.n_vectors = 0

    def _random_rotation(self, dim: int) -> np.ndarray:
        A = np.random.randn(dim, dim)
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        return Q.astype(np.float32)

    def _compute_hash_vector(self, vector: np.ndarray, table_idx: int) -> Tuple[int, ...]:
        hashes = []
        for i in range(self.k):
            rotation = self.rotations[table_idx][i]
            rotated = rotation @ vector
            abs_rotated = np.abs(rotated)
            max_idx = np.argmax(abs_rotated)
            sign_bit = 0 if rotated[max_idx] >= 0 else 1
            hashes.append(int(max_idx * 2 + sign_bit))
        return tuple(hashes)

    def _multi_probe_hashes(self, vector: np.ndarray, table_idx: int, n_probes: int) -> List[Tuple[int, ...]]:
        M = min(self.dim, max(2, int(np.ceil(n_probes ** (1/self.k))) * 2))
        
        parts = []
        for i in range(self.k):
            rotation = self.rotations[table_idx][i]
            rotated = rotation @ vector
            abs_rotated = np.abs(rotated)
            
            top_indices = np.argpartition(abs_rotated, -M)[-M:]
            top_indices = top_indices[np.argsort(abs_rotated[top_indices])[::-1]]
            
            part_hashes = []
            for idx in top_indices:
                sign_bit = 0 if rotated[idx] >= 0 else 1
                h = int(idx * 2 + sign_bit)
                score = float(abs_rotated[idx])
                part_hashes.append((h, score))
            parts.append(part_hashes)
            
        probe_candidates = []
        import itertools
        for combo in itertools.product(*parts):
            hash_tuple = tuple(c[0] for c in combo)
            total_score = sum(c[1] for c in combo)
            probe_candidates.append((total_score, hash_tuple))
            
        probe_candidates.sort(reverse=True, key=lambda x: x[0])
        return [c[1] for c in probe_candidates[:n_probes]]

    def index(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)
        self.n_vectors = vectors.shape[0]

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-10)

        self.tables = [{} for _ in range(self.n_tables)]

        for t in range(self.n_tables):
            hashes_for_table = np.zeros((self.k, self.n_vectors), dtype=int)
            for i in range(self.k):
                rotation = self.rotations[t][i]
                
                rotated = rotation @ self.vectors.T 
                abs_rotated = np.abs(rotated)
                
                max_idx = np.argmax(abs_rotated, axis=0)
                max_vals = rotated[max_idx, np.arange(self.n_vectors)]
                sign_bits = (max_vals < 0).astype(int)
                
                hashes_for_table[i] = max_idx * 2 + sign_bits
                
            hashes_t = hashes_for_table.T
            for idx in range(self.n_vectors):
                hash_key = tuple(hashes_t[idx])
                if hash_key not in self.tables[t]:
                    self.tables[t][hash_key] = []
                self.tables[t][hash_key].append(idx)

    def query(self, query: np.ndarray, k: int = 10, n_candidates: Optional[int] = None, n_probes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if query.ndim == 2:
            query = query.squeeze(0)

        n_candidates = n_candidates or self.config.n_candidates
        n_probes = n_probes or self.config.n_probes

        query = query.astype(np.float32)
        query = query / np.linalg.norm(query)

        candidates = set()

        for t in range(self.n_tables):
            probe_hashes = self._multi_probe_hashes(query, t, n_probes)
            for h in probe_hashes:
                if h in self.tables[t]:
                    candidates.update(self.tables[t][h])

        candidates_list = list(candidates)

        if len(candidates_list) < k:
            extra = np.random.choice(self.n_vectors, size=min(k * 10, self.n_vectors), replace=False)
            candidates.update(extra)
            candidates_list = list(candidates)

        if len(candidates_list) > n_candidates:
            candidates_list = candidates_list[:n_candidates]

        if not candidates_list:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        candidates_arr = np.array(candidates_list)
        candidate_vectors = self.vectors[candidates_arr]

        distances = np.linalg.norm(candidate_vectors - query, axis=1)
        sorted_indices = np.argsort(distances)[:min(k, len(distances))]

        return candidates_arr[sorted_indices], distances[sorted_indices]

    def get_recall(self, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
        n_queries = queries.shape[0]
        total_found = 0
        total_expected = n_queries * k

        for i in range(n_queries):
            predicted, _ = self.query(queries[i], k=k)
            found = len(set(predicted) & set(ground_truth[i]))
            total_found += found

        return total_found / total_expected
