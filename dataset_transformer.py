"""
Dataset Transformer para M2M Vector Search.

Convierte embeddings planos en Gaussian Splats estructurados que 
aprovechan la arquitectura jerárquica de M2M.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass
from typing import List, Optional, Tuple
import struct
import json


@dataclass
class GaussianSplat:
    """Representación gaussiana de un cluster de vectores."""
    mu: np.ndarray        # Centroide [D]
    alpha: float          # Peso del splat
    kappa: float          # Concentración
    n_vectors: int        # Vectores originales
    indices: np.ndarray   # Índices originales


@dataclass 
class HRM2Node:
    """Nodo de la jerarquía HRM2."""
    splat: GaussianSplat
    children: List['HRM2Node']
    level: int
    parent: Optional['HRM2Node']


class M2MDatasetTransformer:
    """
    Transforma datasets de embeddings para optimizar M2M.
    
    Proceso:
    1. Clustering jerárquico con AgglomerativeClustering (Ward linkage)
    2. Conversión de clusters a Gaussian Splats
    3. Construcción de jerarquía HRM2
    4. Particionado para memoria 3-tier (hot/warm/cold)
    
    Usage:
        transformer = M2MDatasetTransformer(vectors, n_clusters_base=200)
        result = transformer.transform()
        transformer.save_for_m2m('output.bin')
    """
    
    def __init__(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[dict]] = None,
        n_clusters_base: int = 200,
        hierarchy_levels: int = 4,
        min_cluster_size: int = 10
    ):
        """
        Args:
            vectors: Array [N, D] de embeddings
            metadata: Metadatos opcionales por vector
            n_clusters_base: Clusters en nivel hoja
            hierarchy_levels: Profundidad del árbol HRM2
            min_cluster_size: Tamaño mínimo de cluster
        """
        self.vectors = vectors.astype(np.float32)
        self.metadata = metadata or [{} for _ in range(len(vectors))]
        self.n_clusters_base = n_clusters_base
        self.hierarchy_levels = hierarchy_levels
        self.min_cluster_size = min_cluster_size
        
        self.splats: List[GaussianSplat] = []
        self.hierarchy: Optional[HRM2Node] = None
        self.access_patterns: np.ndarray = None
    
    def transform(self) -> dict:
        """Ejecuta transformación completa y retorna resultado."""
        cluster_tree = self._build_cluster_tree()
        self.splats = self._clusters_to_splats(cluster_tree)
        self.hierarchy = self._build_hrm2_hierarchy(cluster_tree)
        self.access_patterns = self._simulate_access_patterns()
        partitions = self._partition_for_memory_tiers()
        
        return {
            'splats': self.splats,
            'hierarchy': self.hierarchy,
            'partitions': partitions,
            'stats': self._compute_stats()
        }
    
    def _build_cluster_tree(self) -> dict:
        """Construye árbol de clusters jerárquicos."""
        
        def cluster_recursive(vectors, indices, level=0):
            node = {
                'vectors': vectors,
                'indices': indices,
                'children': [],
                'level': level
            }
            
            if level >= self.hierarchy_levels or len(vectors) < self.min_cluster_size * 2:
                return node
            
            n_children = min(
                self.n_clusters_base // max(1, self.hierarchy_levels - level),
                len(vectors) // self.min_cluster_size
            )
            
            if n_children < 2:
                return node
            
            clustering = AgglomerativeClustering(
                n_clusters=n_children,
                linkage='ward'
            )
            labels = clustering.fit_predict(vectors)
            
            for i in range(n_children):
                mask = labels == i
                child = cluster_recursive(
                    vectors[mask],
                    indices[mask],
                    level + 1
                )
                node['children'].append(child)
            
            return node
        
        return cluster_recursive(self.vectors, np.arange(len(self.vectors)))
    
    def _clusters_to_splats(self, cluster_tree: dict) -> List[GaussianSplat]:
        """Convierte clusters hoja a Gaussian Splats."""
        splats = []
        
        def extract(node):
            if not node['children']:
                vectors = node['vectors']
                n = len(vectors)
                
                mu = np.mean(vectors, axis=0)
                distances = np.linalg.norm(vectors - mu, axis=1)
                variance = np.var(distances) + 1e-8
                kappa = np.clip(1.0 / (variance + np.mean(distances) + 1e-8), 0.1, 100.0)
                alpha = n / len(self.vectors)
                
                splats.append(GaussianSplat(
                    mu=mu.astype(np.float32),
                    alpha=float(alpha),
                    kappa=float(kappa),
                    n_vectors=n,
                    indices=node['indices'].copy()
                ))
            else:
                for child in node['children']:
                    extract(child)
        
        extract(cluster_tree)
        return splats
    
    def _build_hrm2_hierarchy(self, cluster_tree: dict) -> HRM2Node:
        """Construye jerarquía HRM2 explícita."""
        
        def build(node, level=0, parent=None):
            vectors = node['vectors']
            mu = np.mean(vectors, axis=0)
            distances = np.linalg.norm(vectors - mu, axis=1)
            kappa = np.clip(1.0 / (np.var(distances) + np.mean(distances) + 1e-8), 0.1, 100.0)
            
            splat = GaussianSplat(
                mu=mu.astype(np.float32),
                alpha=len(vectors) / len(self.vectors),
                kappa=float(kappa),
                n_vectors=len(vectors),
                indices=node['indices']
            )
            
            hrm2_node = HRM2Node(splat=splat, children=[], level=level, parent=parent)
            
            for child in node['children']:
                hrm2_node.children.append(build(child, level + 1, hrm2_node))
            
            return hrm2_node
        
        return build(cluster_tree)
    
    def _simulate_access_patterns(self) -> np.ndarray:
        """Simula patrones de acceso para particionado."""
        n_splats = len(self.splats)
        access = np.zeros(n_splats)
        
        # Simular 1000 queries
        for _ in range(1000):
            q_idx = np.random.randint(len(self.vectors))
            q = self.vectors[q_idx]
            distances = [np.linalg.norm(q - s.mu) for s in self.splats]
            access[np.argmin(distances)] += 1
        
        # Combinar con tamaño y concentración
        sizes = np.array([s.n_vectors for s in self.splats])
        kappas = np.array([s.kappa for s in self.splats])
        
        access = access / access.max() if access.max() > 0 else access
        result = 0.4 * access + 0.3 * (sizes / sizes.max()) + 0.3 * (kappas / kappas.max())
        
        return result / result.sum() if result.sum() > 0 else np.ones(n_splats) / n_splats
    
    def _partition_for_memory_tiers(self) -> dict:
        """Particiona splats en hot/warm/cold."""
        sorted_idx = np.argsort(self.access_patterns)[::-1]
        n = len(self.splats)
        
        return {
            'hot': {'indices': sorted_idx[:int(n*0.2)], 'tier': 'vram'},
            'warm': {'indices': sorted_idx[int(n*0.2):int(n*0.5)], 'tier': 'ram'},
            'cold': {'indices': sorted_idx[int(n*0.5):], 'tier': 'ssd'}
        }
    
    def _compute_stats(self) -> dict:
        """Computa estadísticas del resultado."""
        original_size = self.vectors.nbytes
        compressed_size = sum(
            s.mu.nbytes + 16 + s.indices.nbytes 
            for s in self.splats
        )
        
        return {
            'original_count': len(self.vectors),
            'splat_count': len(self.splats),
            'compression_ratio': len(self.vectors) / len(self.splats),
            'original_size_mb': original_size / 1024**2,
            'compressed_size_mb': compressed_size / 1024**2,
            'memory_savings_pct': (1 - compressed_size/original_size) * 100
        }
    
    def save_for_m2m(self, output_path: str) -> dict:
        """Guarda dataset en formato binario M2M."""
        result = self.transform()
        
        with open(output_path, 'wb') as f:
            dim = self.vectors.shape[1]
            
            # Header: 4 ints
            f.write(struct.pack('IIII', len(self.splats), dim, len(self.vectors), self.hierarchy_levels))
            
            # Cada splat
            for s in self.splats:
                f.write(s.mu.tobytes())
                f.write(struct.pack('ffI', s.alpha, s.kappa, s.n_vectors))
                f.write(s.indices.astype(np.int32).tobytes())
        
        # Metadatos JSON
        with open(output_path.replace('.bin', '_meta.json'), 'w') as f:
            json.dump(result['stats'], f, indent=2)
        
        print(f"✅ Guardado: {output_path}")
        print(f"   Splats: {len(self.splats):,}")
        print(f"   Compresión: {result['stats']['compression_ratio']:.1f}x")
        print(f"   Ahorro: {result['stats']['memory_savings_pct']:.1f}%")
        
        return result
