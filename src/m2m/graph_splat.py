import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


class NodeType(str, enum.Enum):
    """Tipos de nodos soportados en el grafo Gaussiano."""

    DOCUMENT = "DOCUMENT"
    ENTITY = "ENTITY"
    CONCEPT = "CONCEPT"
    RELATION = "RELATION"


@dataclass
class GraphEdge:
    """Arista dirigida en el grafo de conocimiento M2M."""

    source_id: int
    target_id: int
    relation_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphSplat:
    """
    Nodo en el grafo Gaussiano M2M. Combina la representación
    espacial continua de un GaussianSplat con topología de grafo.
    """

    id: int
    node_type: NodeType
    content: str
    embedding: np.ndarray  # S^639 hypersphere coordinates
    mu: np.ndarray  # Mean of splat
    alpha: float  # Opacity/importance
    kappa: float  # Concentration/precision

    metadata: Dict[str, Any] = field(default_factory=dict)

    # Graph topology
    incoming: List[GraphEdge] = field(default_factory=list)
    outgoing: List[GraphEdge] = field(default_factory=list)

    @property
    def degree(self) -> int:
        return len(self.incoming) + len(self.outgoing)


class GaussianGraphStore:
    """
    Store híbrido que combina búsqueda vectorial jerárquica (HRM2)
    con un grafo de conocimiento subyacente.
    """

    def __init__(self, config=None, dim=640, max_nodes=100000):
        self.config = config
        self.dim = dim
        self.max_nodes = max_nodes

        # Almacenamiento primario
        self.nodes: Dict[int, GraphSplat] = {}
        self.edges: List[GraphEdge] = []

        # Índices secundarios
        self.type_index: Dict[NodeType, Set[int]] = {t: set() for t in NodeType}
        self.entity_name_index: Dict[str, int] = {}

        # Integración con HRM2 para búsqueda vectorial
        # In a real setup, we would inject the actual engine config here
        self._next_id = 0

        # Arrays numpy para ruteo semántico súper rápido
        self.mu_buffer = np.zeros((self.max_nodes, self.dim), dtype=np.float32)
        self.active_mask = np.zeros(self.max_nodes, dtype=bool)

    def _get_next_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def add_document(self, text: str, embedding: np.ndarray, metadata: Dict = None) -> int:
        """Añade un documento completo al grafo."""
        doc_id = self._get_next_id()
        metadata = metadata or {}

        splat = GraphSplat(
            id=doc_id,
            node_type=NodeType.DOCUMENT,
            content=text,
            embedding=embedding,
            mu=embedding,  # Simplifying assumptions for now
            alpha=1.0,
            kappa=10.0,
            metadata=metadata,
        )
        self._add_node(splat)
        return doc_id

    def add_entity(self, name: str, embedding: np.ndarray, entity_type: str = "unknown") -> int:
        """Añade una entidad extrada al grafo (con deduplicación por nombre/vector)."""
        # Simple deduplication by name for this example
        name_lower = name.lower()
        if name_lower in self.entity_name_index:
            return self.entity_name_index[name_lower]

        entity_id = self._get_next_id()
        splat = GraphSplat(
            id=entity_id,
            node_type=NodeType.ENTITY,
            content=name,
            embedding=embedding,
            mu=embedding,
            alpha=0.8,
            kappa=15.0,  # Entities are more concentrated than documents
            metadata={"entity_type": entity_type},
        )
        self._add_node(splat)
        self.entity_name_index[name_lower] = entity_id
        return entity_id

    def add_relation(
        self, source_id: int, target_id: int, relation_type: str, weight: float = 1.0
    ) -> Optional[GraphEdge]:
        """Añade una arista dirigida entre dos nodos."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
        )

        self.nodes[source_id].outgoing.append(edge)
        self.nodes[target_id].incoming.append(edge)
        self.edges.append(edge)

        return edge

    def _add_node(self, node: GraphSplat):
        """Método interno para registrar el nodo en todos los índices."""
        self.nodes[node.id] = node
        self.type_index[node.node_type].add(node.id)

        # Actualizar buffer para búsqueda vectorial
        if node.id < self.max_nodes:
            self.mu_buffer[node.id] = node.embedding
            self.active_mask[node.id] = True

    def search_entities(self, query_emb: np.ndarray, k: int = 5) -> List[GraphSplat]:
        """Busca entidades similares usando k-NN en los embeddings."""
        # Filtrar solo entidades
        entity_ids = list(self.type_index[NodeType.ENTITY])
        if not entity_ids:
            return []

        # Extracción de muñez de entidades simulando la S^639 hypersphere
        entity_mus = self.mu_buffer[entity_ids]

        # Distancia coseno / producto punto en vectores normalizados
        similarities = np.dot(entity_mus, query_emb)

        # Top-K
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            real_id = entity_ids[idx]
            results.append(self.nodes[real_id])

        return results

    def traverse(self, start_node_id: int, max_depth: int = 2) -> Dict[str, Any]:
        """Realiza un recorrido BFS en el grafo desde un nodo inicial."""
        if start_node_id not in self.nodes:
            return {}

        visited = {start_node_id}
        queue = [(start_node_id, 0)]
        subgraph = {"nodes": [self.nodes[start_node_id]], "edges": []}

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            current_node = self.nodes[current_id]
            for edge in current_node.outgoing:
                subgraph["edges"].append(edge)
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    subgraph["nodes"].append(self.nodes[edge.target_id])
                    queue.append((edge.target_id, depth + 1))

        return subgraph

    def get_subgraph(self, node_ids: List[int]) -> Dict[str, Any]:
        """Devuelve un subgrafo inducido por los ids dados."""
        valid_ids = set(n for n in node_ids if n in self.nodes)
        nodes = [self.nodes[n] for n in valid_ids]
        edges = []

        for n_id in valid_ids:
            for edge in self.nodes[n_id].outgoing:
                if edge.target_id in valid_ids:
                    edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    def hybrid_search(
        self, query_text: str, query_emb: np.ndarray, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida que usa similitud vectorial + contexto del grafo.
        Combina el score de embedding con el PageRank / Grado del nodo.
        """
        # Step 1: Vector search over DOCUMENTS
        doc_ids = list(self.type_index[NodeType.DOCUMENT])
        if not doc_ids:
            return []

        doc_mus = self.mu_buffer[doc_ids]
        similarities = np.dot(doc_mus, query_emb)

        # Top-K inicial (k*2 para re-ranking)
        candidates_k = min(k * 2, len(doc_ids))
        top_indices = np.argsort(similarities)[::-1][:candidates_k]

        results = []
        for idx in top_indices:
            real_id = doc_ids[idx]
            node = self.nodes[real_id]
            sim_score = float(similarities[idx])

            # Context boost por entidades conectadas
            graph_boost = 0.0
            if node.outgoing:
                graph_boost = min(len(node.outgoing) * 0.05, 0.2)

            final_score = sim_score + graph_boost

            results.append(
                {
                    "node": node,
                    "score": final_score,
                    "vector_score": sim_score,
                    "graph_boost": graph_boost,
                }
            )

        # Ordenar por final_score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del grafo."""
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": {},
            "active_buffer_size": int(np.sum(self.active_mask)),
        }

        for n_type in NodeType:
            stats["node_types"][n_type.value] = len(self.type_index[n_type])

        return stats
