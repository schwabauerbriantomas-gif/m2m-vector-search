#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M LlamaIndex RAG Example

Demonstrates Retrieval-Augmented Generation (RAG) using M2M (Machine-to-Memory)
as vectorstore with LlamaIndex.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys

# Import M2M modules
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from m2m import M2MConfig, normalize_sphere, SplatStore, M2MEngine
except ImportError as e:
    print(f"[ERROR] Could not import M2M modules: {e}")
    print("[INFO] Please ensure all M2M modules are in the projects/m2m directory")
    sys.exit(1)


class LlamaIndexRAG:
    """M2M-based RAG system compatible with LlamaIndex."""
    
    def __init__(self, config: M2MConfig):
        self.config = config
        
        # Initialize M2M Engine
        print("[INFO] Initializing M2M Engine for LlamaIndex RAG...")
        self.m2m = M2MEngine(config)
        
        # Sample documents (in real system, load from disk)
        self.documents = {
            "doc1": "M2M is a high-performance Gaussian Splat storage and retrieval system designed specifically for AI applications requiring persistent, long-term memory.",
            "doc2": "The system uses a 3-tier memory hierarchy (VRAM Hot, RAM Warm, SSD Cold) for efficient storage and retrieval.",
            "doc3": "It achieves 9x-92x speedup vs linear search through Hierarchical Region Merging (HRM2) clustering.",
            "doc4": "Self-Organized Criticality (SOC) automatically consolidates representations without human intervention.",
            "doc5": "GPU acceleration via Vulkan compute shaders enables massive parallelism for energy calculations and KNN search.",
            "doc6": "The system is optimized for AMD RX 6650XT (8GB VRAM) with full Vulkan support.",
            "doc7": "M2M provides REST/gRPC APIs for external tools and native Python SDK for easy integration.",
            "doc8": "M2M is licensed under Apache 2.0, allowing commercial use, proprietary modifications, and redistribution.",
            "doc9": "The system is specifically designed for applications requiring persistent, long-term memory with ultra-low latency.",
            "doc10": "M2M can handle 10K-100K dynamic Gaussian Splats with automatic expansion and SOC-based consolidation."
        }
        
        print(f"[INFO] Loaded {len(self.documents)} documents")
        
        # Embed documents (use BERT/GPT-2 embeddings in real system)
        print("[INFO] Creating document embeddings...")
        self.doc_embeddings = []
        for i in range(len(self.documents)):
            # In real system, use BERT/GPT-2 to embed text
            # Here we use random embeddings on S^639 for demonstration
            doc_vec = torch.randn(self.config.latent_dim)
            doc_vec_normalized = normalize_sphere(doc_vec)
            self.doc_embeddings.append(doc_vec_normalized)
        
        self.doc_embeddings_tensor = torch.stack(self.doc_embeddings)
        print(f"[INFO] Created document embeddings: {self.doc_embeddings_tensor.shape}")
        
        # Add documents to M2M
        print("[INFO] Adding documents to M2M VectorStore...")
        n_added = self.m2m.add_splats(self.doc_embeddings_tensor)
        print(f"[INFO] Added {n_added} documents to M2M")
    
    def embed_query(self, query: str) -> torch.Tensor:
        """Embed query to spherical space."""
        # In real system, use BERT/GPT-2 to embed query text
        # Here we use a random embedding on S^639 for demonstration
        query_vec = torch.randn(self.config.latent_dim)
        query_vec_normalized = normalize_sphere(query_vec)
        return query_vec_normalized.unsqueeze(0)
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[str]:
        """Retrieve k-relevant documents for query."""
        print(f"[INFO] Retrieving top-{k} documents for query: '{query}'")
        
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Retrieve from M2M
        neighbors_mu, neighbors_alpha, neighbors_kappa = self.m2m.search(query_embedding, k)
        
        # Get document indices
        top_k_indices = torch.topk(neighbors_kappa, k).indices
        
        # Retrieve documents
        retrieved_docs = [self.documents[i] for i in top_k_indices[0].tolist()]
        
        return retrieved_docs
    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response based on retrieved documents."""
        print(f"[INFO] Generating response for query: '{query}'")
        print(f"[INFO] Context documents: {len(context_docs)}")
        
        # In real system, pass query + context to LLM
        # Here we provide a simple template response
        
        response = f"Based on the following relevant documents:\n\n"
        for i, doc in enumerate(context_docs, 1):
            response += f"{i}. {doc}\n"
        
        response += f"\nHere is the answer to: '{query}'"
        
        return response


def main():
    """Main function to run LlamaIndex RAG example."""
    print("=" * 60)
    print("M2M LlamaIndex RAG Example")
    print("=" * 60)
    print()
    
    # Configuration
    config = M2MConfig(
        device='cpu',  # Use CPU for this example
        latent_dim=640,
        n_splats_init=10000,
        max_splats=100000,
        knn_k=64
    )
    
    # Initialize RAG system
    print("[STEP 1/4] Initializing M2M LlamaIndex RAG system...")
    rag = LlamaIndexRAG(config)
    
    # Query
    query = "What is M2M and how does it work?"
    print(f"[STEP 2/4] Processing query: '{query}'")
    
    # Retrieve relevant documents
    print("[STEP 3/4] Retrieving relevant documents...")
    context_docs = rag.retrieve_relevant_docs(query, k=3)
    
    # Generate response
    print("[STEP 4/4] Generating response...")
    response = rag.generate_response(query, context_docs)
    
    # Display results
    print()
    print("=" * 60)
    print("M2M LlamaIndex RAG Results")
    print("=" * 60)
    print()
    print(f"Query: {query}")
    print()
    print("Retrieved Documents:")
    for i, doc in enumerate(context_docs, 1):
        print(f"  {i}. {doc[:100]}...")
    print()
    print("Generated Response:")
    print(response)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
