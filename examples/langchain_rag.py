#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M LangChain RAG Example

Demonstrates Retrieval-Augmented Generation (RAG) using M2M (Machine-to-Memory)
as the vectorstore.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys

# Import M2M modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from m2m import M2MConfig, SplatStore, normalize_sphere, geodesic_distance, M2MEngine
except ImportError as e:
    print(f"[ERROR] Could not import M2M modules: {e}")
    sys.exit(1)


class LangChainRAG:
    """M2M-based RAG system."""
    
    def __init__(self, config: M2MConfig):
        self.config = config
        
        # Initialize M2M Engine
        print("[INFO] Initializing M2M Engine for RAG...")
        self.m2m = M2MEngine(config)
        
        # Initialize vectorstore
        print("[INFO] Initializing M2M VectorStore...")
        self.vectorstore = self.m2m.splats
        
        # Sample documents
        self.documents = {
            "doc1": "M2M is a high-performance Gaussian Splat storage system designed for AI applications requiring persistent, long-term memory.",
            "doc2": "The system uses a 3-tier memory hierarchy (VRAM Hot, RAM Warm, SSD Cold) for efficient storage and retrieval.",
            "doc3": "It achieves 9x-92x speedup vs linear search through Hierarchical Region Merging (HRM2) clustering.",
            "doc4": "Self-Organized Criticality (SOC) automatically consolidates representations without human intervention.",
            "doc5": "GPU acceleration via Vulkan compute shaders enables massive parallelism for energy calculations and KNN search.",
            "doc6": "M2M is optimized for AMD RX 6650XT (8GB VRAM) with full Vulkan support."
        }
        
        print(f"[INFO] Loaded {len(self.documents)} documents")
    
    def embed_documents(self) -> torch.Tensor:
        """Embed documents using simple random embeddings (in real system, use BERT/GPT-2)."""
        print("[INFO] Creating document embeddings...")
        
        # In real system, use BERT, GPT-2, or Sentence-BERT
        # Here we use random embeddings for demonstration
        doc_vectors = []
        for i in range(len(self.documents)):
            # Random vector on S^639
            vec = torch.randn(self.config.latent_dim)
            doc_vectors.append(vec)
        
        # Normalize to unit sphere
        doc_vectors_tensor = torch.stack(doc_vectors)
        doc_vectors_normalized = normalize_sphere(doc_vectors_tensor)
        
        print(f"[INFO] Created {doc_vectors_normalized.shape[0]} document embeddings")
        
        return doc_vectors_normalized
    
    def add_to_vectorstore(self, doc_embeddings: torch.Tensor) -> None:
        """Add documents to M2M vectorstore."""
        print("[INFO] Adding documents to M2M vectorstore...")
        
        # Add to splat store
        for i, doc_embedding in enumerate(doc_embeddings):
            self.m2m.add_splat(doc_embedding)
        
        print(f"[INFO] Added {doc_embeddings.shape[0]} documents to vectorstore")
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[str]:
        """Retrieve k-relevant documents for a query."""
        print(f"[INFO] Retrieving top-{k} relevant documents for query: '{query}'")
        
        # In real system, embed query (use BERT, GPT-2, etc.)
        # Here we use a random query embedding for demonstration
        query_embedding = torch.randn(self.config.latent_dim)
        query_embedding = normalize_sphere(query_embedding)
        
        # Retrieve k-nearest neighbors from M2M
        neighbors_mu, neighbors_alpha, neighbors_kappa = self.vectorstore.find_neighbors(
            query_embedding.unsqueeze(0),
            k=k
        )
        
        print(f"[INFO] Retrieved {k} documents")
        
        return [
            self.documents[i]
            for i in torch.topk(neighbors_kappa.squeeze(0), k).indices.tolist()
        ]
    
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
    """Main function to run M2M RAG example."""
    print("=" * 60)
    print("M2M LangChain RAG Example")
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
    
    # Initialize M2M RAG system
    print("[INFO] Initializing M2M RAG system...")
    rag = LangChainRAG(config)
    print()
    
    # Embed documents
    print("[STEP 1/4] Embedding documents...")
    doc_embeddings = rag.embed_documents()
    print()
    
    # Add to vectorstore
    print("[STEP 2/4] Adding documents to vectorstore...")
    rag.add_to_vectorstore(doc_embeddings)
    print()
    
    # Query
    query = "What is M2M and how does it work?"
    print(f"[STEP 3/4] Processing query: '{query}'")
    
    # Retrieve relevant documents
    print("[STEP 4/4] Retrieving relevant documents...")
    context_docs = rag.retrieve_relevant_docs(query, k=3)
    print()
    
    # Generate response
    print("Generating response...")
    response = rag.generate_response(query, context_docs)
    print()
    
    # Display results
    print("=" * 60)
    print("M2M RAG Results")
    print("=" * 60)
    print()
    print(f"Query: {query}")
    print()
    print("Retrieved Documents:")
    for i, doc in enumerate(context_docs, 1):
        print(f"  {i}. {doc[:80]}...")
    print()
    print("Generated Response:")
    print(response)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
