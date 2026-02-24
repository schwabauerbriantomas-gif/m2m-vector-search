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
        
        # Initialize M2M VectorStore
        print("[INFO] Initializing M2M VectorStore...")
        from langchain_core.embeddings import FakeEmbeddings
        from integrations.langchain import M2MVectorStore
        
        # Use FakeEmbeddings for demonstration (in real system, use HuggingFaceEmbeddings)
        self.embeddings = FakeEmbeddings(size=config.latent_dim)
        
        # Initialize the M2M store wrapper
        self.vectorstore = M2MVectorStore(
            embeddings=self.embeddings,
            config=self.config
        )
        
        # Sample documents
        self.documents = [
            "M2M is a high-performance Gaussian Splat storage system designed for AI applications requiring persistent, long-term memory.",
            "The system uses a 3-tier memory hierarchy (VRAM Hot, RAM Warm, SSD Cold) for efficient storage and retrieval.",
            "It achieves 9x-92x speedup vs linear search through Hierarchical Region Merging (HRM2) clustering.",
            "Self-Organized Criticality (SOC) automatically consolidates representations without human intervention.",
            "GPU acceleration via Vulkan compute shaders enables massive parallelism for energy calculations and KNN search.",
            "M2M is optimized for AMD RX 6650XT (8GB VRAM) with full Vulkan support."
        ]
        
        print(f"[INFO] Loaded {len(self.documents)} documents")
    
    def embed_documents(self) -> None:
        """Add documents to M2M vectorstore."""
        print("[INFO] Adding documents to M2M vectorstore...")
        
        self.vectorstore.add_texts(self.documents)
        
        print(f"[INFO] Added {len(self.documents)} documents to vectorstore")
    
    def add_to_vectorstore(self, doc_embeddings) -> None:
        pass # Not needed anymore
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[str]:
        """Retrieve k-relevant documents for a query."""
        print(f"[INFO] Retrieving top-{k} relevant documents for query: '{query}'")
        
        # Retrieve k-nearest neighbors from M2M via LangChain API
        docs = self.vectorstore.similarity_search(query, k=k)
        
        print(f"[INFO] Retrieved {len(docs)} documents")
        
        return [doc.page_content for doc in docs]
    
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
    rag.embed_documents()
    print()
    
    # Query
    query = "What is M2M and how does it work?"
    print(f"[STEP 2/4] Processing query: '{query}'")
    
    # Retrieve relevant documents
    print("[STEP 3/4] Retrieving relevant documents...")
    context_docs = rag.retrieve_relevant_docs(query, k=3)
    print()
    
    # Generate response
    print("[STEP 4/4] Generating response...")
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
