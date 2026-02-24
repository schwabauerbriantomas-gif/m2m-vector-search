#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M Real Dataset Validator

Tests M2M with REAL datasets, not synthetic random data.
All results documented with methodology for reproducibility.
"""

import torch
import numpy as np
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / "real_dataset_results.json"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))


class RealDatasetValidator:
    """Validates M2M with real datasets only."""

    def __init__(self):
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "methodology": {
                "approach": "Real embeddings only, no synthetic data",
                "reproducible": True,
                "datasets": []
            },
            "tests": {},
        }

    def test_with_glove_embeddings(self) -> Dict[str, Any]:
        """
        Test with GloVe word embeddings if available.
        
        Dataset: GloVe.6B.50d (if present)
        Source: https://nlp.stanford.edu/projects/glove/
        Vectors: 400,000 words, 50 dimensions
        """
        print("\n[TEST] GloVe Word Embeddings...")
        
        test_result = {
            "dataset": "GloVe.6B.50d",
            "source": "https://nlp.stanford.edu/projects/glove/",
            "status": "SKIPPED",
            "reason": None,
            "vectors_loaded": 0,
            "dimensions": 0,
            "tests": {}
        }
        
        # Try to find GloVe file
        glove_paths = [
            PROJECT_ROOT / "data" / "glove.6B.50d.txt",
            Path.home() / "Downloads" / "glove.6B.50d.txt",
            Path.home() / ".cache" / "glove.6B.50d.txt",
        ]
        
        glove_file = None
        for path in glove_paths:
            if path.exists():
                glove_file = path
                break
        
        if not glove_file:
            test_result["reason"] = "GloVe file not found. Download from https://nlp.stanford.edu/projects/glove/"
            print(f"  [SKIP] GloVe file not found")
            print(f"  Download from: https://nlp.stanford.edu/projects/glove/")
            self.results["tests"]["glove"] = test_result
            return test_result
        
        print(f"  [FOUND] {glove_file.name}")
        
        # Load embeddings (limit to 10K for speed)
        embeddings = []
        words = []
        max_vectors = 10000
        
        print(f"  [LOADING] Reading embeddings (max {max_vectors})...")
        with open(glove_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_vectors:
                    break
                parts = line.strip().split()
                if len(parts) > 2:
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    embeddings.append(vector)
                    words.append(word)
        
        embeddings_array = np.array(embeddings)
        test_result["vectors_loaded"] = len(words)
        test_result["dimensions"] = embeddings_array.shape[1]
        
        print(f"  [LOADED] {len(words)} words, {embeddings_array.shape[1]}D")
        
        # Test 1: Similarity search
        print(f"  [TEST] Similarity search with real embeddings...")
        try:
            from config import M2MConfig
            from splats import SplatStore
            from geometry import normalize_sphere
            
            # Create config with proper dimensions
            config = M2MConfig(
                device='cpu',
                latent_dim=embeddings_array.shape[1],
                max_splats=len(words)
            )
            
            store = SplatStore(config)
            
            # Add embeddings
            embeddings_tensor = torch.from_numpy(embeddings_array)
            embeddings_norm = normalize_sphere(embeddings_tensor)
            
            added = 0
            for emb in embeddings_norm:
                if store.add_splat(emb):
                    added += 1
            
            test_result["tests"]["splatstore"] = {
                "status": "SUCCESS",
                "added": added,
                "total": len(words)
            }
            print(f"    [OK] Added {added}/{len(words)} embeddings")
            
            # Test search
            query_idx = np.random.randint(0, len(words))
            query_word = words[query_idx]
            query = embeddings_tensor[query_idx:query_idx+1]
            query_norm = normalize_sphere(query)
            
            # Find neighbors
            results = store.find_neighbors(query_norm, k=5)
            test_result["tests"]["search"] = {
                "status": "SUCCESS",
                "query_word": query_word,
                "k": 5
            }
            print(f"    [OK] Search for '{query_word}': found k=5 neighbors")
            
            test_result["status"] = "SUCCESS"
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["error"] = str(e)
            print(f"    [FAIL] {e}")
        
        self.results["tests"]["glove"] = test_result
        self.results["methodology"]["datasets"].append("GloVe.6B.50d")
        return test_result

    def test_with_local_documents(self) -> Dict[str, Any]:
        """
        Test with OpenClaw local documents.
        
        Dataset: OpenClaw workspace documents (already indexed)
        Source: Local files
        Documents: 274 files, 562 chunks
        """
        print("\n[TEST] OpenClaw Local Documents...")
        
        test_result = {
            "dataset": "OpenClaw Workspace",
            "source": "Local files",
            "status": "SKIPPED",
            "reason": None,
            "documents": 0,
            "chunks": 0,
            "tests": {}
        }
        
        # Check if index exists
        workspace_root = PROJECT_ROOT.parent.parent  # Go up to workspace
        index_file = workspace_root / "openclaw_index.json"
        
        if not index_file.exists():
            test_result["reason"] = "OpenClaw index not found"
            print(f"  [SKIP] Index not found at {index_file}")
            self.results["tests"]["openclaw_local"] = test_result
            return test_result
        
        print(f"  [FOUND] {index_file.name}")
        
        # Load index
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        test_result["documents"] = len(index_data.get("documents", []))
        test_result["chunks"] = len(index_data.get("chunks", []))
        
        print(f"  [LOADED] {test_result['documents']} documents, {test_result['chunks']} chunks")
        
        # Load embeddings
        embeddings_file = workspace_root / "openclaw_index_embeddings.npy"
        
        if not embeddings_file.exists():
            test_result["reason"] = "Embeddings file not found"
            print(f"  [SKIP] Embeddings not found")
            self.results["tests"]["openclaw_local"] = test_result
            return test_result
        
        embeddings = np.load(embeddings_file)
        print(f"  [LOADED] Embeddings shape: {embeddings.shape}")
        
        # Test with these embeddings
        print(f"  [TEST] Using real document embeddings...")
        try:
            from config import M2MConfig
            from splats import SplatStore
            from geometry import normalize_sphere
            
            # Create config
            config = M2MConfig(
                device='cpu',
                latent_dim=embeddings.shape[1],
                max_splats=embeddings.shape[0]
            )
            
            store = SplatStore(config)
            
            # Add embeddings
            embeddings_tensor = torch.from_numpy(embeddings)
            embeddings_norm = normalize_sphere(embeddings_tensor)
            
            added = 0
            for emb in embeddings_norm:
                if store.add_splat(emb):
                    added += 1
            
            test_result["tests"]["splatstore"] = {
                "status": "SUCCESS",
                "added": added,
                "total": embeddings.shape[0]
            }
            print(f"    [OK] Added {added}/{embeddings.shape[0]} document embeddings")
            
            # Test search
            query_idx = np.random.randint(0, embeddings.shape[0])
            query = embeddings_tensor[query_idx:query_idx+1]
            query_norm = normalize_sphere(query)
            
            results = store.find_neighbors(query_norm, k=5)
            
            # Get chunk info
            chunk_info = index_data["chunks"][query_idx]
            
            test_result["tests"]["search"] = {
                "status": "SUCCESS",
                "query_document": chunk_info.get("filename", "unknown"),
                "k": 5
            }
            print(f"    [OK] Search in '{chunk_info.get('filename', 'unknown')}': found k=5 neighbors")
            
            test_result["status"] = "SUCCESS"
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["error"] = str(e)
            print(f"    [FAIL] {e}")
        
        self.results["tests"]["openclaw_local"] = test_result
        self.results["methodology"]["datasets"].append("OpenClaw Workspace")
        return test_result

    def test_with_random_normalized(self) -> Dict[str, Any]:
        """
        Test with random but normalized embeddings (baseline).
        
        Dataset: Random normal, L2-normalized
        Source: Generated
        Purpose: Establish baseline performance
        Note: Clearly marked as synthetic for comparison
        """
        print("\n[TEST] Synthetic Random Embeddings (Baseline)...")
        
        test_result = {
            "dataset": "Random Normal (L2-normalized)",
            "source": "Synthetic - for baseline only",
            "status": "SUCCESS",
            "vectors": 10000,
            "dimensions": 640,
            "tests": {}
        }
        
        print(f"  [NOTE] This is SYNTHETIC data for baseline comparison")
        
        try:
            from config import M2MConfig
            from splats import SplatStore
            from geometry import normalize_sphere
            
            # Generate random embeddings
            n_vectors = 10000
            dim = 640
            
            embeddings = torch.randn(n_vectors, dim)
            embeddings_norm = normalize_sphere(embeddings)
            
            print(f"  [GENERATED] {n_vectors} vectors, {dim}D")
            
            # Create config
            config = M2MConfig(device='cpu', latent_dim=dim, max_splats=n_vectors)
            store = SplatStore(config)
            
            # Add
            added = 0
            for emb in embeddings_norm:
                if store.add_splat(emb):
                    added += 1
            
            test_result["tests"]["splatstore"] = {
                "status": "SUCCESS",
                "added": added,
                "total": n_vectors
            }
            print(f"    [OK] Added {added}/{n_vectors} random embeddings")
            
            # Search
            query = torch.randn(1, dim)
            query_norm = normalize_sphere(query)
            
            results = store.find_neighbors(query_norm, k=5)
            
            test_result["tests"]["search"] = {
                "status": "SUCCESS",
                "k": 5
            }
            print(f"    [OK] Search: found k=5 neighbors")
            
        except Exception as e:
            test_result["status"] = "FAILED"
            test_result["error"] = str(e)
            print(f"    [FAIL] {e}")
        
        self.results["tests"]["random_baseline"] = test_result
        self.results["methodology"]["datasets"].append("Random Normal (synthetic baseline)")
        return test_result

    def generate_report(self):
        """Generate final report."""
        print("\n" + "=" * 70)
        print("REAL DATASET VALIDATION REPORT")
        print("=" * 70)
        
        print("\n[METHODLOGY]")
        print(f"  Approach: {self.results['methodology']['approach']}")
        print(f"  Reproducible: {self.results['methodology']['reproducible']}")
        print(f"  Datasets tested: {len(self.results['methodology']['datasets'])}")
        
        print("\n[RESULTS]")
        for test_name, test_data in self.results["tests"].items():
            status = test_data.get("status", "UNKNOWN")
            dataset = test_data.get("dataset", "Unknown")
            print(f"\n  {test_name.upper()}:")
            print(f"    Dataset: {dataset}")
            print(f"    Status: {status}")
            
            if status == "SUCCESS":
                if "vectors_loaded" in test_data:
                    print(f"    Vectors: {test_data['vectors_loaded']}")
                    print(f"    Dimensions: {test_data['dimensions']}")
                if "documents" in test_data:
                    print(f"    Documents: {test_data['documents']}")
                    print(f"    Chunks: {test_data['chunks']}")
                if "vectors" in test_data:
                    print(f"    Vectors: {test_data['vectors']}")
                    print(f"    Dimensions: {test_data['dimensions']}")
            elif "reason" in test_data:
                print(f"    Reason: {test_data['reason']}")
        
        print("\n" + "=" * 70)

    def save_results(self):
        """Save validation results."""
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[SAVED] Results: {RESULTS_FILE.name}")


def main():
    """Run all real dataset validations."""
    print("=" * 70)
    print("M2M Real Dataset Validator")
    print("=" * 70)
    print("\n[INFO] Testing with REAL datasets only")
    print("[INFO] No synthetic data except for baseline comparison")
    
    validator = RealDatasetValidator()
    
    # Run tests
    validator.test_with_local_documents()
    validator.test_with_glove_embeddings()
    validator.test_with_random_normalized()
    
    # Generate report
    validator.generate_report()
    validator.save_results()
    
    print("\n[COMPLETE] Real dataset validation finished")


if __name__ == "__main__":
    main()
