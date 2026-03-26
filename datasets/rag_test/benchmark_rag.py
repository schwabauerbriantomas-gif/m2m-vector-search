"""
RAG Benchmark for M2M-Vector-Search
Runs predefined queries, reports precision@5 and recall@5, compares M2M vs linear scan.
"""
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(r"C:\Users\Brian\Desktop\m2m-vector-search-main")
DATASET_DIR = PROJECT_ROOT / "datasets" / "rag_test"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Predefined queries with expected relevant sources ──────────────
QUERIES = [
    {
        "query": "What is the Hierarchical Reasoning Model (HRM)?",
        "relevant_sources": ["HRM"],
        "keywords": ["hierarchical", "reasoning", "HRM"],
    },
    {
        "query": "How does flow matching work for generative modeling?",
        "relevant_sources": ["flow_matching"],
        "keywords": ["flow", "matching", "generative"],
    },
    {
        "query": "What are latent chain-of-thought techniques?",
        "relevant_sources": ["latent_cot"],
        "keywords": ["latent", "chain", "thought"],
    },
    {
        "query": "Langevin dynamics for score-based models",
        "relevant_sources": ["langevin"],
        "keywords": ["langevin", "score", "dynamics"],
    },
    {
        "query": "How does the tiny recursive model work?",
        "relevant_sources": ["tiny_recursive"],
        "keywords": ["tiny", "recursive", "model"],
    },
    {
        "query": "M2M vector search using Gaussian splats",
        "relevant_sources": ["README"],
        "keywords": ["m2m", "gaussian", "splat"],
    },
    {
        "query": "Looped language models and iterative computation",
        "relevant_sources": ["looped_language", "looped_transformers"],
        "keywords": ["looped", "language", "iterative"],
    },
    {
        "query": "Stochastic attention mechanisms in deep learning",
        "relevant_sources": ["stochastic_attention"],
        "keywords": ["stochastic", "attention"],
    },
    {
        "query": "Equilibrium matching for diffusion models",
        "relevant_sources": ["equilibrium_matching"],
        "keywords": ["equilibrium", "matching", "diffusion"],
    },
    {
        "query": "Robustness of score-based generative models",
        "relevant_sources": ["robustness_langevin", "langevin"],
        "keywords": ["robustness", "score", "generative"],
    },
    {
        "query": "Modern literature on language models 2025-2026",
        "relevant_sources": ["modern_literature"],
        "keywords": ["modern", "literature", "2025"],
    },
    {
        "query": "GPU and hardware optimization for language models",
        "relevant_sources": ["LMs_locales_hardware"],
        "keywords": ["gpu", "hardware", "optimization"],
    },
    {
        "query": "How to use the M2M API for vector search?",
        "relevant_sources": ["API"],
        "keywords": ["api", "search", "vector"],
    },
    {
        "query": "M2M cluster and distributed search architecture",
        "relevant_sources": ["cluster"],
        "keywords": ["cluster", "distributed", "architecture"],
    },
    {
        "query": "Performance benchmarks for M2M vector search",
        "relevant_sources": ["PERFORMANCE"],
        "keywords": ["benchmark", "performance", "speedup"],
    },
    {
        "query": "How does the M2M troubleshooting guide help?",
        "relevant_sources": ["TROUBLESHOOTING"],
        "keywords": ["troubleshoot", "error", "fix"],
    },
    {
        "query": "Deep dive into HRM mechanistic analysis",
        "relevant_sources": ["HRM_mechanistic_analysis"],
        "keywords": ["mechanistic", "analysis", "HRM"],
    },
    {
        "query": "Energy-based models and exploration",
        "relevant_sources": ["README"],
        "keywords": ["energy", "exploration", "ebm"],
    },
    {
        "query": "LangChain RAG integration with M2M",
        "relevant_sources": ["README", "API"],
        "keywords": ["langchain", "rag", "retriever"],
    },
    {
        "query": "Transformer architecture and attention mechanisms",
        "relevant_sources": ["looped_transformers", "modern_literature"],
        "keywords": ["transformer", "attention"],
    },
]


def load_dataset():
    embeddings = np.load(DATASET_DIR / "embeddings.npy")
    with open(DATASET_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    texts = []
    with open(DATASET_DIR / "texts.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return embeddings, metadata, texts


def linear_search(query_emb, embeddings, k=5):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    normed = embeddings / norms
    q_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
    sims = normed @ q_norm
    top_k = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_k]


def is_relevant(result_idx, metadata, texts, relevant_sources, keywords, k=5):
    """Check if a result is relevant based on source match or keyword overlap."""
    meta = metadata[result_idx]
    text = texts[result_idx].lower()
    
    # Source match
    src = meta["source"].lower()
    for rs in relevant_sources:
        if rs.lower() in src:
            return True
    
    # Keyword match (at least 1 keyword in text)
    kw_hits = sum(1 for kw in keywords if kw.lower() in text)
    if kw_hits >= 1:
        return True
    
    return False


def evaluate_query(query_data, embedder, embeddings, metadata, texts, k=5):
    q_emb = embedder.encode([query_data["query"]])[0]
    results = linear_search(q_emb, embeddings, k=k)
    
    relevant_count = 0
    for idx, sim in results:
        if is_relevant(idx, metadata, texts, query_data["relevant_sources"], query_data["keywords"]):
            relevant_count += 1
    
    precision = relevant_count / k
    recall = relevant_count / max(len(query_data["relevant_sources"]), 1)
    return {
        "query": query_data["query"][:60],
        "precision@k": precision,
        "recall@k": min(recall, 1.0),
        "relevant": relevant_count,
        "top_sim": results[0][1] if results else 0,
    }


def benchmark_linear(embedder, embeddings, metadata, texts, k=5):
    """Benchmark pure linear scan."""
    timings = []
    for qd in QUERIES:
        t0 = time.perf_counter()
        q_emb = embedder.encode([qd["query"]])[0]
        _ = linear_search(q_emb, embeddings, k=k)
        timings.append(time.perf_counter() - t0)
    return timings


def benchmark_m2m(embedder, embeddings, metadata, texts, k=5):
    """Benchmark M2M index if available."""
    try:
        from m2m import AdvancedVectorDB, M2MConfig
        config = M2MConfig(
            latent_dim=384,
            max_splats=len(embeddings),
            enable_vulkan=False,
        )
        db = AdvancedVectorDB(config)
        ids = [str(i) for i in range(len(embeddings))]
        db.add(ids=ids, vectors=embeddings)
        
        timings = []
        for qd in QUERIES:
            t0 = time.perf_counter()
            q_emb = embedder.encode([qd["query"]])[0]
            _ = db.search(q_emb, k=k)
            timings.append(time.perf_counter() - t0)
        return timings, db
    except Exception as e:
        print(f"  M2M not available: {e}")
        return None, None


def main():
    print("=" * 60)
    print("RAG Benchmark - M2M Vector Search")
    print("=" * 60)
    
    print("\n[1/4] Loading dataset...")
    embeddings, metadata, texts = load_dataset()
    print(f"  {len(embeddings)} chunks, dim={embeddings.shape[1]}")
    
    print("\n[2/4] Loading embedder...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"\n[3/4] Running {len(QUERIES)} queries (linear scan, k=5)...")
    k = 5
    results = []
    for qd in QUERIES:
        r = evaluate_query(qd, embedder, embeddings, metadata, texts, k=k)
        results.append(r)
        status = "✓" if r["precision@k"] >= 0.4 else "✗"
        print(f"  {status} P@{k}={r['precision@k']:.2f} R@{k}={r['recall@k']:.2f} | {r['query']}")
    
    # Aggregate metrics
    avg_p = np.mean([r["precision@k"] for r in results])
    avg_r = np.mean([r["recall@k"] for r in results])
    avg_sim = np.mean([r["top_sim"] for r in results])
    print(f"\n  ─── Average: P@{k}={avg_p:.3f}  R@{k}={avg_r:.3f}  top-sim={avg_sim:.3f}")
    
    print("\n[4/4] Benchmarking search speed...")
    linear_times = benchmark_linear(embedder, embeddings, metadata, texts, k=k)
    avg_linear = np.mean(linear_times) * 1000
    print(f"  Linear scan:  {avg_linear:.2f}ms avg (embedding included)")
    
    m2m_times, _ = benchmark_m2m(embedder, embeddings, metadata, texts, k=k)
    if m2m_times:
        avg_m2m = np.mean(m2m_times) * 1000
        print(f"  M2M index:    {avg_m2m:.2f}ms avg (embedding included)")
        print(f"  Speedup:      {avg_linear/avg_m2m:.1f}x")
    else:
        print("  M2M: not tested (import/config issue)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Queries:           {len(QUERIES)}")
    print(f"  Dataset size:      {len(embeddings)} chunks")
    print(f"  Precision@{k}:      {avg_p:.3f}")
    print(f"  Recall@{k}:         {avg_r:.3f}")
    print(f"  Avg top similarity:{avg_sim:.3f}")
    
    # Save results
    report = {
        "queries": len(QUERIES),
        "dataset_size": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "k": k,
        "avg_precision": float(avg_p),
        "avg_recall": float(avg_r),
        "avg_top_similarity": float(avg_sim),
        "linear_search_ms": float(avg_linear),
        "results": results,
    }
    out_path = DATASET_DIR / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
