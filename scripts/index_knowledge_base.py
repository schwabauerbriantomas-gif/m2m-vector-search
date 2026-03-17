"""
M2M Knowledge Base Indexer

Indexes all code, reports, research, and documentation into M2M vector search.
Uses sentence-transformers locally on GPU for embedding generation.
"""

import os
import sys
import json
import hashlib
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project to path
PROJECT = r"C:\Users\Brian\Desktop\m2m-vector-search-main"
sys.path.insert(0, os.path.join(PROJECT, "src"))

def get_files_to_index():
    """Collect all files worth indexing."""
    files = []
    
    # 1. M2M source code
    src_dir = os.path.join(PROJECT, "src")
    for ext in ["*.py"]:
        for f in Path(src_dir).rglob(ext):
            files.append((str(f), "m2m-code", f"m2m/{f.relative_to(src_dir)}"))
    
    # 2. M2M reports and documentation
    m2m_docs = [
        "ANALYSIS_REPORT.md", "PERFORMANCE_OPTIMIZATION.md",
        "benchmark_stats.md", "security_audit.md", "optimization_report.md",
        "chaos_report.md", "pentest_report.md", "research_findings.md",
        "README.md", "CHANGELOG.md", "CONTRIBUTING.md",
    ]
    for doc in m2m_docs:
        p = os.path.join(PROJECT, doc)
        if os.path.exists(p):
            files.append((p, "m2m-report", doc))
    
    # 3. OpenClaw workspace docs
    ws = r"C:\Users\Brian\.openclaw\workspace"
    ws_docs = ["MEMORY.md", "TOOLS.md", "SOUL.md", "USER.md", "AGENTS.md", "IDENTITY.md"]
    for doc in ws_docs:
        p = os.path.join(ws, doc)
        if os.path.exists(p):
            files.append((p, "openclaw-doc", doc))
    
    # 4. Memory files
    mem_dir = os.path.join(ws, "memory")
    if os.path.exists(mem_dir):
        for f in Path(mem_dir).glob("*.md"):
            files.append((str(f), "memory", f"memory/{f.name}"))
    
    # 5. Skills
    skills_dir = os.path.join(ws, "skills")
    if os.path.exists(skills_dir):
        for f in Path(skills_dir).rglob("*.py"):
            files.append((str(f), "skill-code", f"skills/{f.relative_to(skills_dir)}"))
        for f in Path(skills_dir).rglob("SKILL.md"):
            files.append((str(f), "skill-doc", f"skills/{f.relative_to(skills_dir)}"))
    
    # 6. Research and config files in workspace
    for f in Path(ws).glob("*.md"):
        rel = f.name
        if rel not in ws_docs and not rel.startswith("memory"):
            files.append((str(f), "workspace-doc", rel))
    
    return files


def read_file_safe(path):
    """Read file content safely."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"[Error reading file: {e}]"


def chunk_text(text, path, max_tokens=512):
    """Split text into chunks with metadata."""
    lines = text.split("\n")
    chunks = []
    current = []
    current_lines = 0
    header = f"[{path}]"
    
    for line in lines:
        current.append(line)
        current_lines += 1
        if current_lines >= max_tokens:
            chunks.append(header + "\n" + "\n".join(current))
            current = []
            current_lines = 0
    
    if current:
        chunks.append(header + "\n" + "\n".join(current))
    
    return chunks


def main():
    print("=" * 60)
    print("M2M Knowledge Base Indexer")
    print("=" * 60)
    
    # Collect files
    files = get_files_to_index()
    print(f"\nFound {len(files)} files to index")
    
    # Group by type
    by_type = {}
    for path, ftype, rel in files:
        by_type.setdefault(ftype, []).append((path, rel))
    for t, items in sorted(by_type.items()):
        print(f"  {t}: {len(items)} files")
    
    # Generate embeddings using sentence-transformers on GPU
    print("\nLoading embedding model (GPU)...")
    from sentence_transformers import SentenceTransformer
    
    model_name = "all-MiniLM-L6-v2"  # 384D, fast on GPU
    model = SentenceTransformer(model_name, device="cuda")
    dim = model.get_sentence_embedding_dimension()
    print(f"Model: {model_name}, dim={dim}, device=cuda")
    
    # Process all files
    all_chunks = []
    all_metadata = []
    
    for path, ftype, rel in files:
        content = read_file_safe(path)
        if not content or len(content.strip()) < 10:
            continue
        
        chunks = chunk_text(content, rel, max_tokens=200)
        for i, chunk in enumerate(chunks):
            file_hash = hashlib.md5(f"{rel}:{i}".encode()).hexdigest()[:12]
            all_chunks.append(chunk)
            all_metadata.append({
                "id": file_hash,
                "source": rel,
                "type": ftype,
                "path": path,
                "chunk_idx": i,
                "total_chunks": len(chunks),
                "indexed_at": datetime.now().isoformat(),
            })
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # Batch embed
    print("Generating embeddings on GPU...")
    t0 = time.perf_counter()
    embeddings = model.encode(all_chunks, batch_size=128, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    embed_time = time.perf_counter() - t0
    print(f"Embedded {len(all_chunks)} chunks in {embed_time:.1f}s ({len(all_chunks)/embed_time:.0f} chunks/s)")
    
    # Project to 640D if needed (M2M optimal dimension)
    if dim != 640:
        print(f"Projecting {dim}D -> 640D...")
        rng = np.random.RandomState(42)
        proj = rng.randn(dim, 640).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0)
        embeddings = embeddings @ proj
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-8, None)
    
    # Save index
    ws = r"C:\Users\Brian\.openclaw\workspace"
    index_dir = os.path.join(ws if os.path.exists(ws) else PROJECT, "knowledge_index")
    os.makedirs(index_dir, exist_ok=True)
    
    index_path = os.path.join(index_dir, "index_embeddings.npy")
    meta_path = os.path.join(index_dir, "index_metadata.json")
    
    np.save(index_path, embeddings.astype(np.float32))
    print(f"\nSaved embeddings: {index_path} ({os.path.getsize(index_path)/1e6:.1f} MB)")
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata: {meta_path} ({os.path.getsize(meta_path)/1e6:.1f} MB)")
    
    # Quick search test
    print("\n--- Search Test ---")
    query = "security vulnerabilities M2M"
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    if dim != 640:
        q_emb = q_emb @ proj
        q_emb = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-8, None)
    
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_emb, embeddings)[0]
    top5 = np.argsort(sims)[-5:][::-1]
    for idx in top5:
        m = all_metadata[idx]
        print(f"  [{sims[idx]:.3f}] {m['source']} ({m['type']}, chunk {m['chunk_idx']})")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
