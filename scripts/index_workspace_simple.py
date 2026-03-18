import sys; sys.stdout.reconfigure(line_buffering=True)
import os
import time
from datetime import datetime
from pathlib import Path

# Setup paths
WORKSPACE = Path.home() / ".openclaw" / "workspace"
M2M_PROJECT = Path(r"C:\Users\Brian\Desktop\m2m-vector-search-main")
sys.path.insert(0, str(M2M_PROJECT / "src"))

INDEX_PATH = M2M_PROJECT / "alfred_index"

# Load sentence-transformers once (global)
print("Loading encoder...", flush=True)
from sentence_transformers import SentenceTransformer
_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
ENCODER_DIM = _encoder.get_sentence_embedding_dimension()

print(f"Encoder ready. Dimension: {ENCODER_DIM}", flush=True)

# Initialize AlfredMemoryDB
from m2m import AlfredMemoryDB

db = AlfredMemoryDB(
    encoder=lambda t: _encoder.encode(t, show_progress_bar=False),
    latent_dim=384,
    storage_path=str(INDEX_PATH),
    auto_categorize=True,
    temporal_decay=True,
    temporal_half_life_days=60.0,
    device="cpu",
    mode="standard",
)

print(f"DB initialized. {db.stats()}", flush=True)

def chunk_text(text, max_chars=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        # Try to break at paragraph
        last_nl = chunk.rfind("\n\n")
        if last_nl > max_chars // 2:
            chunk = chunk[:last_nl]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c and len(c) > 20]

# Find all files
all_files = []

# Workspace files
for ext in ["*.md", "*.py", "*.json"]:
    for f in WORKSPACE.rglob(ext):
        all_files.append(("workspace", f))
        
# M2M source
for ext in ["*.py"]:
    for f in (M2M_PROJECT / "src").rglob(ext):
        all_files.append(("m2m_source", f))

# Filter
skip_patterns = ["__pycache__", ".pyc", "node_modules", ".git", ".png", ".jpg", ".spv", "test_", "venv"]
all_files = [(s, f) for s, f in all_files if not any(p in str(f).lower() for p in skip_patterns)]

print(f"Found {len(all_files)} files to index", flush=True)

# Process each file
total_chunks = 0
start_time = time.time()

for source, file_path in all_files:
    try:
        rel_path = file_path.relative_to(WORKSPACE) if file_path.is_relative_to(WORKSPACE) else file_path.relative_to(M2M_PROJECT)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            text = f"[{rel_path}]\n{chunk}"
            
            # Store in DB
            db.store(text, {
                "source": str(rel_path),
                "chunk_index": i,
                "indexed_at": datetime.now().isoformat(),
            })
            total_chunks += 1
            
        if len(chunks) > 0:
            print(f"  {rel_path}: {len(chunks)} chunks", flush=True)
    except Exception as e:
        print(f"ERROR {file_path}: {e}", flush=True)

elapsed = time.time() - start_time
print(f"\nIndexing complete!", flush=True)
print(f"  Total chunks: {total_chunks}", flush=True)
print(f"  Time: {elapsed:.1f}s", flush=True)
print(f"  DB stats: {db.stats()}", flush=True)

# Save index
db.save()
print("Index saved.", flush=True)

# Test search
print("\nTesting search...", flush=True)
results = db.search("Brian prefiere respuestas directas", k=3)
for r in results:
    print(f"  Score: {r.score:.4f} | {r.doc[:80]}...", flush=True)
