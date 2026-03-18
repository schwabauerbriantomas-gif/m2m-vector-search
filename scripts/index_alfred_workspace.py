#!/usr/bin/env python3
"""
index_alfred_workspace.py — Index Alfred's workspace into AlfredMemoryDB

Indexes all .md, .py, .json files from:
- ~/.openclaw/workspace/ (SOUL.md, USER.md, TOOLS.md, AGENTS.md, etc.)
- ~/.openclaw/workspace/memory/ (daily memory files)
- ~/.openclaw/workspace/skills/ (skill documentation)
- M2M source code (if available)

Usage:
    python index_alfred_workspace.py [--storage-path ./indexed_memory] [--reindex]

Requires sentence-transformers:
    pip install sentence-transformers
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure M2M is importable
M2M_SRC = r"C:\Users\Brian\Desktop\m2m-vector-search-main\src"
if M2M_SRC not in sys.path:
    sys.path.insert(0, M2M_SRC)

WORKSPACE = Path.home() / ".openclaw" / "workspace"
M2M_PROJECT = Path(r"C:\Users\Brian\Desktop\m2m-vector-search-main")


def get_encoder():
    """Load sentence-transformers encoder. Uses bge-small-en-v1.5 (best all-rounder per Z.AI research)."""
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"Loading encoder: {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"Encoder loaded. Dimension: {model.get_sentence_embedding_dimension()}")

        def encode(text):
            if isinstance(text, list):
                return model.encode(text, show_progress_bar=False, batch_size=64)
            return model.encode(text, show_progress_bar=False)

        return encode, model.get_sentence_embedding_dimension()
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading encoder: {e}")
        sys.exit(1)


def find_files(base_path: Path, extensions: Set[str]) -> List[Path]:
    """Find all files with given extensions recursively."""
    files = []
    if not base_path.exists():
        return files
    for ext in extensions:
        files.extend(base_path.rglob(f"*{ext}"))
    # Sort for deterministic ordering
    return sorted(set(files))


def classify_file(file_path: Path) -> Dict:
    """Classify a file and return metadata."""
    rel = file_path
    try:
        rel = file_path.relative_to(WORKSPACE)
    except ValueError:
        try:
            rel = file_path.relative_to(M2M_PROJECT)
        except ValueError:
            pass

    rel_str = str(rel).replace("\\", "/")

    # Determine type and category
    meta = {
        "source": rel_str,
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Type classification
    suffix = file_path.suffix.lower()
    if suffix == ".md":
        meta["file_type"] = "markdown"
        # Category based on path
        parts = rel_str.lower().split("/")
        if "memory" in parts:
            meta["category"] = "memory"
        elif "soul" in rel_str:
            meta["category"] = "identity"
        elif "user" in rel_str:
            meta["category"] = "user_info"
        elif "tools" in rel_str:
            meta["category"] = "configuration"
        elif "agents" in rel_str:
            meta["category"] = "configuration"
        elif "skills" in parts:
            meta["category"] = "skill"
        elif "readme" in rel_str:
            meta["category"] = "documentation"
        else:
            meta["category"] = "note"
    elif suffix == ".py":
        meta["file_type"] = "python"
        meta["category"] = "code"
    elif suffix == ".json":
        meta["file_type"] = "json"
        meta["category"] = "data"
    elif suffix == ".txt":
        meta["file_type"] = "text"
        meta["category"] = "note"
    else:
        meta["file_type"] = suffix
        meta["category"] = "other"

    # File stats
    stat = file_path.stat()
    meta["size_bytes"] = stat.st_size
    meta["modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Content hash for change detection
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        meta["content_hash"] = hashlib.md5(content.encode()).hexdigest()
    except Exception:
        meta["content_hash"] = "error"

    return meta


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better search granularity.

    Large files are chunked so searches can find specific sections.
    Research: chunking improves retrieval for long documents.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars

        # Try to break at a paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + max_chars // 2:
                end = para_break + 2

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def should_skip(file_path: Path) -> bool:
    """Check if a file should be skipped."""
    rel_str = str(file_path).replace("\\", "/").lower()
    skip_patterns = [
        "__pycache__",
        ".pyc",
        "node_modules",
        ".git/",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
        ".pyc",
        ".exe", ".dll", ".so", ".dylib",
        "venv/", ".venv/",
        ".spv",  # Shader binaries
        "dist/",
    ]
    return any(p in rel_str for p in skip_patterns)


def index_workspace(
    storage_path: str,
    reindex: bool = False,
    batch_size: int = 50,
) -> Dict:
    """
    Index the entire Alfred workspace into AlfredMemoryDB.

    Args:
        storage_path: Where to store the index
        reindex: If True, delete existing index and start fresh
        batch_size: Number of documents to process per batch

    Returns:
        Stats dict with counts and timing
    """
    from m2m import AlfredMemoryDB

    t0 = time.time()

    # Load encoder
    encoder, dim = get_encoder()

    # Check for existing index state
    state_file = Path(storage_path) / "index_state.json"
    if state_file.exists() and not reindex:
        with open(state_file, "r") as f:
            old_state = json.load(f)
        print(f"Existing index found with {old_state.get('total_files', 0)} files.")
        print("Use --reindex to start fresh, or files will be re-indexed if changed.")
    else:
        old_state = {}

    # Create DB
    db = AlfredMemoryDB(
        encoder=encoder,
        latent_dim=dim,
        storage_path=storage_path,
        auto_categorize=True,
        temporal_decay=True,
        temporal_half_life_days=60.0,  # Workspace docs decay slowly
    )

    # Find all files
    extensions = {".md", ".py", ".json", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
    all_files: List[Tuple[Path, str]] = []  # (file_path, source_label)

    # Workspace files
    for f in find_files(WORKSPACE, extensions):
        if not should_skip(f):
            all_files.append((f, "workspace"))

    # M2M project source files
    src_path = M2M_PROJECT / "src"
    for f in find_files(src_path, extensions):
        if not should_skip(f):
            all_files.append((f, "m2m_source"))

    print(f"\nFound {len(all_files)} files to index.")

    # Process files
    stats = {
        "total_files": len(all_files),
        "indexed_files": 0,
        "skipped_unchanged": 0,
        "total_chunks": 0,
        "total_bytes": 0,
        "errors": [],
    }

    texts_batch = []
    metas_batch = []
    ids_batch = []

    def flush_batch():
        nonlocal texts_batch, metas_batch, ids_batch
        if not texts_batch:
            return
        db.batch_store(texts_batch, metas_batch, ids_batch)
        stats["indexed_files"] += 1
        stats["total_chunks"] += len(texts_batch)
        texts_batch = []
        metas_batch = []
        ids_batch = []

    for i, (file_path, source) in enumerate(all_files):
        try:
            meta = classify_file(file_path)
            meta["source_project"] = source

            # Check if file changed
            old_hash = old_state.get(meta["source"], {}).get("hash")
            if old_hash == meta["content_hash"] and not reindex:
                stats["skipped_unchanged"] += 1
                continue

            # Read content
            content = file_path.read_text(encoding="utf-8", errors="replace")
            stats["total_bytes"] += meta["size_bytes"]

            # Chunk if large
            chunks = chunk_text(content)
            for j, chunk in enumerate(chunks):
                chunk_id = f"{meta['source']}::chunk_{j}"
                chunk_meta = dict(meta)
                chunk_meta["chunk_index"] = j
                chunk_meta["total_chunks"] = len(chunks)

                # Prepend file path for context
                text = f"[{meta['source']}]\n{chunk}"

                texts_batch.append(text)
                metas_batch.append(chunk_meta)
                ids_batch.append(chunk_id)

            if len(texts_batch) >= batch_size:
                flush_batch()

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(all_files)} files...")

        except Exception as e:
            stats["errors"].append(f"{file_path}: {e}")

    flush_batch()

    # Save index state
    elapsed = time.time() - t0
    new_state = {
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(all_files),
        "indexed_files": stats["indexed_files"],
        "skipped_unchanged": stats["skipped_unchanged"],
        "total_chunks": stats["total_chunks"],
        "elapsed_seconds": round(elapsed, 1),
        "files": {
            classify_file(f)["source"]: {"hash": classify_file(f).get("content_hash")}
            for f, _ in all_files
        },
    }

    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(new_state, f, indent=2)

    # Print summary
    db_stats = db.stats()
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"  Files scanned:    {stats['total_files']}")
    print(f"  Files indexed:    {stats['indexed_files']}")
    print(f"  Skipped (cached): {stats['skipped_unchanged']}")
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Total bytes:      {stats['total_bytes']:,}")
    print(f"  Errors:           {len(stats['errors'])}")
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print(f"\n  DB Stats:")
    print(f"    Total memories: {db_stats['total_memories']}")
    print(f"    BM25 indexed:   {db_stats['bm25_indexed']}")
    print(f"    Fusion method:  {db_stats['fusion_method']}")
    print(f"    Temporal decay: {db_stats['temporal_decay']}")
    print(f"    Storage:        {db_stats['storage_path']}")
    print(f"{'='*60}")

    if stats["errors"]:
        print(f"\n⚠️  Errors ({len(stats['errors'])}):")
        for err in stats["errors"][:10]:
            print(f"  - {err}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Alfred's workspace into AlfredMemoryDB")
    parser.add_argument("--storage-path", default="./alfred_indexed_memory", help="Storage path for the index")
    parser.add_argument("--reindex", action="store_true", help="Force re-index all files")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for indexing")
    args = parser.parse_args()

    index_workspace(
        storage_path=args.storage_path,
        reindex=args.reindex,
        batch_size=args.batch_size,
    )
