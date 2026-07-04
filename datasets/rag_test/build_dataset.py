"""
RAG Test Dataset Builder for M2M-Vector-Search
Extracts text from PDFs and markdown files, chunks it, generates embeddings.
"""
import sys
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parents[2]
PDF_DIR = Path(os.environ.get("M2M_PDF_DIR", str(Path.home() / "papers")))
MD_DIRS = [
    PDF_DIR,
    PROJECT_ROOT,
]
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64
# Exclude dirs/files
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".github", "dist", "build"}
EXCLUDE_MD = {"CHANGELOG.md", "LESSONS.md", "SECURITY.md", "SPECS_M2M_IMPROVEMENTS.md", "SPECS_VALIDATION.md"}

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4

def chunk_text(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    """Split text into overlapping chunks by character approximation of tokens."""
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start + max_chars // 2, end + 200)
            if last_period > start:
                end = last_period + 1
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Skip tiny fragments
            chunks.append(chunk)
        start = end - overlap_chars if end < len(text) else end
    return chunks

def extract_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text from PDF and chunk it."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    chunks = []
    chunk_id = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():
            continue
        page_chunks = chunk_text(text)
        for ch in page_chunks:
            chunks.append({
                "text": ch,
                "source": str(pdf_path.name),
                "source_path": str(pdf_path),
                "page": page_num + 1,
                "chunk_id": chunk_id,
                "type": "pdf",
            })
            chunk_id += 1
    doc.close()
    return chunks

def extract_md(md_path: Path) -> List[Dict[str, Any]]:
    """Extract markdown by sections (headers)."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    chunks = []
    chunk_id = 0
    current_section = "intro"
    current_lines = []
    
    for line in lines:
        if line.startswith("#"):
            # Save previous section
            section_text = "\n".join(current_lines).strip()
            if len(section_text) > 50:
                section_chunks = chunk_text(section_text, max_tokens=600)
                for ch in section_chunks:
                    chunks.append({
                        "text": ch,
                        "source": str(md_path.relative_to(md_path.parent.parent) if md_path.parent.parent.exists() else md_path.name),
                        "source_path": str(md_path),
                        "header": current_section,
                        "chunk_id": chunk_id,
                        "type": "markdown",
                    })
                    chunk_id += 1
            current_section = line.lstrip("#").strip()[:80]
            current_lines = []
        else:
            current_lines.append(line)
    
    # Last section
    section_text = "\n".join(current_lines).strip()
    if len(section_text) > 50:
        section_chunks = chunk_text(section_text, max_tokens=600)
        for ch in section_chunks:
            chunks.append({
                "text": ch,
                "source": str(md_path.relative_to(md_path.parent.parent) if md_path.parent.parent.exists() else md_path.name),
                "source_path": str(md_path),
                "header": current_section,
                "chunk_id": chunk_id,
                "type": "markdown",
            })
            chunk_id += 1
    
    return chunks

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    
    # 1. Extract PDFs
    print("=== Extracting PDFs ===")
    for pdf_path in PDF_DIR.glob("**/*.pdf"):
        print(f"  {pdf_path.name}...")
        try:
            chunks = extract_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # 2. Extract Markdown files
    print("\n=== Extracting Markdown ===")
    for md_dir in MD_DIRS:
        for md_path in md_dir.glob("**/*.md"):
            if any(excl in md_path.parts for excl in EXCLUDE_DIRS):
                continue
            if md_path.name in EXCLUDE_MD:
                continue
            rel = md_path.name
            print(f"  {rel}...")
            try:
                chunks = extract_md(md_path)
                all_chunks.extend(chunks)
                print(f"    → {len(chunks)} chunks")
            except Exception as e:
                print(f"    ERROR: {e}")
    
    print(f"\n=== Total chunks: {len(all_chunks)} ===")
    
    # 3. Generate embeddings
    print("\n=== Generating embeddings (all-MiniLM-L6-v2) ===")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # 4. Validate numerical stability
    assert np.all(np.isfinite(embeddings)), "NaN/Inf in embeddings!"
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    # 5. Save
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    
    # Save metadata with text
    metadata = []
    for i, chunk in enumerate(all_chunks):
        meta = {k: v for k, v in chunk.items() if k != "text"}
        meta["index"] = i
        meta["char_count"] = len(chunk["text"])
        meta["est_tokens"] = estimate_tokens(chunk["text"])
        metadata.append(meta)
    
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save texts separately (metadata.json gets big enough)
    with open(OUTPUT_DIR / "texts.jsonl", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps({"index": chunk["chunk_id"], "text": chunk["text"]}, ensure_ascii=False) + "\n")
    
    # Save chunk list
    with open(OUTPUT_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Done! ===")
    print(f"  embeddings.npy: {embeddings.nbytes / 1024:.0f} KB")
    print(f"  metadata.json: {os.path.getsize(OUTPUT_DIR / 'metadata.json') / 1024:.0f} KB")
    print(f"  texts.jsonl: {os.path.getsize(OUTPUT_DIR / 'texts.jsonl') / 1024:.0f} KB")
    print(f"  Total chunks: {len(all_chunks)}")

if __name__ == "__main__":
    main()
