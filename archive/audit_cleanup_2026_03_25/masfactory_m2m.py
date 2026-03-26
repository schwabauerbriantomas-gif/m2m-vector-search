"""
MASFactory + M2M Integration
=================================================
Knowledge base semántica para agentes MASFactory.
Usa sentence-transformers para embeddings semánticos reales.
Fallback a TF-IDF si sentence-transformers no está disponible.
"""

import sys
import os
import json
import math
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from collections import Counter

M2M_PATH = Path(r"C:\Users\Brian\Desktop\m2m-vector-search-main\src")
sys.path.insert(0, str(M2M_PATH))


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def partial_fit(self, texts: list):
        pass  # Not needed for sentence-transformers
    
    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode(text)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    
    def embed_batch(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


class TFIDFEmbedder:
    """Fallback TF-IDF embedder when sentence-transformers is not available."""
    
    STOPWORDS = {"de", "la", "el", "en", "un", "una", "que", "es", "por", "con", "para",
                 "del", "las", "los", "se", "al", "no", "su", "y", "o", "a", "lo",
                 "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
                 "for", "of", "and", "or", "not", "with", "by", "from", "it", "this",
                 "que", "como", "pero", "mas", "hay", "tiene", "ser", "fue", "son",
                 "can", "has", "had", "be", "been", "do", "does", "did", "will",
                 "se", "un", "una", "es", "lo", "su", "las", "los", "del", "en",
                 "if", "but", "also", "than", "then", "when", "where", "which",
                 "error", "fix", "bug", "use", "using", "used", "make", "set", "get"}
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vocab = {}  # word -> hash index
        self.idf = {}    # word -> idf score
        self.doc_freq = Counter()
        self.n_docs = 0
    
    def _tokenize(self, text: str) -> list:
        text = text.lower()
        # Extract words, numbers, and technical terms
        tokens = re.findall(r'[a-záéíóúñü]+|[0-9]+|#[a-z0-9_]+|\.[a-z]+', text)
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]
    
    def _word_to_idx(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = hash(word) % self.dim
        return self.vocab[word]
    
    def _compute_idf(self):
        self.idf = {}
        for word, freq in self.doc_freq.items():
            self.idf[word] = math.log((self.n_docs + 1) / (freq + 1)) + 1
    
    def partial_fit(self, texts: list):
        """Update IDF statistics with new documents (call before embed)."""
        for text in texts:
            tokens = set(self._tokenize(text))
            self.doc_freq.update(tokens)
            self.n_docs += 1
        self._compute_idf()
    
    def embed(self, text: str) -> np.ndarray:
        """Embed text to vector of shape (dim,)."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        
        vec = np.zeros(self.dim, dtype=np.float32)
        for word, count in tf.items():
            idx = self._word_to_idx(word)
            tfidf = count * self.idf.get(word, 1.0)
            vec[idx] += tfidf
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    
    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed multiple texts. Returns (n, dim) array."""
        self.partial_fit(texts)
        return np.stack([self.embed(t) for t in texts])


class MASFactoryKB:
    """Knowledge base semántica para flujos MASFactory con M2M + cosine similarity."""
    
    def __init__(self, latent_dim: int = 384, device: str = "cpu", persist_dir: Optional[str] = None,
                 use_sentence_transformers: bool = True, st_model: str = "all-MiniLM-L6-v2"):
        self.latent_dim = latent_dim
        self.device = device
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.embedding_matrix = np.empty((0, latent_dim), dtype=np.float32)
        self.documents = []
        
        if use_sentence_transformers:
            try:
                self.embedder = SentenceTransformerEmbedder(model_name=st_model, device=device)
                actual_dim = self.embedder.dim
                if actual_dim != latent_dim:
                    print(f"[MASFactory KB] Auto-adjusting latent_dim {latent_dim} -> {actual_dim} (from {st_model})")
                    self.latent_dim = actual_dim
                    self.embedding_matrix = np.empty((0, actual_dim), dtype=np.float32)
            except Exception as e:
                print(f"[MASFactory KB] sentence-transformers failed ({e}), falling back to TF-IDF")
                self.embedder = TFIDFEmbedder(dim=latent_dim)
        else:
            self.embedder = TFIDFEmbedder(dim=latent_dim)
    
    def set_embedder(self, embed_fn: Callable):
        """Custom embedder: fn(str) -> np.ndarray of shape (latent_dim,)."""
        test = np.array(embed_fn("test"), dtype=np.float32).flatten()
        if test.shape[0] != self.latent_dim:
            raise ValueError(f"Embedding dim {test.shape[0]} != {self.latent_dim}")
        self.embedder = embed_fn
    
    def index(self, role: str, content: str, metadata: dict = None) -> int:
        """Indexar output de un agente."""
        self.embedder.partial_fit([content])
        emb = self.embedder.embed(content).reshape(1, -1).astype(np.float32)
        
        idx = len(self.documents)
        self.embedding_matrix = np.vstack([self.embedding_matrix, emb]) if self.embedding_matrix.shape[0] > 0 else emb
        
        self.documents.append({
            "idx": idx,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        return idx
    
    def query(self, question: str, k: int = 5, role_filter: str = None) -> list:
        """Buscar documentos relevantes por cosine similarity sobre embeddings."""
        if self.embedding_matrix.shape[0] == 0:
            return []
        
        q_emb = self.embedder.embed(question).reshape(1, -1).astype(np.float32)
        
        # Cosine similarity (embeddings are already normalized)
        scores = (self.embedding_matrix @ q_emb.T).flatten()
        
        # Sort descending
        ranked = np.argsort(-scores)
        
        results = []
        for doc_idx in ranked:
            doc = self.documents[int(doc_idx)]
            if role_filter and doc["role"] != role_filter:
                continue
            results.append({
                "idx": int(doc_idx),
                "role": doc["role"],
                "content": doc["content"],
                "score": float(scores[doc_idx]),
                "metadata": doc["metadata"]
            })
            if len(results) >= k:
                break
        return results
    
    def get_by_role(self, role: str) -> list:
        return [d for d in self.documents if d["role"] == role]
    
    def get_context_for_agent(self, agent_role: str, task: str, k: int = 5) -> str:
        """Generar contexto relevante para un agente basado en su tarea.
        
        Returns formatted string con los documentos más relevantes.
        """
        results = self.query(task, k=k)
        if not results:
            return ""
        
        lines = ["## Contexto relevante del Knowledge Base (M2M)\n"]
        for r in results:
            lines.append(f"### [{r['role']}] (score={r['score']:.3f})")
            # Truncate to avoid blowing up context
            content = r['content'][:1500]
            if len(r['content']) > 1500:
                content += "..."
            lines.append(content)
            lines.append("")
        return "\n".join(lines)
    
    def summary(self) -> str:
        roles = {}
        for d in self.documents:
            roles[d["role"]] = roles.get(d["role"], 0) + 1
        lines = [f"MASFactory KB (M2M): {len(self.documents)} documentos, dim={self.latent_dim}"]
        for role, count in sorted(roles.items()):
            lines.append(f"  {role}: {count}")
        return "\n".join(lines)
    
    def save(self, path: str = None):
        path = path or str(self.persist_dir / "kb_docs.json") if self.persist_dir else None
        if not path:
            raise ValueError("No persist_dir/path")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        # Save embeddings and embedder
        np.save(str(Path(path).parent / "kb_embeddings.npy"), self.embedding_matrix)
        emb_path = str(Path(path).parent / "kb_embedder.json")
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump({
                "dim": self.latent_dim,
                "vocab": self.embedder.vocab,
                "idf": self.embedder.idf,
                "doc_freq": dict(self.embedder.doc_freq),
                "n_docs": self.embedder.n_docs
            }, f)
    
    def load(self, path: str = None) -> int:
        path = path or str(self.persist_dir / "kb_docs.json") if self.persist_dir else None
        if not path or not Path(path).exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        emb_path = str(Path(path).parent / "kb_embedder.json")
        if Path(emb_path).exists():
            with open(emb_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.embedder.vocab = state["vocab"]
            self.embedder.idf = state["idf"]
            self.embedder.doc_freq = Counter(state["doc_freq"])
            self.embedder.n_docs = state["n_docs"]
        
        npy_path = str(Path(path).parent / "kb_embeddings.npy")
        if Path(npy_path).exists():
            self.embedding_matrix = np.load(npy_path)
        else:
            # Rebuild
            self.embedding_matrix = np.empty((0, self.latent_dim), dtype=np.float32)
        
        self.documents = docs
        return len(self.documents)


if __name__ == "__main__":
    print("Loading sentence-transformers model...")
    kb = MASFactoryKB()  # Auto-detects sentence-transformers, falls back to TF-IDF
    
    # Simular outputs de agentes
    kb.index("architect", 
        "SAM Labeler tiene acoplamiento fuerte entre KeyHandler y SAMLabeler por duck-typing. "
        "Se recomienda crear un Protocolo formal con los atributos esperados. "
        "También se detecto que get_image() recarga desde disco en cada frame del loop, "
        "causando 30 I/O reads por segundo sin cache.", 
        {"phase": 1, "priority": "P0"})
    
    kb.index("qa", 
        "Bug HIGH: cv2.fillPoly con alpha no funciona en OpenCV BGR. La funcion fillPoly "
        "ignora el 4to valor RGBA. Solucion: usar color solido sin alpha. "
        "Bug HIGH: Modo F no muestra overlay visual al usuario indicando que esta "
        "seleccionando referencia. Bug MEDIUM: input() en goto congela la GUI.", 
        {"phase": 1, "priority": "HIGH"})
    
    kb.index("security", 
        "Atomic save de annotations.json es must-fix. Un crash durante json.dump corrompe "
        "el archivo permanentemente sin backup. Mitigacion: escribir a .tmp y rename. "
        "GPU OOM en compute_tile_features: batch_size 64 puede exceder VRAM. "
        "Validacion de parametros de config: downsample>0, confidence en [0,1].", 
        {"phase": 1, "priority": "MEDIUM"})
    
    kb.index("implementor",
        "Fixes aplicados: image cache en TileManager, _segment_and_store unificado, "
        "fillPoly sin alpha, overlay modo F, goto numerico 1-9, atexit handler, "
        "atomic save .tmp->rename->.bak, config validation, GPU OOM fallback.", 
        {"phase": 2})
    
    print(kb.summary())
    print()
    
    # Queries
    queries = [
        "que problemas tiene el renderer?",
        "como se guarda el archivo de anotaciones?",
        "que bugs encontro QA en la interfaz?"
    ]
    
    for q in queries:
        results = kb.query(q, k=2)
        print(f"Query: {q}")
        for r in results:
            print(f"  [{r['role']}] score={r['score']:.4f}: {r['content'][:100]}...")
        print()
    
    # Context for agent example
    ctx = kb.get_context_for_agent("critic", "revisar fixes del renderer y save", k=3)
    print("=== Contexto para Critic ===")
    print(ctx)
