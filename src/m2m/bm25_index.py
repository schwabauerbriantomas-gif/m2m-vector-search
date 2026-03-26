"""
BM25 Index for hybrid keyword + vector search.

Lightweight implementation using term frequency / inverse document frequency.
No external dependencies beyond Python stdlib + numpy.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class BM25Index:
    """
    BM25 (Okapi BM25) full-text search index.

    Indexes documents by tokenized text and supports fast keyword search.
    Designed for small-to-medium collections (AI agent memory scale: ~1-10K docs).

    Args:
        k1: Term frequency saturation parameter (default 1.5)
        b: Length normalization parameter (default 0.75)
        tokenizer_fn: Custom tokenizer. Default: lowercase word splitting.

    Example:
        >>> bm25 = BM25Index()
        >>> bm25.add("doc_1", "M2M vector search for semantic memory")
        >>> bm25.add("doc_2", "Brian decided to use M2M for semantic memory")
        >>> scores = bm25.search("M2M semantic", k=5)
        >>> # scores = [("doc_2", 0.85), ("doc_1", 0.72)]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer_fn=None):
        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer_fn or self._default_tokenizer

        # Document storage
        self._docs: Dict[str, str] = {}          # id -> raw text
        self._doc_tokens: Dict[str, List[str]] = {}  # id -> token list
        self._doc_lengths: Dict[str, int] = {}    # id -> token count
        self._doc_freq: Counter = Counter()        # term -> number of docs containing term
        self._term_freqs: Dict[str, Counter] = {}  # id -> term -> count in doc
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Default tokenizer: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-záéíóúñü0-9]+', text.lower())

    def add(self, doc_id: str, text: str):
        """
        Add a document to the BM25 index.

        Args:
            doc_id: Unique document identifier
            text: Document text to index
        """
        tokens = self._tokenizer(text)

        # Remove old doc if re-adding
        if doc_id in self._docs:
            self.remove(doc_id)

        self._docs[doc_id] = text
        self._doc_tokens[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)
        self._term_freqs[doc_id] = Counter(tokens)

        # Update document frequency
        unique_terms = set(tokens)
        for term in unique_terms:
            self._doc_freq[term] += 1

        self._n_docs += 1
        self._update_avg_dl()

    def remove(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document identifier to remove

        Returns:
            True if document existed and was removed
        """
        if doc_id not in self._docs:
            return False

        # Update document frequency
        for term in set(self._doc_tokens[doc_id]):
            self._doc_freq[term] -= 1
            if self._doc_freq[term] <= 0:
                del self._doc_freq[term]

        del self._docs[doc_id]
        del self._doc_tokens[doc_id]
        del self._doc_lengths[doc_id]
        del self._term_freqs[doc_id]
        self._n_docs -= 1
        self._update_avg_dl()
        return True

    def _update_avg_dl(self):
        """Recalculate average document length."""
        if self._n_docs > 0:
            self._avg_dl = sum(self._doc_lengths.values()) / self._n_docs
        else:
            self._avg_dl = 0.0

    def search(self, query: str, k: int = 10, doc_filter: Optional[set] = None) -> List[Tuple[str, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query text
            k: Maximum number of results
            doc_filter: Optional set of doc IDs to restrict search

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if self._n_docs == 0:
            return []

        query_tokens = self._tokenizer(query)
        scores: Dict[str, float] = defaultdict(float)

        N = self._n_docs
        avg_dl = self._avg_dl if self._avg_dl > 0 else 1.0

        for term in query_tokens:
            if term not in self._doc_freq:
                continue

            df = self._doc_freq[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf_counter in self._term_freqs.items():
                if doc_filter is not None and doc_id not in doc_filter:
                    continue

                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue

                dl = self._doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                scores[doc_id] += idf * numerator / denominator

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def search_ids(self, query: str, k: int = 10, doc_filter: Optional[set] = None) -> List[str]:
        """Search returning only doc IDs (no scores)."""
        return [doc_id for doc_id, _ in self.search(query, k, doc_filter)]

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return self._n_docs

    def clear(self):
        """Remove all documents from the index."""
        self._docs.clear()
        self._doc_tokens.clear()
        self._doc_lengths.clear()
        self._doc_freq.clear()
        self._term_freqs.clear()
        self._avg_dl = 0.0
        self._n_docs = 0

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def __len__(self) -> int:
        return self._n_docs
