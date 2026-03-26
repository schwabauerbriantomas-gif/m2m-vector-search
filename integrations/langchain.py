import uuid
from typing import Any, Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class M2MVectorStore(VectorStore):
    """
    M2M (Machine-to-Memory) integration for LangChain.
    """

    def __init__(
        self, embeddings: Embeddings, m2m_engine=None, config=None, **kwargs: Any
    ):
        """
        Initialize M2M VectorStore with embeddings and M2M config.
        """
        self._embeddings = embeddings
        self._config = config

        if m2m_engine is not None:
            self._m2m = m2m_engine
        else:
            try:
                from m2m import M2MConfig, M2MEngine

                if self._config is None:
                    self._config = M2MConfig(device="cpu")
                self._m2m = M2MEngine(self._config)
            except ImportError:
                raise ImportError(
                    "Could not import m2m python package. "
                    "Please ensure it is installed or run in the m2m directory."
                )

        # To store metadata and actual text of the splats since M2M is a feature index
        self._store = {}

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Embed documents and add them to M2M engine.
        """
        texts_list = list(texts)
        if len(texts_list) == 0:
            return []

        embeddings_list = self._embeddings.embed_documents(texts_list)

        # Convert embeddings to tensor batch
        emb_tensor = np.array(embeddings_list, dtype=np.float32)

        from m2m import normalize_sphere

        emb_tensor = normalize_sphere(emb_tensor)

        # Get start index for tracking
        start_idx = self._m2m.m2m.splats.n_active

        # Add to M2MEngine
        n_added = self._m2m.add_splats(emb_tensor)

        ids = []
        for i in range(n_added):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)

            # Store metadata based on insertion index
            meta = metadatas[i] if metadatas else {}
            self._store[start_idx + i] = {
                "id": doc_id,
                "text": texts_list[i],
                "metadata": meta,
            }

        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Return k most similar documents to the query.
        """
        query_emb = self._embeddings.embed_query(query)
        query_tensor = np.array(query_emb, dtype=np.float32)

        from m2m import normalize_sphere

        query_tensor = normalize_sphere(query_tensor)

        # We need to unsqueeze to make it a batch of 1
        query_batch = query_tensor[np.newaxis, :]

        # Run search using M2MEngine public API
        neighbors_mu, neighbors_alpha, neighbors_kappa = self._m2m.search(
            query_batch, k=k
        )

        # Get the top-k indices
        top_k_indices = np.argsort(neighbors_kappa[0])[-k:][::-1].tolist()

        results = []
        for idx in top_k_indices:
            if idx in self._store:
                doc_info = self._store[idx]
                results.append(
                    Document(
                        page_content=doc_info["text"], metadata=doc_info["metadata"]
                    )
                )

        return results

    def delete(self, ids: List[str]) -> None:
        """
        Remove documents by ID from the store.
        """
        if not ids:
            return
        id_set = set(ids)
        indices_to_remove = [
            idx for idx, info in self._store.items() if info["id"] in id_set
        ]
        for idx in indices_to_remove:
            del self._store[idx]

    def update(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Update existing documents. Matches by id and replaces text/metadata.
        """
        if ids is None:
            ids = [doc.metadata.get("id") for doc in documents]

        id_to_idx = {
            info["id"]: idx for idx, info in self._store.items()
        }
        id_to_doc = {doc_id: doc for doc_id, doc in zip(ids, documents)}

        for doc_id, doc in id_to_doc.items():
            if doc_id in id_to_idx:
                store_idx = id_to_idx[doc_id]
                self._store[store_idx]["text"] = doc.page_content
                self._store[store_idx]["metadata"] = doc.metadata

    def filter_by_metadata(
        self,
        metadata: dict,
        k: int = 4,
    ) -> List[Document]:
        """
        Search with metadata constraints. Returns only documents whose metadata
        contains all key-value pairs from the filter.
        """
        matching = []
        for idx, info in self._store.items():
            if all(
                info["metadata"].get(key) == value
                for key, value in metadata.items()
            ):
                matching.append(info)

        if len(matching) <= k:
            return [
                Document(page_content=m["text"], metadata=m["metadata"])
                for m in matching
            ]

        # Return first k matches
        return [
            Document(page_content=m["text"], metadata=m["metadata"])
            for m in matching[:k]
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        config=None,
        **kwargs: Any,
    ) -> "M2MVectorStore":
        """
        Create an M2MVectorStore and add initial texts.
        """
        store = cls(embeddings=embedding, config=config, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store
