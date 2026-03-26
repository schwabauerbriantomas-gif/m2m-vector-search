"""Full LangChain integration tests for M2M."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

pytest.importorskip("langchain_core")
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from integrations.langchain import M2MVectorStore
from m2m import M2MConfig


@pytest.fixture
def config():
    return M2MConfig(device="cpu", latent_dim=64)


@pytest.fixture
def embeddings():
    return FakeEmbeddings(size=64)


@pytest.fixture
def vectorstore(embeddings, config):
    return M2MVectorStore(embeddings=embeddings, config=config)


class TestCRUDLifecycle:
    """Full CRUD lifecycle: add → search → update → search → delete → verify gone."""

    def test_full_lifecycle(self, vectorstore):
        # Add
        texts = [
            "Machine learning is a subset of AI.",
            "Neural networks are powerful models.",
            "Python is great for data science.",
        ]
        metadatas = [{"topic": "ai"}, {"topic": "ai"}, {"topic": "programming"}]
        ids = vectorstore.add_texts(texts, metadatas=metadatas)
        assert len(ids) == 3

        # Search
        results = vectorstore.similarity_search("machine learning", k=2)
        assert len(results) == 2

        # Update
        updated_docs = [
            Document(
                page_content="Deep learning is a subset of machine learning.",
                metadata={"topic": "ai", "id": ids[0]},
            )
        ]
        vectorstore.update(updated_docs, ids=[ids[0]])

        # Search again — should find updated text
        results = vectorstore.similarity_search("deep learning", k=3)
        contents = [r.page_content for r in results]
        assert any("Deep learning" in c for c in contents)

        # Delete
        vectorstore.delete([ids[0]])
        results = vectorstore.similarity_search("deep learning", k=3)
        contents = [r.page_content for r in results]
        assert not any("Deep learning" in c for c in contents)

    def test_delete_nonexistent(self, vectorstore):
        """Deleting nonexistent IDs should not raise."""
        vectorstore.delete(["nonexistent-id-1", "nonexistent-id-2"])

    def test_update_nonexistent(self, vectorstore):
        """Updating nonexistent IDs should be a no-op."""
        doc = Document(page_content="new", metadata={"id": "nope"})
        vectorstore.update([doc], ids=["nope"])


class TestMetadataFilter:
    def test_filter_by_metadata(self, vectorstore):
        texts = [
            "Python tutorial",
            "JavaScript guide",
            "Python advanced",
            "Rust reference",
        ]
        metadatas = [
            {"lang": "python"},
            {"lang": "javascript"},
            {"lang": "python"},
            {"lang": "rust"},
        ]
        vectorstore.add_texts(texts, metadatas=metadatas)

        results = vectorstore.filter_by_metadata({"lang": "python"}, k=10)
        assert len(results) == 2
        for r in results:
            assert r.metadata["lang"] == "python"

        results = vectorstore.filter_by_metadata({"lang": "rust"}, k=10)
        assert len(results) == 1
        assert "Rust" in results[0].page_content

    def test_filter_no_match(self, vectorstore):
        vectorstore.add_texts(["hello"], metadatas=[{"x": 1}])
        results = vectorstore.filter_by_metadata({"x": 2}, k=10)
        assert len(results) == 0

    def test_filter_multiple_keys(self, vectorstore):
        vectorstore.add_texts(
            ["doc1", "doc2"],
            metadatas=[{"a": 1, "b": 2}, {"a": 1, "b": 3}],
        )
        results = vectorstore.filter_by_metadata({"a": 1, "b": 2}, k=10)
        assert len(results) == 1
        assert results[0].page_content == "doc1"


class TestAsRetriever:
    def test_retriever_interface(self, vectorstore):
        vectorstore.add_texts(
            ["The cat sat on the mat.", "Dogs are loyal animals."],
            metadatas=[{"topic": "cat"}, {"topic": "dog"}],
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        results = retriever.invoke("cat")
        assert len(results) == 1
        assert "cat" in results[0].page_content.lower()

    def test_retriever_in_chain(self, vectorstore):
        """Test vectorstore works as a retriever in a simple chain."""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
        except ImportError:
            pytest.skip("langchain_core.runnables not available")

        vectorstore.add_texts(
            ["Paris is the capital of France.", "Berlin is the capital of Germany."],
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        def format_docs(docs):
            return "\n".join(d.page_content for d in docs)

        # Build a simple retrieval chain (no LLM needed — just test the retriever works)
        chain = {"docs": retriever} | RunnablePassthrough()
        result = chain.invoke("capital of France")

        assert "docs" in result
        assert len(result["docs"]) == 1
        assert "Paris" in result["docs"][0].page_content


class TestM2MMemory:
    """Test M2MMemory store → recall → verify relevance."""

    def test_memory_store_recall(self, vectorstore):
        # Store memories
        memories = [
            "User prefers dark mode in IDEs.",
            "User works with Python and Rust.",
            "User likes keyboard shortcuts over mouse.",
        ]
        ids = vectorstore.add_texts(memories)
        assert len(ids) == 3

        # Recall relevant memory
        results = vectorstore.similarity_search("coding preferences", k=2)
        assert len(results) >= 1

        contents = [r.page_content for r in results]
        # At least one should be about programming/coding preferences
        assert any("Python" in c or "shortcuts" in c for c in contents)

    def test_memory_relevance_ordering(self, vectorstore):
        texts = [
            "Alpha Centauri is a star system.",
            "Baking bread requires yeast.",
            "Neural networks use backpropagation for training.",
            "Support vector machines find optimal hyperplanes.",
        ]
        vectorstore.add_texts(texts)

        results = vectorstore.similarity_search("machine learning algorithms", k=4)
        # ML-related docs should rank higher
        contents = [r.page_content for r in results]
        # Top result should be ML-related
        assert any(
            word in contents[0].lower()
            for word in ["neural", "support vector", "backpropagation", "hyperplane"]
        ), f"Expected ML-related top result, got: {contents[0]}"
