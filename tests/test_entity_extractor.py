import numpy as np

from m2m.entity_extractor import EntityCandidate, M2MEntityExtractor
from m2m.graph_splat import GaussianGraphStore


class MockEmbeddingModel:
    def __init__(self, dim=640):
        self.dim = dim

    def encode(self, texts):
        return np.random.randn(len(texts), self.dim).astype(np.float32)


def test_extractor_without_dataset_transformer():
    """Valida funcionamiento SIN DatasetTransformer. Usa clustering ad-hoc en hiperesferas"""
    extractor = M2MEntityExtractor(
        use_structural_patterns=True,
        use_ngram_analysis=True,
        use_semantic_clustering=True,
    )

    embedding_model = MockEmbeddingModel(dim=640)

    text = """
    Apple Inc. reported quarterly earnings. 
    Microsoft Corporation announced Azure growth.
    Google LLC released new AI features.
    """

    entities = extractor.extract(
        text,
        embedding_model=embedding_model,
    )

    assert len(entities) > 0
    for entity in entities:
        assert 0 <= entity.score <= 1.0


def test_extractor_with_dataset_transformer():
    """Valida funcionamiento CON DatasetTransformer. Usa splats pre-computados"""
    extractor = M2MEntityExtractor()
    embedding_model = MockEmbeddingModel(dim=640)

    splat_data = {
        "mu": np.random.randn(50, 640).astype(np.float32),
        "alpha": np.random.uniform(0.01, 0.1, 50).astype(np.float32),
        "kappa": np.random.uniform(5.0, 20.0, 50).astype(np.float32),
    }

    text = "Apple Inc. reported strong earnings."

    entities = extractor.extract(
        text,
        embedding_model=embedding_model,
        splat_data=splat_data,
    )

    assert len(entities) > 0
    for entity in entities:
        if entity.embedding is not None and entity.cluster_id >= 0:
            assert entity.cluster_id < 50


def test_structural_patterns():
    """Test de patrones estructurales (emails, URLs, etc.)."""
    extractor = M2MEntityExtractor(
        use_ngram_analysis=False, use_semantic_clustering=False
    )
    text = "Contact me at test@example.com or visit https://example.com. Call 555-123-4567. Date: 2024-01-01, Price: $1,234.56"

    entities = extractor.extract(text)

    assert len(entities) == 5
    types = set(e.entity_type for e in entities)
    assert "contact" in types
    assert "url" in types
    assert "date" in types
    assert "money" in types


def test_ngram_analysis():
    """Test de análisis de n-grams."""
    extractor = M2MEntityExtractor(
        use_structural_patterns=False, use_semantic_clustering=False
    )
    text = "Steve Jobs founded Apple Computer in Cupertino."

    entities = extractor.extract(text)

    # Should find Steve Jobs, Apple Computer, Cupertino
    texts = [e.text for e in entities]
    assert "Steve Jobs" in texts
    assert "Apple Computer" in texts
    assert "Cupertino" in set(t.strip() for t in texts)


def test_semantic_validation():
    """Test de validación semántica en S^639."""
    extractor = M2MEntityExtractor(
        use_structural_patterns=False, use_ngram_analysis=False
    )
    # We will manually inject candidates just to test the validation method
    candidates = [
        EntityCandidate(
            text="Entity A",
            entity_type="test",
            score=0.5,
            embedding=np.random.randn(640).astype(np.float32),
        ),
        EntityCandidate(
            text="Entity B",
            entity_type="test",
            score=0.5,
            embedding=np.random.randn(640).astype(np.float32),
        ),
    ]

    # Normalize
    for c in candidates:
        norm = np.linalg.norm(c.embedding)
        c.embedding = c.embedding / norm

    splat_data = {"mu": np.random.randn(5, 640).astype(np.float32)}
    # Normalize splat centers
    splat_data["mu"] = splat_data["mu"] / np.linalg.norm(
        splat_data["mu"], axis=1, keepdims=True
    )

    extractor._validate_semantic(candidates, splat_data, None)

    for c in candidates:
        assert c.cluster_id >= 0
        assert c.cluster_id < 5


def test_integration_with_graph_store():
    """Test de integración con GaussianGraphStore."""
    from m2m.entity_extractor import M2MGraphEntityExtractor

    store = GaussianGraphStore(dim=640)
    extractor = M2MEntityExtractor()
    graph_extractor = M2MGraphEntityExtractor(extractor, store)

    embedding_model = MockEmbeddingModel(dim=640)
    doc_embedding = np.random.randn(640).astype(np.float32)

    doc_id = store.add_document("This is a document about Open AI.", doc_embedding)

    result = graph_extractor.extract_and_store(
        text="Open AI is building models.",
        doc_embedding=doc_embedding,
        doc_id=doc_id,
        embedding_model=embedding_model,
    )

    assert result["entities_found"] > 0
    assert result["entities_stored"] > 0
    assert result["relations_created"] > 0

    # Check that the graph is updated
    stats = store.get_stats()
    assert stats["total_nodes"] >= 2  # 1 doc + at least 1 entity
    assert stats["total_edges"] >= 1  # at least 1 relation
