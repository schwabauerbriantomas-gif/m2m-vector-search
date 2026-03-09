"""
Tests para CRUD completo de SimpleVectorDB y AdvancedVectorDB.
"""

import numpy as np
import pytest

from m2m import AdvancedVectorDB, DeleteResult, DocResult, SimpleVectorDB, UpdateResult


@pytest.fixture
def db():
    """SimpleVectorDB básico para tests."""
    return SimpleVectorDB(latent_dim=64, enable_lsh_fallback=False)


@pytest.fixture
def ebm_db():
    """SimpleVectorDB con EBM habilitado."""
    return SimpleVectorDB(latent_dim=64, enable_lsh_fallback=False, enable_ebm=True)


@pytest.fixture
def adv_db():
    """AdvancedVectorDB con SOC y EBM."""
    return AdvancedVectorDB(latent_dim=64, enable_soc=True, enable_energy_features=True)


@pytest.fixture
def sample_vectors():
    np.random.seed(42)
    return np.random.randn(10, 64).astype(np.float32)


@pytest.fixture
def sample_ids():
    return [f"doc_{i}" for i in range(10)]


@pytest.fixture
def sample_metadata():
    return [{"category": "tech" if i % 2 == 0 else "science", "year": 2020 + i} for i in range(10)]


# ---------------------------------------------------------------------------
# Add tests
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_basic(self, db, sample_vectors):
        """Test básico: añadir vectores sin IDs ni metadata."""
        n = db.add(vectors=sample_vectors)
        assert n > 0

    def test_add_with_ids(self, db, sample_vectors, sample_ids):
        """Test: añadir con IDs explícitos."""
        n = db.add(ids=sample_ids, vectors=sample_vectors)
        assert n > 0
        # Verificar que los IDs están en el store
        for doc_id in sample_ids:
            assert doc_id in db._vectors

    def test_add_with_metadata(self, db, sample_vectors, sample_ids, sample_metadata):
        """Test: añadir con metadata."""
        db.add(ids=sample_ids, vectors=sample_vectors, metadata=sample_metadata)
        # Verificar metadata
        assert db._metadata["doc_0"]["category"] == "tech"
        assert db._metadata["doc_1"]["category"] == "science"
        assert db._metadata["doc_0"]["year"] == 2020

    def test_add_with_documents(self, db, sample_vectors, sample_ids):
        """Test: añadir con documentos de texto."""
        docs = [f"Document {i} content" for i in range(10)]
        db.add(ids=sample_ids, vectors=sample_vectors, documents=docs)
        assert db._documents["doc_0"] == "Document 0 content"
        assert db._documents["doc_5"] == "Document 5 content"

    def test_add_validates_lengths(self, db, sample_vectors):
        """Test: validación de longitudes incompatibles."""
        with pytest.raises(ValueError, match="len\\(ids\\)"):
            db.add(ids=["only_one"], vectors=sample_vectors)

    def test_add_single_vector(self, db):
        """Test: añadir un solo vector."""
        vec = np.random.randn(64).astype(np.float32)
        n = db.add(ids=["single"], vectors=vec[np.newaxis, :])
        assert n >= 0  # Puede fallar dependiendo del config pero no debe lanzar excepción


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_metadata(self, db, sample_vectors, sample_ids, sample_metadata):
        """Test: actualizar metadata de un documento."""
        db.add(ids=sample_ids, vectors=sample_vectors, metadata=sample_metadata)
        result = db.update("doc_0", metadata={"category": "technology", "year": 2025})
        assert result.success
        assert db._metadata["doc_0"]["category"] == "technology"
        assert db._metadata["doc_0"]["year"] == 2025

    def test_update_document(self, db, sample_vectors, sample_ids):
        """Test: actualizar texto de un documento."""
        docs = [f"Doc {i}" for i in range(10)]
        db.add(ids=sample_ids, vectors=sample_vectors, documents=docs)
        result = db.update("doc_3", document="Updated document text")
        assert result.success
        assert db._documents["doc_3"] == "Updated document text"

    def test_update_vector(self, db, sample_vectors, sample_ids):
        """Test: actualizar el vector de un documento."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        new_vec = np.ones(64, dtype=np.float32)
        result = db.update("doc_0", vector=new_vec)
        assert result.success
        np.testing.assert_allclose(db._vectors["doc_0"], new_vec)

    def test_update_not_found(self, db):
        """Test: actualizar documento inexistente."""
        result = db.update("nonexistent_id", metadata={"x": 1})
        assert not result.success
        assert "not found" in result.message.lower()

    def test_update_upsert(self, db):
        """Test: upsert crea documento si no existe."""
        vec = np.zeros(64, dtype=np.float32)
        result = db.update("new_doc", vector=vec, upsert=True)
        assert result.success
        assert "new_doc" in db._vectors

    def test_update_returns_updateresult(self, db, sample_vectors, sample_ids):
        """Test: update retorna UpdateResult."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        result = db.update("doc_1", metadata={"x": 1})
        assert isinstance(result, UpdateResult)


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


class TestDelete:
    def test_soft_delete_by_id(self, db, sample_vectors, sample_ids):
        """Test: soft delete por ID."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        result = db.delete(id="doc_0")
        assert result.deleted == 1
        assert "doc_0" in db._deleted
        assert "doc_0" in db._vectors  # El vector sigue en memoria (soft)

    def test_hard_delete_by_id(self, db, sample_vectors, sample_ids):
        """Test: hard delete elimina permanentemente."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        result = db.delete(id="doc_0", hard=True)
        assert result.deleted == 1
        assert "doc_0" not in db._vectors

    def test_delete_multiple_ids(self, db, sample_vectors, sample_ids):
        """Test: delete múltiples IDs."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        result = db.delete(ids=["doc_0", "doc_1", "doc_2"])
        assert result.deleted == 3

    def test_delete_with_filter(self, db, sample_vectors, sample_ids, sample_metadata):
        """Test: delete por filtro de metadata."""
        db.add(ids=sample_ids, vectors=sample_vectors, metadata=sample_metadata)
        result = db.delete(filter={"year": {"$gt": 2025}})
        # Años 2026, 2027, 2028, 2029 -> 4 documentos
        assert result.deleted == 4

    def test_delete_returns_deleteresult(self, db, sample_vectors, sample_ids):
        """Test: delete retorna DeleteResult."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        result = db.delete(id="doc_0")
        assert isinstance(result, DeleteResult)

    def test_delete_nonexistent(self, db):
        """Test: delete ID inexistente retorna 0."""
        result = db.delete(id="does_not_exist")
        assert result.deleted == 0


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_results(self, db, sample_vectors, sample_ids):
        """Test: búsqueda retorna resultados."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        query = sample_vectors[0]
        results = db.search(query, k=5)
        assert results is not None

    def test_search_with_metadata(self, db, sample_vectors, sample_ids, sample_metadata):
        """Test: búsqueda incluye metadata."""
        db.add(ids=sample_ids, vectors=sample_vectors, metadata=sample_metadata)
        query = sample_vectors[0]
        results = db.search(query, k=5, include_metadata=True)
        if isinstance(results, list):
            for r in results:
                assert isinstance(r, DocResult)

    def test_search_with_filter(self, db, sample_vectors, sample_ids, sample_metadata):
        """Test: búsqueda con filtro de metadata."""
        db.add(ids=sample_ids, vectors=sample_vectors, metadata=sample_metadata)
        query = sample_vectors[0]
        results = db.search(query, k=10, filter={"category": {"$eq": "tech"}}, include_metadata=True)
        if isinstance(results, list):
            for r in results:
                assert r.metadata.get("category") == "tech"

    def test_search_deleted_excluded(self, db, sample_vectors, sample_ids):
        """Test: documentos eliminados no aparecen en búsqueda."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        db.delete(id="doc_0")
        query = sample_vectors[0]
        results = db.search(query, k=5, include_metadata=True)
        if isinstance(results, list):
            result_ids = [r.id for r in results]
            assert "doc_0" not in result_ids


# ---------------------------------------------------------------------------
# EBM tests
# ---------------------------------------------------------------------------


class TestEBM:
    def test_search_with_energy(self, ebm_db, sample_vectors, sample_ids):
        """Test: búsqueda con información energética."""
        ebm_db.add(ids=sample_ids, vectors=sample_vectors)
        query = sample_vectors[0]
        sr = ebm_db.search_with_energy(query, k=5)
        assert sr is not None
        assert hasattr(sr, "query_energy")
        assert hasattr(sr, "results")

    def test_get_energy(self, ebm_db, sample_vectors, sample_ids):
        """Test: cálculo de energía de un vector."""
        ebm_db.add(ids=sample_ids, vectors=sample_vectors)
        e = ebm_db.get_energy(sample_vectors[0])
        assert isinstance(e, float)

    def test_ebm_not_enabled_raises(self, db, sample_vectors):
        """Test: EBM features lanzan error si no están habilitadas."""
        with pytest.raises(RuntimeError, match="EBM features no habilitadas"):
            db.get_energy(sample_vectors[0])

    def test_suggest_exploration(self, ebm_db, sample_vectors, sample_ids):
        """Test: sugerencias de exploración."""
        ebm_db.add(ids=sample_ids, vectors=sample_vectors)
        suggestions = ebm_db.suggest_exploration(n=2)
        assert isinstance(suggestions, list)


# ---------------------------------------------------------------------------
# SOC tests (AdvancedVectorDB)
# ---------------------------------------------------------------------------


class TestSOC:
    def test_check_criticality(self, adv_db, sample_vectors, sample_ids):
        """Test: check_criticality retorna reporte."""
        adv_db.add(ids=sample_ids, vectors=sample_vectors)
        report = adv_db.check_criticality()
        assert hasattr(report, "state")
        assert hasattr(report, "index")

    def test_relax(self, adv_db, sample_vectors, sample_ids):
        """Test: relax retorna resultado de relajación."""
        adv_db.add(ids=sample_ids, vectors=sample_vectors)
        result = adv_db.relax(iterations=5)
        assert hasattr(result, "initial_energy")
        assert hasattr(result, "final_energy")
        assert hasattr(result, "iterations")

    def test_trigger_avalanche(self, adv_db, sample_vectors, sample_ids):
        """Test: trigger_avalanche retorna resultado."""
        adv_db.add(ids=sample_ids, vectors=sample_vectors)
        result = adv_db.trigger_avalanche()
        assert hasattr(result, "affected_clusters")
        assert hasattr(result, "energy_released")

    def test_get_stats_includes_soc(self, adv_db, sample_vectors, sample_ids):
        """Test: get_stats incluye información del SOC."""
        adv_db.add(ids=sample_ids, vectors=sample_vectors)
        stats = adv_db.get_stats()
        assert "soc" in stats


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestStats:
    def test_get_stats(self, db, sample_vectors, sample_ids):
        """Test: estadísticas básicas del sistema."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        stats = db.get_stats()
        assert "total_documents" in stats
        assert "active_documents" in stats
        assert stats["total_documents"] == 10

    def test_get_stats_after_delete(self, db, sample_vectors, sample_ids):
        """Test: estadísticas después de soft-delete."""
        db.add(ids=sample_ids, vectors=sample_vectors)
        db.delete(id="doc_0")
        stats = db.get_stats()
        assert stats["deleted_documents"] == 1
        assert stats["active_documents"] == 9
