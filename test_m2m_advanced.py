"""
Pytest Suite - M2M Vector Search
Tests completos para todas las funcionalidades avanzadas.

Run with:
    pytest test_m2m_advanced.py -v
    pytest test_m2m_advanced.py -v -k "gpu"  # Solo GPU tests
    pytest test_m2m_advanced.py -v -k "cache"  # Solo cache tests
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from m2m import SimpleVectorDB, AdvancedVectorDB, M2MConfig
from m2m.gpu_auto_tune import GPUAutoTuner, GPUMemoryPool, get_gpu_tuner
from m2m.query_optimizer import QueryCache, QueryPrefetcher, QueryOptimizer
from m2m.auto_scaling import AutoScaler, MetricsCollector, HorizontalScaler, NodeMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_vectors():
    """Genera vectores de prueba."""
    np.random.seed(42)
    return np.random.randn(1000, 640).astype(np.float32)


@pytest.fixture
def sample_query():
    """Genera query de prueba."""
    np.random.seed(42)
    return np.random.randn(640).astype(np.float32)


@pytest.fixture
def simple_db():
    """Crea SimpleVectorDB para tests."""
    db = SimpleVectorDB(latent_dim=640, mode='edge')
    return db


@pytest.fixture
def advanced_db():
    """Crea AdvancedVectorDB para tests."""
    db = AdvancedVectorDB(latent_dim=640, enable_soc=True, enable_energy_features=True)
    return db


# =============================================================================
# Core Functionality Tests
# =============================================================================

class TestSimpleVectorDB:
    """Tests para SimpleVectorDB."""
    
    def test_add_vectors(self, simple_db, sample_vectors):
        """Test añadir vectores."""
        ids = [f"doc_{i}" for i in range(len(sample_vectors))]
        n = simple_db.add(ids=ids, vectors=sample_vectors)
        
        assert n == len(sample_vectors)
        stats = simple_db.get_stats()
        assert stats['total_documents'] == len(sample_vectors)
    
    def test_search_basic(self, simple_db, sample_vectors, sample_query):
        """Test búsqueda básica."""
        ids = [f"doc_{i}" for i in range(100)]
        simple_db.add(ids=ids, vectors=sample_vectors[:100])
        
        results = simple_db.search(sample_query, k=10)
        
        # Legacy mode returns tuple
        if isinstance(results, tuple):
            vectors, alphas, kappas = results
            assert len(vectors) == 10
        else:
            assert len(results) <= 10
    
    def test_search_with_metadata(self, simple_db, sample_vectors, sample_query):
        """Test búsqueda con metadata."""
        ids = [f"doc_{i}" for i in range(100)]
        metadata = [{"category": "tech" if i % 2 == 0 else "science"} for i in range(100)]
        
        simple_db.add(ids=ids, vectors=sample_vectors[:100], metadata=metadata)
        
        results = simple_db.search(sample_query, k=10, include_metadata=True)
        
        assert isinstance(results, list)
        assert len(results) <= 10
    
    def test_update_document(self, simple_db, sample_vectors):
        """Test actualizar documento."""
        simple_db.add(ids=["doc_1"], vectors=sample_vectors[:1])
        
        new_vector = np.random.randn(640).astype(np.float32)
        result = simple_db.update("doc_1", vector=new_vector, metadata={"updated": True})
        
        assert result.success
    
    def test_delete_document(self, simple_db, sample_vectors):
        """Test eliminar documento."""
        simple_db.add(ids=["doc_1", "doc_2"], vectors=sample_vectors[:2])
        
        result = simple_db.delete(id="doc_1")
        
        assert result.deleted == 1
        stats = simple_db.get_stats()
        assert stats['deleted_documents'] == 1
    
    def test_crud_complete(self, simple_db, sample_vectors, sample_query):
        """Test CRUD completo."""
        # Create
        ids = [f"doc_{i}" for i in range(50)]
        simple_db.add(ids=ids, vectors=sample_vectors[:50])
        
        # Read
        results = simple_db.search(sample_query, k=5)
        assert len(results) > 0
        
        # Update
        simple_db.update("doc_0", metadata={"updated": True})
        
        # Delete
        simple_db.delete(id="doc_1")
        
        stats = simple_db.get_stats()
        assert stats['active_documents'] == 49


class TestAdvancedVectorDB:
    """Tests para AdvancedVectorDB."""
    
    def test_soc_consolidation(self, advanced_db, sample_vectors):
        """Test consolidación SOC."""
        ids = [f"doc_{i}" for i in range(100)]
        advanced_db.add(ids=ids, vectors=sample_vectors[:100])
        
        n_removed = advanced_db.consolidate(threshold=0.01)
        
        # No debería remover splats con threshold muy bajo
        assert n_removed >= 0
    
    def test_energy_computation(self, advanced_db, sample_vectors):
        """Test cálculo de energía."""
        ids = [f"doc_{i}" for i in range(50)]
        advanced_db.add(ids=ids, vectors=sample_vectors[:50])
        
        query = sample_vectors[0]
        energy = advanced_db.get_energy(query)
        
        assert isinstance(energy, float)
    
    def test_search_with_energy(self, advanced_db, sample_vectors):
        """Test búsqueda con información energética."""
        ids = [f"doc_{i}" for i in range(50)]
        advanced_db.add(ids=ids, vectors=sample_vectors[:50])
        
        query = sample_vectors[0]
        result = advanced_db.search_with_energy(query, k=10)
        
        assert result.query_energy is not None
        assert len(result.results) <= 10
    
    def test_exploration_suggestions(self, advanced_db, sample_vectors):
        """Test sugerencias de exploración."""
        ids = [f"doc_{i}" for i in range(50)]
        advanced_db.add(ids=ids, vectors=sample_vectors[:50])
        
        suggestions = advanced_db.suggest_exploration(n=3)
        
        assert len(suggestions) <= 3


# =============================================================================
# GPU Auto-Tuning Tests
# =============================================================================

class TestGPUAutoTuner:
    """Tests para GPU auto-tuning."""
    
    def test_gpu_detection(self):
        """Test detección de GPU."""
        tuner = GPUAutoTuner()
        profile = tuner.detect_gpu()
        
        # Puede ser None si no hay GPU
        if profile:
            assert profile.vendor in ["AMD", "NVIDIA", "Intel", "ARM", "Unknown"]
            assert profile.vram_mb > 0
            assert profile.max_workgroup_size > 0
    
    def test_optimal_config_without_gpu(self):
        """Test configuración óptima sin GPU."""
        tuner = GPUAutoTuner()
        tuner.profile = None  # Simular sin GPU
        
        config = tuner.get_optimal_config()
        
        assert config['batch_size'] == 100
        assert config['enable_vulkan'] is False
    
    def test_memory_pool(self):
        """Test memory pool."""
        pool = GPUMemoryPool(max_buffers=5)
        
        # Obtener buffer (retorna None si no hay GPU)
        buffer = pool.get_buffer(1024)
        
        # Retornar buffer
        if buffer:
            pool.return_buffer(buffer, 1024)
        
        pool.clear()
    
    def test_batch_size_estimation(self):
        """Test estimación de batch size."""
        tuner = GPUAutoTuner()
        
        # 8GB VRAM
        batch = tuner._estimate_optimal_batch(8192)
        assert batch > 0
        
        # 4GB VRAM
        batch = tuner._estimate_optimal_batch(4096)
        assert batch > 0


# =============================================================================
# Query Optimizer Tests
# =============================================================================

class TestQueryCache:
    """Tests para query cache."""
    
    def test_cache_put_get(self, sample_query):
        """Test básico de cache."""
        cache = QueryCache(max_entries=10)
        
        results = (sample_query[:10], np.ones(10), np.ones(10) * 10)
        cache.put(sample_query, results, k=10)
        
        cached = cache.get(sample_query, k=10)
        
        assert cached is not None
        np.testing.assert_array_equal(cached[0], results[0])
    
    def test_cache_miss(self, sample_query):
        """Test cache miss."""
        cache = QueryCache()
        
        result = cache.get(sample_query, k=10)
        
        assert result is None
    
    def test_cache_eviction(self):
        """Test eviction LRU."""
        cache = QueryCache(max_entries=3)
        
        for i in range(5):
            query = np.random.randn(640).astype(np.float32)
            cache.put(query, (query[:10],), k=10)
        
        stats = cache.get_stats()
        
        # Debe haber evictado al menos 2 entradas
        assert stats['evictions'] >= 2
    
    def test_cache_ttl(self, sample_query):
        """Test TTL de cache."""
        cache = QueryCache(default_ttl=0.1)  # 100ms TTL
        
        results = (sample_query[:10],)
        cache.put(sample_query, results, k=10)
        
        # Inmediatamente debe estar
        cached = cache.get(sample_query, k=10)
        assert cached is not None
        
        # Esperar expiración
        time.sleep(0.15)
        cached = cache.get(sample_query, k=10)
        assert cached is None
    
    def test_cache_stats(self, sample_query):
        """Test estadísticas de cache."""
        cache = QueryCache()
        
        # Put
        cache.put(sample_query, (sample_query[:10],), k=10)
        
        # Hit
        cache.get(sample_query, k=10)
        
        # Miss
        cache.get(np.random.randn(640).astype(np.float32), k=10)
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestQueryPrefetcher:
    """Tests para prefetcher."""
    
    def test_pattern_detection(self):
        """Test detección de patrones."""
        prefetcher = QueryPrefetcher()
        
        # Registrar patrón repetitivo
        pattern = ["q1", "q2", "q3"]
        for _ in range(3):
            for q in pattern:
                prefetcher.record_query(q)
        
        # Debe detectar el patrón
        assert len(prefetcher._patterns) > 0
    
    def test_predict_next(self):
        """Test predicción de siguiente query."""
        prefetcher = QueryPrefetcher()
        
        # Crear patrón más explícito
        for _ in range(10):
            prefetcher.record_query("q1")
            prefetcher.record_query("q2")
        
        predicted = prefetcher.predict_next("q1")
        
        # Debe predecir algo (puede ser q1 o q2 dependiendo del patrón)
        assert predicted in ["q1", "q2"]
    
    def test_prefetch_candidates(self):
        """Test obtención de candidatos."""
        prefetcher = QueryPrefetcher()
        
        # Registrar queries
        for i in range(10):
            prefetcher.record_query(f"q{i}")
        
        candidates = prefetcher.get_prefetch_candidates(n=3)
        
        assert len(candidates) <= 3


class TestQueryOptimizer:
    """Tests para optimizador completo."""
    
    def test_execute_with_cache(self, sample_query):
        """Test ejecución con cache."""
        optimizer = QueryOptimizer(cache_entries=10)
        
        # Función de búsqueda simulada
        search_count = [0]
        def search_fn(query, k):
            search_count[0] += 1
            return (query[:k], np.ones(k), np.ones(k) * 10)
        
        # Primera ejecución
        result1 = optimizer.execute_with_cache(sample_query, 10, search_fn)
        assert search_count[0] == 1
        
        # Segunda ejecución (debe usar cache)
        result2 = optimizer.execute_with_cache(sample_query, 10, search_fn)
        assert search_count[0] == 1  # No incrementó
        
        # Verificar que retornó lo mismo
        np.testing.assert_array_equal(result1[0], result2[0])
    
    def test_optimizer_stats(self, sample_query):
        """Test estadísticas del optimizador."""
        optimizer = QueryOptimizer()
        
        def search_fn(query, k):
            return (query[:k],)
        
        # Ejecutar varias veces
        for i in range(10):
            query = np.random.randn(640).astype(np.float32)
            optimizer.execute_with_cache(query, 10, search_fn)
        
        stats = optimizer.get_stats()
        
        assert stats['total_queries'] == 10


# =============================================================================
# Auto-Scaling Tests
# =============================================================================

class TestMetricsCollector:
    """Tests para recolector de métricas."""
    
    def test_record_metrics(self):
        """Test registro de métricas."""
        collector = MetricsCollector()
        
        metrics = NodeMetrics(
            node_id="node1",
            cpu_percent=50.0,
            memory_percent=60.0,
            qps=100.0,
            latency_ms=10.0,
            active_queries=5,
            uptime_seconds=3600
        )
        
        collector.record(metrics)
        
        stats = collector.get_cluster_stats()
        
        assert stats['nodes'] == 1
        assert stats['avg_cpu'] == 50.0
    
    def test_trend_detection(self):
        """Test detección de tendencias."""
        collector = MetricsCollector()
        
        # Trend creciente
        for i in range(20):
            metrics = NodeMetrics(
                node_id="node1",
                cpu_percent=float(i * 5),  # 0, 5, 10, ..., 95
                memory_percent=50.0,
                qps=100.0,
                latency_ms=10.0,
                active_queries=5,
                uptime_seconds=i
            )
            collector.record(metrics)
        
        trend = collector.get_trend('cpu_percent')
        
        assert trend == 'increasing'


class TestAutoScaler:
    """Tests para auto-scaler."""
    
    def test_scale_up_decision(self):
        """Test decisión de escalar up."""
        scaler = AutoScaler(min_nodes=1, max_nodes=5, scale_up_threshold=80.0)
        
        # Simular métricas con alta carga
        for i in range(10):
            metrics = NodeMetrics(
                node_id=f"node{i % 2}",
                cpu_percent=90.0,  # Alta carga
                memory_percent=50.0,
                qps=100.0,
                latency_ms=10.0,
                active_queries=10,
                uptime_seconds=i
            )
            scaler.update_metrics(metrics)
        
        decision = scaler.evaluate_scaling()
        
        if decision:
            assert decision.action.value == "scale_up"
    
    def test_scale_down_decision(self):
        """Test decisión de escalar down."""
        scaler = AutoScaler(min_nodes=1, max_nodes=5, scale_down_threshold=30.0)
        scaler.current_nodes = 3
        
        # Simular métricas con baja carga
        for i in range(10):
            metrics = NodeMetrics(
                node_id=f"node{i % 3}",
                cpu_percent=20.0,  # Baja carga
                memory_percent=25.0,
                qps=50.0,
                latency_ms=5.0,
                active_queries=2,
                uptime_seconds=i
            )
            scaler.update_metrics(metrics)
        
        decision = scaler.evaluate_scaling()
        
        if decision:
            assert decision.action.value == "scale_down"
    
    def test_cooldown(self):
        """Test cooldown entre escalados."""
        scaler = AutoScaler(cooldown_seconds=1.0)
        scaler._last_scale_time = time.time()
        
        # Intentar escalar inmediatamente
        decision = scaler.evaluate_scaling()
        
        # No debe decidir nada por cooldown
        assert decision is None
    
    def test_scaling_callbacks(self):
        """Test callbacks de escalado."""
        scaler = AutoScaler(min_nodes=1, max_nodes=3)
        
        # Registrar callbacks
        scale_up_called = [False]
        scale_down_called = [False]
        
        def on_scale_up():
            scale_up_called[0] = True
            return True
        
        def on_scale_down():
            scale_down_called[0] = True
            return True
        
        scaler.register_callbacks(scale_up=on_scale_up, scale_down=on_scale_down)
        
        # Simular escalado manual
        from m2m.auto_scaling import ScalingDecision, ScalingDirection, ScalingTrigger
        
        decision = ScalingDecision(
            action=ScalingDirection.SCALE_UP,
            trigger=ScalingTrigger.MANUAL,
            current_nodes=1,
            target_nodes=2,
            reason="Manual scale up",
            metrics={'avg_cpu': 50.0}
        )
        
        scaler.execute_scaling(decision)
        
        assert scale_up_called[0]


class TestHorizontalScaler:
    """Tests para escalador horizontal."""
    
    def test_scale_up(self):
        """Test añadir nodo."""
        scaler = HorizontalScaler(node_template={'cpu': 4, 'memory': '8GB'})
        
        success = scaler.scale_up()
        
        assert success
        assert len(scaler.get_active_nodes()) == 1
    
    def test_scale_down(self):
        """Test remover nodo."""
        scaler = HorizontalScaler(node_template={})
        
        # Añadir 2 nodos
        scaler.scale_up()
        scaler.scale_up()
        
        initial_count = len(scaler.get_active_nodes())
        
        # Remover 1
        success = scaler.scale_down()
        
        assert success
        assert len(scaler.get_active_nodes()) == initial_count - 1
    
    def test_cannot_remove_last_node(self):
        """Test que no se puede remover el último nodo."""
        scaler = HorizontalScaler(node_template={})
        scaler.scale_up()
        
        success = scaler.scale_down()
        
        assert success is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Tests de integración."""
    
    def test_full_workflow(self, sample_vectors, sample_query):
        """Test workflow completo."""
        # 1. Crear DB
        db = AdvancedVectorDB(latent_dim=640, enable_soc=True)
        
        # 2. Añadir datos
        ids = [f"doc_{i}" for i in range(100)]
        metadata = [{"category": "test"} for _ in range(100)]
        db.add(ids=ids, vectors=sample_vectors[:100], metadata=metadata)
        
        # 3. Búsqueda
        results = db.search_with_energy(sample_query, k=10)
        assert len(results.results) > 0
        
        # 4. Exploración
        suggestions = db.suggest_exploration(n=3)
        assert len(suggestions) <= 3
        
        # 5. Actualización
        db.update("doc_0", metadata={"updated": True})
        
        # 6. Eliminación
        db.delete(id="doc_1")
        
        # 7. Stats
        stats = db.get_stats()
        assert stats['active_documents'] == 99
    
    def test_cache_integration(self, simple_db, sample_vectors, sample_query):
        """Test integración con cache."""
        optimizer = QueryOptimizer(cache_entries=100)
        
        # Añadir datos
        ids = [f"doc_{i}" for i in range(50)]
        simple_db.add(ids=ids, vectors=sample_vectors[:50])
        
        # Búsqueda con cache
        def search_fn(query, k):
            return simple_db.search(query, k)
        
        # Primera búsqueda
        result1 = optimizer.execute_with_cache(sample_query, 10, search_fn)
        
        # Segunda búsqueda (cached)
        result2 = optimizer.execute_with_cache(sample_query, 10, search_fn)
        
        stats = optimizer.get_stats()
        assert stats['cache_hit_rate'] > 0


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Tests de rendimiento."""
    
    @pytest.mark.slow
    def test_large_scale_ingestion(self):
        """Test ingestión a gran escala."""
        db = SimpleVectorDB(latent_dim=640, mode='edge')
        
        # 10K vectores
        vectors = np.random.randn(10000, 640).astype(np.float32)
        ids = [f"doc_{i}" for i in range(10000)]
        
        start = time.time()
        n = db.add(ids=ids, vectors=vectors)
        elapsed = time.time() - start
        
        assert n == 10000
        print(f"\nIngestion rate: {n/elapsed:.0f} docs/sec")
    
    @pytest.mark.slow
    def test_search_performance(self):
        """Test rendimiento de búsqueda."""
        db = SimpleVectorDB(latent_dim=640, mode='edge')
        
        # Preparar datos
        vectors = np.random.randn(10000, 640).astype(np.float32)
        ids = [f"doc_{i}" for i in range(10000)]
        db.add(ids=ids, vectors=vectors)
        
        # Benchmark
        queries = np.random.randn(1000, 640).astype(np.float32)
        
        start = time.time()
        for query in queries:
            db.search(query, k=10)
        elapsed = time.time() - start
        
        qps = 1000 / elapsed
        print(f"\nSearch throughput: {qps:.1f} QPS")
        
        assert qps > 50  # Al menos 50 QPS


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
