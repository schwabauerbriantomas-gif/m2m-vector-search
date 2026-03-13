"""
Query Optimizer con Cache y Prefetching para M2M.

Características:
- LRU cache para queries frecuentes
- Prefetching predictivo basado en patrones
- Query plan optimization
- Result caching con TTL
- Adaptive caching basado en memoria
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    """Entrada del cache de queries."""
    query_hash: str
    results: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    ttl_seconds: float = 300.0  # 5 minutos default
    
    def is_expired(self) -> bool:
        """Verifica si la entrada expiró."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds
    
    def touch(self):
        """Actualiza timestamp de acceso."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class QueryPattern:
    """Patrón de queries detectado."""
    query_ids: List[str]
    frequency: int
    last_seen: float
    next_predicted: Optional[str] = None


class QueryCache:
    """
    Cache LRU para resultados de queries.
    
    Características:
    - LRU eviction cuando se llena
    - TTL para expiración automática
    - Métricas de hit/miss
    - Adaptive sizing basado en memoria
    """
    
    def __init__(self, 
                 max_entries: int = 1000,
                 max_memory_mb: int = 100,
                 default_ttl: float = 300.0):
        """
        Args:
            max_entries: Máximo número de entradas
            max_memory_mb: Máximo uso de memoria en MB
            default_ttl: TTL default en segundos
        """
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_memory_bytes = 0
        
        # Métricas
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _hash_query(self, query: np.ndarray, k: int, filters: Optional[Dict] = None) -> str:
        """
        Genera hash único para una query.
        
        Args:
            query: Vector de query
            k: Número de resultados
            filters: Filtros aplicados
            
        Returns:
            Hash string
        """
        # Hash del vector
        query_bytes = query.tobytes()
        vector_hash = hashlib.sha256(query_bytes).hexdigest()[:16]
        
        # Hash de parámetros
        params_str = f"k={k}|filters={filters}"
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        
        return f"{vector_hash}_{params_hash}"
    
    def get(self, 
            query: np.ndarray, 
            k: int = 10,
            filters: Optional[Dict] = None) -> Optional[Any]:
        """
        Busca resultado en cache.
        
        Args:
            query: Vector de query
            k: Número de resultados
            filters: Filtros aplicados
            
        Returns:
            Resultados cacheados o None
        """
        query_hash = self._hash_query(query, k, filters)
        
        if query_hash not in self._cache:
            self.misses += 1
            return None
        
        entry = self._cache[query_hash]
        
        # Verificar expiración
        if entry.is_expired():
            self._evict(query_hash)
            self.misses += 1
            return None
        
        # Mover al final (LRU)
        self._cache.move_to_end(query_hash)
        entry.touch()
        
        self.hits += 1
        return entry.results
    
    def put(self,
            query: np.ndarray,
            results: Any,
            k: int = 10,
            filters: Optional[Dict] = None,
            ttl: Optional[float] = None):
        """
        Almacena resultado en cache.
        
        Args:
            query: Vector de query
            results: Resultados a cachear
            k: Número de resultados
            filters: Filtros aplicados
            ttl: TTL personalizado (None = default)
        """
        query_hash = self._hash_query(query, k, filters)
        
        # Estimar tamaño
        size_bytes = self._estimate_size(results)
        
        # Verificar si necesitamos hacer espacio
        while (len(self._cache) >= self.max_entries or 
               self._current_memory_bytes + size_bytes > self.max_memory_mb * 1024 * 1024):
            if not self._evict_oldest():
                break
        
        # Crear entrada
        entry = CacheEntry(
            query_hash=query_hash,
            results=results,
            timestamp=time.time(),
            last_access=time.time(),
            size_bytes=size_bytes,
            ttl_seconds=ttl if ttl is not None else self.default_ttl
        )
        
        # Si ya existe, actualizar
        if query_hash in self._cache:
            old_entry = self._cache[query_hash]
            self._current_memory_bytes -= old_entry.size_bytes
        
        self._cache[query_hash] = entry
        self._current_memory_bytes += size_bytes
    
    def _estimate_size(self, results: Any) -> int:
        """Estima tamaño en bytes de los resultados."""
        if results is None:
            return 0
        
        if isinstance(results, tuple):
            # (vectors, alphas, kappas)
            total = 0
            for item in results:
                if isinstance(item, np.ndarray):
                    total += item.nbytes
            return total
        
        if isinstance(results, list):
            # Lista de DocResult
            return len(results) * 200  # Estimación
        
        return 1000  # Default
    
    def _evict(self, query_hash: str):
        """Evicta una entrada específica."""
        if query_hash in self._cache:
            entry = self._cache.pop(query_hash)
            self._current_memory_bytes -= entry.size_bytes
            self.evictions += 1
    
    def _evict_oldest(self) -> bool:
        """Evicta la entrada más antigua (LRU)."""
        if not self._cache:
            return False
        
        # OrderedDict mantiene orden de inserción
        oldest_key = next(iter(self._cache))
        self._evict(oldest_key)
        return True
    
    def clear(self):
        """Limpia todo el cache."""
        self._cache.clear()
        self._current_memory_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del cache."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'entries': len(self._cache),
            'max_entries': self.max_entries,
            'memory_mb': self._current_memory_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_mb,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate
        }


class QueryPrefetcher:
    """
    Prefetching predictivo basado en patrones de queries.
    
    Detecta patrones secuenciales y prefetcha resultados
    que probablemente serán necesarios.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Tamaño de ventana para detectar patrones
        """
        self.window_size = window_size
        
        # Historial de queries
        self._query_history: List[str] = []
        
        # Patrones detectados
        self._patterns: Dict[str, QueryPattern] = {}
        
        # Cola de prefetch
        self._prefetch_queue: List[str] = []
    
    def record_query(self, query_hash: str):
        """
        Registra una query en el historial.
        
        Args:
            query_hash: Hash de la query ejecutada
        """
        self._query_history.append(query_hash)
        
        # Mantener tamaño de ventana
        if len(self._query_history) > self.window_size * 2:
            self._query_history = self._query_history[-self.window_size:]
        
        # Detectar patrones
        self._detect_patterns()
    
    def _detect_patterns(self):
        """Detecta patrones secuenciales en el historial."""
        if len(self._query_history) < 3:
            return
        
        # Buscar secuencias de 2-3 queries
        for seq_len in [3, 2]:
            if len(self._query_history) < seq_len:
                continue
            
            # Últimas N queries
            recent = self._query_history[-seq_len:]
            pattern_key = '|'.join(recent)
            
            # Actualizar o crear patrón
            if pattern_key not in self._patterns:
                self._patterns[pattern_key] = QueryPattern(
                    query_ids=recent,
                    frequency=1,
                    last_seen=time.time()
                )
            else:
                pattern = self._patterns[pattern_key]
                pattern.frequency += 1
                pattern.last_seen = time.time()
    
    def predict_next(self, current_hash: str) -> Optional[str]:
        """
        Predice la siguiente query basándose en patrones.
        
        Args:
            current_hash: Hash de la query actual
            
        Returns:
            Hash de la query predicha o None
        """
        if not self._patterns:
            return None
        
        # Buscar patrones que terminen con la query actual
        best_pattern = None
        best_score = 0
        
        for pattern_key, pattern in self._patterns.items():
            if pattern.query_ids[-1] == current_hash:
                # Score basado en frecuencia y recencia
                recency = time.time() - pattern.last_seen
                score = pattern.frequency / (1 + recency)
                
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
        
        # Si encontramos patrón, predecir siguiente
        if best_pattern and len(best_pattern.query_ids) >= 2:
            return best_pattern.query_ids[0]  # Query que sigue en el patrón
        
        return None
    
    def get_prefetch_candidates(self, n: int = 3) -> List[str]:
        """
        Obtiene candidatos para prefetching.
        
        Args:
            n: Número de candidatos
            
        Returns:
            Lista de hashes para prefetch
        """
        if not self._query_history:
            return []
        
        # Usar última query como base
        last_query = self._query_history[-1]
        predicted = self.predict_next(last_query)
        
        candidates = []
        if predicted:
            candidates.append(predicted)
        
        # Agregar queries frecuentes del historial
        from collections import Counter
        freq = Counter(self._query_history)
        for query_hash, _ in freq.most_common(n):
            if query_hash not in candidates:
                candidates.append(query_hash)
                if len(candidates) >= n:
                    break
        
        return candidates


class QueryOptimizer:
    """
    Optimizador completo de queries con cache y prefetching.
    
    Integra:
    - LRU cache
    - Prefetching predictivo
    - Query planning
    - Métricas
    """
    
    def __init__(self,
                 cache_entries: int = 1000,
                 cache_memory_mb: int = 100,
                 enable_prefetch: bool = True):
        """
        Args:
            cache_entries: Máximo entradas en cache
            cache_memory_mb: Máxima memoria para cache
            enable_prefetch: Habilitar prefetching
        """
        self.cache = QueryCache(
            max_entries=cache_entries,
            max_memory_mb=cache_memory_mb
        )
        
        self.prefetcher = QueryPrefetcher() if enable_prefetch else None
        self.enable_prefetch = enable_prefetch
        
        # Métricas
        self.total_queries = 0
        self.cache_hits = 0
        self.prefetch_hits = 0
    
    def execute_with_cache(self,
                          query: np.ndarray,
                          k: int,
                          search_fn,
                          filters: Optional[Dict] = None) -> Any:
        """
        Ejecuta query con cache y métricas.
        
        Args:
            query: Vector de query
            k: Número de resultados
            search_fn: Función de búsqueda real
            filters: Filtros opcionales
            
        Returns:
            Resultados de búsqueda
        """
        self.total_queries += 1
        
        # Intentar cache
        cached = self.cache.get(query, k, filters)
        if cached is not None:
            self.cache_hits += 1
            return cached
        
        # Ejecutar búsqueda real
        results = search_fn(query, k)
        
        # Almacenar en cache
        self.cache.put(query, results, k, filters)
        
        # Registrar en prefetcher
        if self.enable_prefetch and self.prefetcher:
            query_hash = self.cache._hash_query(query, k, filters)
            self.prefetcher.record_query(query_hash)
        
        return results
    
    def get_prefetch_suggestions(self, n: int = 3) -> List[str]:
        """
        Obtiene sugerencias de prefetch.
        
        Args:
            n: Número de sugerencias
            
        Returns:
            Lista de hashes para prefetch
        """
        if not self.enable_prefetch or not self.prefetcher:
            return []
        
        return self.prefetcher.get_prefetch_candidates(n)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas."""
        stats = {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_enabled': self.enable_prefetch
        }
        
        stats.update({'cache': self.cache.get_stats()})
        
        return stats
    
    def clear(self):
        """Limpia cache y historial."""
        self.cache.clear()
        if self.prefetcher:
            self.prefetcher._query_history.clear()
            self.prefetcher._patterns.clear()
