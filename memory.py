"""
Memory Manager for Gaussian Splats.

Implements hierarchical memory management for efficient
storage and retrieval of large-scale splat datasets.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import time

from splat_types import GaussianSplat


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    vram_limit: int = 100000      # Max splats in VRAM (hot tier)
    ram_limit: int = 1000000      # Max splats in RAM (warm tier)
    eviction_threshold: float = 0.8  # Evict when usage > 80%
    access_threshold: int = 10    # Promote after N accesses


@dataclass
class MemoryStats:
    """Memory statistics."""
    vram_usage: int = 0
    ram_usage: int = 0
    total_splats: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0


class SplatMemoryManager:
    """
    Manages hierarchical memory for Gaussian splats.
    
    Implements three-tier memory architecture:
    - Hot (VRAM): Frequently accessed splats
    - Warm (RAM): Recently accessed splats
    - Cold (Disk): Infrequently accessed splats
    
    Example:
        >>> manager = SplatMemoryManager(vram_limit=50000)
        >>> manager.add_splats(splats)
        >>> splat = manager.get_splat(splat_id)
    """
    
    def __init__(
        self,
        vram_limit: int = 100000,
        ram_limit: int = 1000000,
        eviction_threshold: float = 0.8,
        access_threshold: int = 10
    ):
        """
        Initialize memory manager.
        
        Args:
            vram_limit: Maximum splats in VRAM
            ram_limit: Maximum splats in RAM
            eviction_threshold: Fraction at which to start eviction
            access_threshold: Accesses needed for promotion
        """
        self.vram_limit = vram_limit
        self.ram_limit = ram_limit
        self.eviction_threshold = eviction_threshold
        self.access_threshold = access_threshold
        
        # Tiers
        self._vram: Dict[int, GaussianSplat] = {}
        self._ram: Dict[int, GaussianSplat] = {}
        self._cold: Dict[int, GaussianSplat] = {}
        
        # Access tracking
        self._access_count: Dict[int, int] = {}
        self._last_access: Dict[int, float] = {}
        
        # Statistics
        self._stats = MemoryStats()
    
    def add_splats(self, splats: List[GaussianSplat], to_cold: bool = True) -> None:
        """
        Add splats to memory.
        
        Args:
            splats: List of splats to add
            to_cold: If True, add to cold storage initially
        """
        for splat in splats:
            if to_cold:
                self._cold[splat.id] = splat
            else:
                self._ram[splat.id] = splat
            
            self._access_count[splat.id] = 0
            self._last_access[splat.id] = 0.0
        
        self._stats.total_splats = len(self._cold) + len(self._ram) + len(self._vram)
    
    def get_splat(self, splat_id: int) -> Optional[GaussianSplat]:
        """
        Get a splat by ID with automatic tier management.
        
        Args:
            splat_id: Splat identifier
        
        Returns:
            GaussianSplat or None if not found
        """
        # Update access tracking
        self._access_count[splat_id] = self._access_count.get(splat_id, 0) + 1
        self._last_access[splat_id] = time.time()
        
        # Check VRAM (hot)
        if splat_id in self._vram:
            self._stats.cache_hits += 1
            return self._vram[splat_id]
        
        # Check RAM (warm)
        if splat_id in self._ram:
            self._stats.cache_hits += 1
            splat = self._ram[splat_id]
            
            # Promote to VRAM if frequently accessed
            if self._access_count[splat_id] >= self.access_threshold:
                self._promote_to_vram(splat_id)
            
            return splat
        
        # Load from cold storage
        if splat_id in self._cold:
            self._stats.cache_misses += 1
            splat = self._cold[splat_id]
            
            # Load to RAM
            self._load_to_ram(splat_id)
            
            return splat
        
        return None
    
    def prefetch_to_warm(self, splat_ids: List[int]) -> None:
        """
        Prefetch splats from cold storage to warm storage asynchronously.
        Useful for Data Lake sequential or sampled iterations.
        """
        for splat_id in splat_ids:
            if splat_id in self._cold and splat_id not in self._ram and splat_id not in self._vram:
                self._load_to_ram(splat_id)
                
    def _promote_to_vram(self, splat_id: int) -> None:
        """Promote splat from RAM to VRAM."""
        if splat_id not in self._ram:
            return
        
        # Check if eviction needed
        if len(self._vram) >= self.vram_limit * self.eviction_threshold:
            self._evict_from_vram()
        
        # Move to VRAM
        self._vram[splat_id] = self._ram.pop(splat_id)
        self._stats.vram_usage = len(self._vram)
        self._stats.ram_usage = len(self._ram)
    
    def _load_to_ram(self, splat_id: int) -> None:
        """Load splat from cold storage to RAM."""
        if splat_id not in self._cold:
            return
        
        # Check if eviction needed
        if len(self._ram) >= self.ram_limit * self.eviction_threshold:
            self._evict_from_ram()
        
        # Load to RAM
        self._ram[splat_id] = self._cold[splat_id]
        self._stats.ram_usage = len(self._ram)
    
    def _evict_from_vram(self) -> None:
        """Evict least recently used splats from VRAM."""
        if not self._vram:
            return
        
        # Find LRU splat
        lru_id = min(self._vram.keys(), key=lambda x: self._last_access.get(x, 0))
        
        # Move to RAM
        self._ram[lru_id] = self._vram.pop(lru_id)
        self._stats.evictions += 1
        self._stats.vram_usage = len(self._vram)
        self._stats.ram_usage = len(self._ram)
    
    def _evict_from_ram(self) -> None:
        """Evict least recently used splats from RAM."""
        if not self._ram:
            return
        
        # Find LRU splat
        lru_id = min(self._ram.keys(), key=lambda x: self._last_access.get(x, 0))
        
        # Keep in cold storage (already there)
        self._ram.pop(lru_id)
        self._stats.evictions += 1
        self._stats.ram_usage = len(self._ram)
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        self._stats.vram_usage = len(self._vram)
        self._stats.ram_usage = len(self._ram)
        self._stats.total_splats = len(self._cold) + len(self._ram) + len(self._vram)
        return self._stats
    
    def clear(self) -> None:
        """Clear all memory."""
        self._vram.clear()
        self._ram.clear()
        self._cold.clear()
        self._access_count.clear()
        self._last_access.clear()
        self._stats = MemoryStats()
    
    @property
    def vram_size(self) -> int:
        """Number of splats in VRAM."""
        return len(self._vram)
    
    @property
    def ram_size(self) -> int:
        """Number of splats in RAM."""
        return len(self._ram)
    
    @property
    def cold_size(self) -> int:
        """Number of splats in cold storage."""
        return len(self._cold)
