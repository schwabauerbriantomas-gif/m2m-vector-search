#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2M (Machine-to-Memory) Configuration File
Centralized configuration for M2M system.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class M2MConfig:
    """
    Configuration for M2M (Machine-to-Memory) system.
    """
    
    # --- System Configuration ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # --- Latent Space Configuration ---
    latent_dim: int = 640  # S^639 hyper-sphere
    n_splats_init: int = 10000  # Initial number of splats
    max_splats: int = 100000   # Maximum capacity
    knn_k: int = 64             # K-nearest neighbors for retrieval
    
    # --- Splat Parameters ---
    init_alpha: float = 1.0       # Initial amplitude
    init_kappa: float = 10.0      # Initial concentration
    min_kappa: float = 1.0       # Minimum concentration (avoid collapse)
    max_kappa: float = 50.0      # Maximum concentration limit
    
    # --- Temperature (for exploration) ---
    splat_temperature: float = 0.1
    weight_decay_start: float = 1.0
    
    # --- Energy Function Weights ---
    energy_splat_weight: float = 1.0    # Weight for splat energy term
    energy_geom_weight: float = 0.1      # Weight for geometric energy
    energy_comp_weight: float = 0.0      # Weight for composition energy
    global_temperature: float = 1.0            # Global temperature for sampling
    
    # --- SOC Parameters ---
    soc_threshold: float = 0.8       # Criticality threshold for consolidation
    soc_buffer_capacity: int = 1000  # Number of splats to track
    soc_update_interval: int = 100  # Update SOC every N samples
    phi_convergence_threshold: float = 0.95  # Order parameter convergence
    
    # --- Hardware Acceleration ---
    enable_vulkan: bool = False      # Enable Vulkan GPU acceleration
    
    # --- Memory Hierarchy ---
    enable_3_tier_memory: bool = True # Enable VRAM/RAM/SSD hierarchy
    memory_tier: str = "3-tier"  # 3-tier memory (VRAM/RAM/SSD)
    
    # --- Hierarchical Context ---
    context_local: int = 12  # Local context (current token)
    context_medium: int = 64  # Medium context (recent history)
    context_global: int = 512  # Global context (long-term memory)
    
    # --- Decoder Configuration (MoE) ---
    vocab_size: int = 50257         # GPT-2 vocabulary
    hidden_dim: int = 1024           # Hidden dimension
    moe_experts: int = 4             # Number of mixture of experts
    moe_active: int = 2               # Number of active experts
    
    # --- Training Configuration ---
    batch_size: int = 32
    seq_length: int = 32
    noise_levels: tuple = (0.01, 0.05, 0.1, 0.2, 0.5)
    
    # --- Optimization ---
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # --- Langevin Dynamics Configuration ---
    langevin_steps: int = 200       # Number of Langevin steps
    langevin_dt: float = 0.001      # Time step
    langevin_gamma: float = 0.1      # Friction coefficient
    langevin_T: float = 1.0           # Temperature
    
    # --- API Configuration ---
    rest_port: int = 8080
    grpc_port: int = 9090
