"""
M2M Custom Embedding Model - Configuration
===========================================
Knowledge Distillation from OpenAI text-embedding-3-large (3072D)
to a lightweight 640D embedding model with Matryoshka representations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EmbeddingConfig:
    """Configuration for M2M custom embedding model training."""

    # Model architecture
    teacher_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    student_base: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 640

    # Matryoshka dimensions (coarse-to-fine)
    matryoshka_dims: Tuple[int, ...] = (64, 128, 256, 640)
    use_matryoshka: bool = True

    # Training
    batch_size: int = 128
    gradient_accumulation_steps: int = 4  # effective batch = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_length: int = 256

    # Loss weights
    mse_loss_weight: float = 1.0
    cosine_loss_weight: float = 0.5
    matryoshka_loss_weight: float = 0.3

    # Knowledge distillation
    teacher_dim: int = 384  # MiniLM output dim
    kd_temperature: float = 1.0

    # Training data
    train_size: int = 50000  # number of training samples
    val_size: int = 5000
    num_workers: int = 4

    # Evaluation
    eval_k_values: Tuple[int, ...] = (1, 5, 10, 50, 100)

    # Paths
    model_save_dir: str = "models/m2m_embeddings"
    log_dir: str = "logs/embedding_training"

    # Mixed precision
    use_amp: bool = True

    # Seed
    seed: int = 42

    # Device
    device: str = "cuda"  # auto-detected if "auto"


@dataclass
class DatasetConfig:
    """Configuration for training dataset generation."""

    # Use synthetic data from Wikipedia-style text
    synthetic: bool = True
    num_synthetic_samples: int = 100000

    # Or use a local parquet dataset
    dataset_path: str = ""

    # Text augmentation for contrastive pairs
    augmentations: List[str] = field(
        default_factory=lambda: [
            "dropout",  # Random token dropout
            "shuffle",  # Sentence reordering
            "synonym",  # Synonym replacement (if available)
        ]
    )

    # Hard negatives mining
    num_hard_negatives: int = 4
    mine_hard_negatives: bool = True
    mining_frequency: int = 1000  # re-mine every N steps
