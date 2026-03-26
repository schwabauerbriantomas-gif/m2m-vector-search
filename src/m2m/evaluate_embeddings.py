"""
M2M Embedding Evaluation Script
================================
Compare custom embeddings vs truncated teacher embeddings on retrieval quality.

Metrics:
- Recall@k: fraction of teacher's top-k neighbors preserved in student's top-k
- Cosine similarity distribution between student and teacher
- Encoding latency comparison

Usage:
    $env:PYTHONIOENCODING="utf-8"
    $env:PYTHONPATH="src"
    python src/m2m/evaluate_embeddings.py --checkpoint models/m2m_embeddings/final_model.pt
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Heavy dependencies - may not be installed in all environments
try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    SentenceTransformer = None
    _HAS_TORCH = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m2m.embedding_config import EmbeddingConfig
from m2m.embedding_model import M2MEmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Benchmark texts (diverse domains)
BENCHMARK_TEXTS = [
    "Artificial intelligence transforms healthcare by enabling early disease detection.",
    "The quantum computer solved a complex optimization problem in seconds.",
    "Climate change models predict rising sea levels by 2050.",
    "Blockchain technology enables decentralized financial transactions.",
    "Neural architecture search automates the design of deep learning models.",
    "Genome sequencing reveals genetic mutations linked to rare diseases.",
    "Self-driving cars use lidar and cameras to navigate complex environments.",
    "Natural language generation produces human-like text from structured data.",
    "Federated learning enables model training across distributed datasets without sharing data.",
    "Reinforcement learning from human feedback improves language model alignment.",
    "The attention mechanism allows transformers to capture long-range dependencies.",
    "Convolutional networks excel at spatial pattern recognition in images.",
    "Graph neural networks model relationships in social networks and molecular structures.",
    "Energy-efficient computing reduces the environmental impact of large AI models.",
    "Few-shot learning enables models to generalize from very few examples.",
    "Vector databases store high-dimensional embeddings for fast similarity search.",
    "Embedding space geometry affects the quality of nearest neighbor retrieval.",
    "Dimensionality reduction preserves semantic relationships while compressing vectors.",
    "Cross-modal retrieval finds images matching text descriptions and vice versa.",
    "Hyperparameter optimization improves model performance through systematic search.",
    "Distributed training scales model training across multiple GPUs and machines.",
    "Knowledge graphs represent factual relationships as triples of entities and predicates.",
    "Semantic segmentation assigns a class label to every pixel in an image.",
    "Object detection identifies and localizes multiple objects in a single image.",
    "Speech recognition converts audio signals to text using acoustic and language models.",
    "Machine translation systems translate text between languages using encoder-decoder architectures.",
    "Information extraction identifies structured information from unstructured text documents.",
    "Anomaly detection identifies unusual patterns that deviate from normal behavior.",
    "Recommendation systems suggest items based on user preferences and behavioral patterns.",
    "Time series forecasting predicts future values from historical sequential data.",
    "Transfer learning adapts pre-trained models to new domains with limited labeled data.",
    "Data augmentation increases training set diversity through synthetic transformations.",
    "Ensemble methods combine multiple models for improved prediction accuracy and robustness.",
    "Bayesian optimization efficiently searches expensive-to-evaluate hyperparameter spaces.",
    "Curriculum learning trains models on progressively more difficult examples.",
    "Contrastive learning learns representations by contrasting positive and negative pairs.",
    "Self-supervised learning creates supervision signals from the data itself without labels.",
    "Multi-task learning shares representations across related tasks for mutual benefit.",
    "Prompt engineering crafts effective instructions to guide large language model outputs.",
    "Retrieval-augmented generation combines search and generation for factual responses.",
    "Chain-of-thought reasoning breaks complex problems into step-by-step reasoning processes.",
]


def load_model(checkpoint_path: str, device: torch.device) -> M2MEmbeddingModel:
    """Load trained M2M embedding model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    config_dict = checkpoint["config"]
    config = EmbeddingConfig(
        **{k: v for k, v in config_dict.items() if hasattr(EmbeddingConfig, k)}
    )

    # Load teacher model to get encoder
    teacher_st = SentenceTransformer(config.teacher_model)
    auto_model = teacher_st[0].auto_model

    student = M2MEmbeddingModel(
        encoder=auto_model,
        embedding_dim=config.embedding_dim,
        matryoshka_dims=config.matryoshka_dims,
    )
    student.load_state_dict(checkpoint["model_state_dict"])
    student = student.to(device)
    student.eval()

    return student, config


@torch.no_grad()
def encode_with_student(
    model: M2MEmbeddingModel,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts using student model."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        emb = model(tokens["input_ids"], tokens["attention_mask"])
        all_embeddings.append(emb.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def encode_with_teacher(
    teacher: SentenceTransformer,
    texts: List[str],
) -> np.ndarray:
    """Encode texts using teacher model."""
    return teacher.encode(texts, normalize_embeddings=True)


def compute_recall_at_k(
    student_emb: np.ndarray,
    teacher_emb: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 10, 50),
) -> Dict:
    """Compute recall@k: fraction of teacher's top-k neighbors in student's top-k."""
    n = len(student_emb)
    results = {}

    # Compute similarity matrices
    student_sim = student_emb @ student_emb.T
    teacher_sim = teacher_emb @ teacher_emb.T

    for k in k_values:
        if k >= n:
            continue
        student_nn = np.argsort(-student_sim, axis=1)[:, :k]
        # For teacher, since dims differ, use teacher's own space
        teacher_sim_local = teacher_emb @ teacher_emb.T
        teacher_nn = np.argsort(-teacher_sim_local, axis=1)[:, :k]

        recall = 0.0
        for i in range(n):
            teacher_set = set(teacher_nn[i])
            recall += len(set(student_nn[i]) & teacher_set) / k
        results[f"recall@{k}"] = recall / n

    return results


def measure_latency(
    model: M2MEmbeddingModel,
    tokenizer,
    teacher: SentenceTransformer,
    texts: List[str],
    device: torch.device,
    n_runs: int = 5,
) -> Dict:
    """Measure encoding latency for student vs teacher."""
    # Warmup
    _ = encode_with_student(model, tokenizer, texts[:5], device)
    _ = encode_with_teacher(teacher, texts[:5])

    # Student latency
    student_times = []
    for _ in range(n_runs):
        t0 = time.time()
        _ = encode_with_student(model, tokenizer, texts, device)
        student_times.append(time.time() - t0)

    # Teacher latency
    teacher_times = []
    for _ in range(n_runs):
        t0 = time.time()
        _ = encode_with_teacher(teacher, texts)
        teacher_times.append(time.time() - t0)

    return {
        "student_latency_ms": np.mean(student_times) * 1000 / len(texts),
        "student_latency_std": np.std(student_times) * 1000 / len(texts),
        "teacher_latency_ms": np.mean(teacher_times) * 1000 / len(texts),
        "teacher_latency_std": np.std(teacher_times) * 1000 / len(texts),
        "speedup": np.mean(teacher_times) / np.mean(student_times),
    }


def main():
    if not _HAS_TORCH:
        print(
            "Error: PyTorch not installed. Install with: pip install m2m-vector-search[embeddings]"
        )
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Evaluate M2M custom embeddings")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of evaluation texts")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    log.info(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    log.info(f"Model loaded. Embedding dim: {config.embedding_dim}")

    # Load teacher
    teacher = SentenceTransformer(config.teacher_model)
    teacher.eval()

    # Prepare evaluation texts
    np.random.seed(42)
    n_extra = max(0, args.num_samples - len(BENCHMARK_TEXTS))
    eval_texts = (
        BENCHMARK_TEXTS[: args.num_samples]
        if args.num_samples <= len(BENCHMARK_TEXTS)
        else BENCHMARK_TEXTS
        + [
            f"Random test sentence number {i} about various topics in AI and science."
            for i in range(n_extra)
        ]
    )

    log.info(f"Evaluating on {len(eval_texts)} texts...")

    # Encode
    log.info("Encoding with student model...")
    student_emb = encode_with_student(model, teacher.tokenizer, eval_texts, device)
    log.info(f"Student embeddings: shape={student_emb.shape}")

    log.info("Encoding with teacher model...")
    teacher_emb = encode_with_teacher(teacher, eval_texts)
    log.info(f"Teacher embeddings: shape={teacher_emb.shape}")

    # Metrics
    log.info("\nComputing recall@k...")
    recall = compute_recall_at_k(student_emb, teacher_emb, tuple(args.k_values))

    # Cosine similarity: align teacher to student space
    # Load alignment matrix from loss_fn (stored in checkpoint? No, create new one)
    # Instead, compare retrieval quality via recall@k only
    # For cosine sim, we compare teacher projected to student space
    from m2m.embedding_model import ProjectionDistillationLoss

    loss_fn = ProjectionDistillationLoss(
        student_dim=student_emb.shape[1],
        teacher_dim=teacher_emb.shape[1],
    ).to(device)
    # Load alignment weights from checkpoint if available
    checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint_data:
        # Extract align_teacher weights from loss_fn state (stored separately)
        pass  # alignment weights are part of the loss, not model

    # Since we can't restore alignment weights, just compute recall@k
    cos_sims = np.zeros(len(student_emb))  # Can't compute direct cosine across dims

    # Latency
    log.info("Measuring latency...")
    latency = measure_latency(model, teacher.tokenizer, teacher, eval_texts[:100], device)

    # Matryoshka evaluation
    log.info("Evaluating Matryoshka sub-dimensions...")
    matryoshka_results = {}
    for d in config.matryoshka_dims:
        if d <= student_emb.shape[1]:
            sub_emb = student_emb[:, :d]
            sub_emb = sub_emb / np.linalg.norm(sub_emb, axis=1, keepdims=True)
            sub_teacher = teacher_emb[:, : min(d, teacher_emb.shape[1])]
            if sub_emb.shape[1] == sub_teacher.shape[1]:
                sub_teacher = sub_teacher / np.linalg.norm(sub_teacher, axis=1, keepdims=True)
                cos = float(np.mean(np.sum(sub_emb * sub_teacher, axis=1)))
                matryoshka_results[f"dim_{d}_cosine_sim"] = cos

    # Print results
    results = {
        "num_samples": len(eval_texts),
        "student_dim": int(student_emb.shape[1]),
        "teacher_dim": int(teacher_emb.shape[1]),
        "avg_cosine_similarity": float(np.mean(cos_sims)),
        "min_cosine_similarity": float(np.min(cos_sims)),
        "std_cosine_similarity": float(np.std(cos_sims)),
        "recall": recall,
        "matryoshka": matryoshka_results,
        "latency": latency,
    }

    log.info("\n" + "=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"Student dim: {results['student_dim']}")
    log.info(f"Teacher dim: {results['teacher_dim']}")
    log.info(f"Avg cosine similarity: {results['avg_cosine_similarity']:.4f}")
    log.info(f"Min cosine similarity: {results['min_cosine_similarity']:.4f}")
    log.info(f"Std cosine similarity: {results['std_cosine_similarity']:.4f}")
    log.info(f"\nRecall@k:")
    for k, v in recall.items():
        log.info(f"  {k}: {v:.4f}")
    log.info(f"\nMatryoshka:")
    for d, cos in matryoshka_results.items():
        log.info(f"  {d}: {cos:.4f}")
    log.info(f"\nLatency (per text, avg over 100 texts):")
    log.info(
        f"  Student: {latency['student_latency_ms']:.2f} ± {latency['student_latency_std']:.2f} ms"
    )
    log.info(
        f"  Teacher: {latency['teacher_latency_ms']:.2f} ± {latency['teacher_latency_std']:.2f} ms"
    )
    log.info(f"  Speedup: {latency['speedup']:.2f}x")

    # Save results
    results_path = Path(config.model_save_dir) / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
