"""
M2M Embedding Training Script
==============================
Knowledge distillation: MiniLM-L6 → 640D embeddings with Matryoshka.

Training data: Generated synthetic text pairs with teacher embeddings from
a pre-trained sentence-transformer model (as proxy for OpenAI embeddings).

Usage:
    $env:PYTHONIOENCODING="utf-8"
    $env:PYTHONPATH="src"
    python src/m2m/train_embeddings.py --epochs 1 --train-size 10000
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

# Local imports
try:
    from m2m.embedding_config import EmbeddingConfig, DatasetConfig
    from m2m.embedding_model import M2MEmbeddingModel, ProjectionDistillationLoss
except ImportError:
    # Fallback for direct execution
    embedding_config_path = PROJECT_ROOT / "src" / "m2m" / "embedding_config.py"
    embedding_model_path = PROJECT_ROOT / "src" / "m2m" / "embedding_model.py"
    if embedding_config_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("embedding_config", str(embedding_config_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        EmbeddingConfig = mod.EmbeddingConfig
        DatasetConfig = mod.DatasetConfig
    
    if embedding_model_path.exists():
        spec = importlib.util.spec_from_file_location("embedding_model", str(embedding_model_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        M2MEmbeddingModel = mod.M2MEmbeddingModel
        ProjectionDistillationLoss = mod.ProjectionDistillationLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Synthetic Training Dataset
# ============================================================

class SyntheticEmbeddingDataset(Dataset):
    """
    Generate synthetic text pairs with teacher embeddings for KD training.
    
    Since the DBpedia dataset is not available, we generate training data
    from Wikipedia-style text snippets using the teacher model to create
    the target embeddings.
    """
    
    # Template sentences covering diverse topics
    TEXT_TEMPLATES = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Natural language processing deals with the interaction between computers and human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Deep learning uses multiple layers to progressively extract higher-level features from raw input.",
        "Reinforcement learning is an area of machine learning where an agent learns to make decisions.",
        "Transfer learning allows models trained on one task to be reused for a different but related task.",
        "Transformer models have revolutionized natural language processing with self-attention mechanisms.",
        "Convolutional neural networks are particularly effective for image recognition and classification.",
        "Generative adversarial networks consist of two neural networks competing in a game-theoretic scenario.",
        "BERT is a transformer-based model designed to pre-train deep bidirectional representations from text.",
        "Word embeddings map words to dense vector representations in a continuous vector space.",
        "The attention mechanism allows models to focus on relevant parts of the input when producing output.",
        "Gradient descent is an optimization algorithm used to minimize the loss function in neural networks.",
        "Batch normalization improves training stability and accelerates convergence in deep neural networks.",
        "Recurrent neural networks are designed to handle sequential data by maintaining a hidden state.",
        "Autoencoders learn efficient codings of unlabeled data for dimensionality reduction or feature learning.",
        "Knowledge distillation transfers knowledge from a large model to a smaller one for efficient inference.",
        "Embedding space is a mathematical space where similar items are close together.",
        "Vector search finds approximate nearest neighbors in high-dimensional embedding spaces.",
        "Gaussian splats provide a probabilistic representation for vector search with uncertainty quantification.",
        "Product quantization compresses high-dimensional vectors for efficient similarity search.",
        "Hierarchical navigable small world graphs enable fast approximate nearest neighbor search.",
        "Inverted file indexes partition the vector space for scalable similarity search.",
        "Cosine similarity measures the cosine of the angle between two non-zero vectors.",
        "Euclidean distance is the straight-line distance between two points in Euclidean space.",
        "Dot product similarity measures alignment between vectors and is efficient to compute.",
        "Semantic search retrieves documents based on meaning rather than keyword matching.",
        "Cross-encoder models jointly encode query-document pairs for more accurate relevance scoring.",
        "Bi-encoders independently encode queries and documents for efficient retrieval at scale.",
        "Hard negative mining improves contrastive learning by selecting difficult negative examples.",
        "Data augmentation techniques like random dropout and cropping improve embedding model robustness.",
        "Temperature scaling controls the sharpness of probability distributions in contrastive learning.",
        "Multi-task learning improves embedding quality by sharing representations across related tasks.",
        "The curse of dimensionality refers to various phenomena that arise when analyzing data in high dimensions.",
        "Dimensionality reduction techniques like PCA and t-SNE help visualize high-dimensional embeddings.",
        "Approximate nearest neighbor search trades accuracy for speed in large-scale retrieval systems.",
        "Information retrieval systems rank documents by relevance to a user's query.",
        "The vector space model represents documents and queries as vectors in a high-dimensional space.",
        "Latent semantic analysis uncovers the latent structure in a corpus of text documents.",
        "Topic modeling discovers abstract topics in a collection of documents using statistical methods.",
        "Named entity recognition identifies and classifies named entities in text into predefined categories.",
        "Sentiment analysis determines whether a piece of text expresses a positive, negative, or neutral opinion.",
        "Question answering systems automatically answer questions posed by humans in natural language.",
        "Text summarization creates concise summaries of longer documents while preserving key information.",
        "Language models predict the probability of a sequence of words appearing in natural language.",
        "Tokenization breaks text into smaller units called tokens for processing by language models.",
        "Positional encoding provides positional information to transformer models that lack recurrence.",
        "Self-attention computes attention scores between all positions in a sequence simultaneously.",
        "Multi-head attention allows the model to jointly attend to information from different subspaces.",
        "The residual connection helps train very deep networks by adding input to the output of a layer.",
        "Dropout randomly deactivates neurons during training to prevent overfitting.",
        "The learning rate controls how much to update model parameters during gradient descent optimization.",
        "Weight decay adds L2 regularization to prevent model weights from growing too large.",
        "Early stopping halts training when validation performance stops improving to prevent overfitting.",
        "Mixed precision training uses float16 and float32 to speed up training and reduce memory usage.",
        "Gradient clipping prevents exploding gradients by capping gradient values during backpropagation.",
        "The softmax function converts logits to a probability distribution over categories.",
        "Cross-entropy loss measures the difference between predicted and true probability distributions.",
        "Backpropagation computes gradients of the loss function with respect to network weights.",
        "Stochastic gradient descent updates parameters using a single or small batch of training examples.",
        "Adam optimizer combines momentum and adaptive learning rates for efficient optimization.",
        "Learning rate scheduling adjusts the learning rate during training to improve convergence.",
        "Hyperparameter tuning searches for the best configuration of model settings.",
    ]
    
    def __init__(
        self,
        teacher_model: SentenceTransformer,
        num_samples: int = 10000,
        augment: bool = True,
        seed: int = 42,
    ):
        self.teacher = teacher_model
        self.num_samples = num_samples
        self.augment = augment
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Generate diverse texts by combining templates
        self.texts = self._generate_texts(num_samples)
        
        # Pre-compute teacher embeddings
        log.info(f"Computing teacher embeddings for {len(self.texts)} texts...")
        self.teacher_embeddings = teacher_model.encode(
            self.texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        log.info(f"Teacher embeddings shape: {self.teacher_embeddings.shape}")
        assert self.teacher_embeddings.shape[1] == 384, \
            f"Expected 384D, got {self.teacher_embeddings.shape[1]}"
    
    def _generate_texts(self, num_samples: int) -> List[str]:
        """Generate diverse training texts from templates."""
        texts = []
        n_templates = len(self.TEXT_TEMPLATES)
        
        # Cycle through templates with variations
        for i in range(num_samples):
            base = self.TEXT_TEMPLATES[i % n_templates]
            
            # Create variation
            variation = self.rng.choice([
                base,
                f"Consider this: {base}",
                f"In the field of AI, {base.lower()}",
                f"Research shows that {base.lower()}",
                f"A fundamental concept: {base}",
                f"{base} This has significant implications for the future.",
                f"When studying AI, one learns that {base.lower()}",
                f"Experts agree that {base.lower()}",
            ])
            texts.append(variation)
        
        return texts
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        text = self.texts[idx]
        teacher_emb = torch.tensor(self.teacher_embeddings[idx], dtype=torch.float32)
        
        # Optional augmentation: create a paired text for contrastive learning
        if self.augment and self.rng.random() > 0.3:
            # Random dropout augmentation (simulate SimCSE)
            words = text.split()
            if len(words) > 5:
                n_drop = self.rng.randint(1, max(1, len(words) // 5))
                drop_indices = self.rng.sample(range(len(words)), n_drop)
                augmented = " ".join(w for i, w in enumerate(words) if i not in drop_indices)
                return {
                    "text": text,
                    "augmented_text": augmented,
                    "teacher_embedding": teacher_emb,
                    "is_positive_pair": True,
                }
        
        return {
            "text": text,
            "augmented_text": text,  # same text = positive pair
            "teacher_embedding": teacher_emb,
            "is_positive_pair": False,
        }


def collate_fn(batch: List[Dict], tokenizer) -> Dict:
    """Collate batch with tokenizer."""
    texts = [item["text"] for item in batch]
    augmented_texts = [item["augmented_text"] for item in batch]
    teacher_embs = torch.stack([item["teacher_embedding"] for item in batch])
    is_positive = torch.tensor([item["is_positive_pair"] for item in batch])
    
    # Tokenize
    text_tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    aug_tokens = tokenizer(
        augmented_texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    
    return {
        "input_ids": text_tokens["input_ids"],
        "attention_mask": text_tokens["attention_mask"],
        "aug_input_ids": aug_tokens["input_ids"],
        "aug_attention_mask": aug_tokens["attention_mask"],
        "teacher_embeddings": teacher_embs,
        "is_positive": is_positive,
    }


def get_teacher_embeddings(
    model: M2MEmbeddingModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Get encoder-only output (teacher base embeddings, 384D)."""
    outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    pooled = sum_embeddings / sum_mask
    return F.normalize(pooled, p=2, dim=-1)


def train_one_epoch(
    model: M2MEmbeddingModel,
    loss_fn: ProjectionDistillationLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    tokenizer,
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    step = 0
    epoch_start = time.time()
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast(enabled=scaler.is_enabled()):
            # Student forward pass
            student_emb = model(batch["input_ids"], batch["attention_mask"])
            
            # Get teacher base embeddings (from encoder, before projection)
            teacher_emb = get_teacher_embeddings(
                model, batch["input_ids"], batch["attention_mask"]
            )
            
            # KD loss: student (640D) vs teacher (384D)
            loss, loss_dict = loss_fn(student_emb, teacher_emb)
            
            # Contrastive loss with augmented pairs
            if batch["is_positive"].any():
                aug_emb = model(batch["aug_input_ids"], batch["aug_attention_mask"])
                contrastive_sim = F.cosine_similarity(student_emb, aug_emb, dim=-1)
                positive_mask = batch["is_positive"].float()
                contrastive_loss = -(contrastive_sim * positive_mask).mean()
                loss = loss + 0.2 * contrastive_loss
                loss_dict["contrastive"] = contrastive_loss.item()
            
            loss = loss / 1.0  # gradient accumulation handled below
        
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % 4 == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            step += 1
            total_loss += loss.item() * 4  # undo division
            total_samples += batch["input_ids"].size(0) * 4
            
            if step % 20 == 0:
                avg_loss = total_loss / total_samples
                elapsed = time.time() - epoch_start
                log.info(
                    f"  Epoch {epoch} | Step {step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Cos: {loss_dict.get('cosine', 0):.4f} | "
                    f"MSE: {loss_dict.get('mse', 0):.4f} | "
                    f"Speed: {total_samples/elapsed:.0f} samples/s"
                )
    
    # Handle remaining gradients
    if (batch_idx + 1) % 4 != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    elapsed = time.time() - epoch_start
    metrics = {
        "epoch_loss": total_loss / max(total_samples, 1),
        "epoch_time": elapsed,
        "samples_per_second": total_samples / max(elapsed, 1),
        "steps": step,
    }
    
    log.info(
        f"Epoch {epoch} complete: loss={metrics['epoch_loss']:.4f}, "
        f"time={elapsed:.1f}s, speed={metrics['samples_per_second']:.0f} samples/s"
    )
    
    return metrics


@torch.no_grad()
def evaluate(
    model: M2MEmbeddingModel,
    loss_fn: ProjectionDistillationLoss,
    dataloader: DataLoader,
    device: torch.device,
    k_values: Tuple[int, ...] = (1, 5, 10, 50, 100),
) -> Dict:
    """Evaluate model quality: loss + recall@k vs teacher embeddings."""
    model.eval()
    
    all_student_emb = []
    all_teacher_emb = []
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        student_emb = model(batch["input_ids"], batch["attention_mask"])
        teacher_emb = get_teacher_embeddings(
            model, batch["input_ids"], batch["attention_mask"]
        )
        
        loss, _ = loss_fn(student_emb, teacher_emb)
        total_loss += loss.item()
        n_batches += 1
        
        all_student_emb.append(student_emb.cpu())
        all_teacher_emb.append(teacher_emb.cpu())
    
    all_student = torch.cat(all_student_emb, dim=0)
    all_teacher = torch.cat(all_teacher_emb, dim=0)
    
    # Compute recall@k: for each query, do student nearest neighbors match teacher's?
    recall_results = {}
    n = len(all_student)
    sample_size = min(n, 2000)  # subsample for speed
    indices = np.random.choice(n, sample_size, replace=False)
    
    query_student = all_student[indices].to(device)
    query_teacher = all_teacher[indices].to(device)
    db_student = all_student.to(device)
    db_teacher = all_teacher.to(device)
    
    # Align teacher to student space for fair comparison
    db_teacher_aligned = F.normalize(loss_fn.align_teacher(db_teacher), p=2, dim=-1)
    query_teacher_aligned = F.normalize(loss_fn.align_teacher(query_teacher), p=2, dim=-1)
    
    # Student's nearest neighbors (in 640D)
    student_sim = torch.mm(query_student, db_student.T)
    # Teacher's nearest neighbors (aligned to student space)
    teacher_sim = torch.mm(query_teacher_aligned, db_teacher_aligned.T)
    
    for k in k_values:
        if k >= n:
            continue
        student_nn = student_sim.topk(k, dim=1).indices
        teacher_nn = teacher_sim.topk(k, dim=1).indices
        
        # Recall@k: fraction of teacher's top-1 in student's top-k
        teacher_top1 = teacher_nn[:, :1]
        recall = 0.0
        for i in range(sample_size):
            recall += float(teacher_top1[i] in student_nn[i])
        recall_results[f"recall@{k}"] = float(recall / sample_size)
    
    # Average cosine similarity: project student to teacher space via alignment
    # Use the loss function's alignment matrix
    avg_cosine = 0.0
    n_eval = len(all_student)
    # Compute cosine in a shared space: project both to same dim
    # Since dims differ, compute via batched alignment
    for start in range(0, n_eval, 512):
        end = min(start + 512, n_eval)
        s = all_student[start:end].to(device)
        t = all_teacher[start:end].to(device)
        # Align teacher to student space via loss function's learned projection
        t_aligned = F.normalize(loss_fn.align_teacher(t), p=2, dim=-1)
        cos = F.cosine_similarity(s, t_aligned, dim=-1).mean().item()
        avg_cosine += cos * (end - start)
    avg_cosine /= max(n_eval, 1)
    
    metrics = {
        "eval_loss": total_loss / max(n_batches, 1),
        "avg_cosine_similarity": avg_cosine,
        **recall_results,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train M2M custom embedding model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--embedding-dim", type=int, default=640)
    parser.add_argument("--save-dir", type=str, default="models/m2m_embeddings")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB")
    
    # Config
    config = EmbeddingConfig(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        train_size=args.train_size,
        val_size=args.val_size,
        use_amp=not args.no_amp,
    )
    log.info(f"Config: {json.dumps(vars(config), indent=2, default=str)}")
    
    # ============================================================
    # 1. Load teacher model
    # ============================================================
    log.info("Loading teacher model (sentence-transformers/all-MiniLM-L6-v2)...")
    teacher_st = SentenceTransformer(config.teacher_model)
    teacher_st.eval()
    log.info(f"Teacher model loaded. Output dim: {teacher_st.get_sentence_embedding_dimension()}")
    
    # ============================================================
    # 2. Create student model
    # ============================================================
    log.info("Creating student model with projection head...")
    
    # Get the transformer from sentence-transformers
    auto_model = teacher_st[0].auto_model  # The actual transformer
    
    student = M2MEmbeddingModel(
        encoder=auto_model,
        embedding_dim=config.embedding_dim,
        matryoshka_dims=config.matryoshka_dims,
        freeze_encoder=False,  # Fine-tune entire model
    )
    
    params = student.get_num_params()
    log.info(f"Student model params: {json.dumps(params, indent=2)}")
    student = student.to(device)
    
    # ============================================================
    # 3. Create datasets
    # ============================================================
    log.info(f"Creating training dataset ({config.train_size} samples)...")
    train_dataset = SyntheticEmbeddingDataset(
        teacher_model=teacher_st,
        num_samples=config.train_size,
        augment=True,
        seed=config.seed,
    )
    
    log.info(f"Creating validation dataset ({config.val_size} samples)...")
    val_dataset = SyntheticEmbeddingDataset(
        teacher_model=teacher_st,
        num_samples=config.val_size,
        augment=False,
        seed=config.seed + 100,
    )
    
    # Tokenizer
    tokenizer = teacher_st.tokenizer
    
    # DataLoaders
    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate, pin_memory=True,
    )
    
    # ============================================================
    # 4. Loss, optimizer, scheduler
    # ============================================================
    teacher_dim = teacher_st.get_sentence_embedding_dimension()
    
    loss_fn = ProjectionDistillationLoss(
        student_dim=config.embedding_dim,
        teacher_dim=teacher_dim,
        mse_weight=config.mse_loss_weight,
        cosine_weight=config.cosine_loss_weight,
        matryoshka_weight=config.matryoshka_loss_weight,
        matryoshka_dims=config.matryoshka_dims,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    total_steps = len(train_loader) // config.gradient_accumulation_steps * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    
    # ============================================================
    # 5. Training loop
    # ============================================================
    log.info(f"Starting training: {config.num_epochs} epochs, {total_steps} steps")
    log.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    all_metrics = []
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        log.info(f"\n{'='*60}")
        log.info(f"EPOCH {epoch}/{config.num_epochs}")
        log.info(f"{'='*60}")
        
        train_metrics = train_one_epoch(
            model=student,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            epoch=epoch,
            tokenizer=tokenizer,
        )
        
        eval_metrics = evaluate(
            model=student,
            loss_fn=loss_fn,
            dataloader=val_loader,
            device=device,
            k_values=config.eval_k_values,
        )
        
        epoch_metrics = {"epoch": epoch, **train_metrics, **eval_metrics}
        all_metrics.append(epoch_metrics)
        
        log.info(f"Eval - Loss: {eval_metrics['eval_loss']:.4f}, "
                f"CosSim: {eval_metrics['avg_cosine_similarity']:.4f}")
        for k, v in eval_metrics.items():
            if k.startswith("recall@"):
                log.info(f"  {k}: {v:.4f}")
        
        # Save checkpoint
        save_path = Path(config.model_save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint, save_path / f"checkpoint_epoch{epoch}.pt")
        log.info(f"Checkpoint saved to {save_path / f'checkpoint_epoch{epoch}.pt'}")
    
    total_time = time.time() - start_time
    log.info(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Save final model
    final_path = Path(config.model_save_dir) / "final_model.pt"
    torch.save({
        "model_state_dict": student.state_dict(),
        "config": vars(config),
        "training_metrics": all_metrics,
        "total_training_time": total_time,
    }, final_path)
    log.info(f"Final model saved to {final_path}")
    
    # Print summary
    log.info("\n" + "=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    log.info(f"Total time: {total_time:.1f}s")
    for m in all_metrics:
        log.info(f"Epoch {m['epoch']}: loss={m.get('epoch_loss', 0):.4f}, "
                f"cos_sim={m.get('avg_cosine_similarity', 0):.4f}")
    
    return all_metrics


if __name__ == "__main__":
    main()
