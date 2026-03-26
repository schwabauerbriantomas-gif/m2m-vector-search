"""
M2M Custom Embedding Model
===========================
Lightweight 640D embedding model with Matryoshka representations,
trained via knowledge distillation from larger teacher embeddings.

Architecture: MiniLM-L6 -> Projection Head -> 640D (Matryoshka: 64/128/256/640)

Note: This module requires torch/sentence-transformers. If not installed,
classes will not be defined to allow module import for documentation generation.
"""

from typing import List, Optional, Tuple

# Heavy dependencies - may not be installed in all environments
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


if _HAS_TORCH:

    class ProjectionHead(nn.Module):
        """Projection head: base_dim -> embedding_dim with Matryoshka support."""

        def __init__(
            self,
            input_dim: int = 384,
            output_dim: int = 640,
            matryoshka_dims: Optional[Tuple[int, ...]] = None,
        ):
            super().__init__()
            self.output_dim = output_dim
            self.matryoshka_dims = matryoshka_dims or []

            # Main projection: input_dim -> output_dim
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
            )

            # Normalize final output
            self.layer_norm = nn.LayerNorm(output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, input_dim) -> (batch, output_dim)"""
            h = self.projection(x)
            h = self.layer_norm(h)
            return F.normalize(h, p=2, dim=-1)

        def forward_matryoshka(self, x: torch.Tensor) -> List[torch.Tensor]:
            """Returns embeddings truncated to each Matryoshka dimension, all normalized."""
            h = self.projection(x)
            h = self.layer_norm(h)

            outputs = []
            for d in self.matryoshka_dims:
                truncated = h[:, :d]
                outputs.append(F.normalize(truncated, p=2, dim=-1))

            return outputs

    class M2MEmbeddingModel(nn.Module):
        """
        M2M Custom Embedding Model.

        Uses a pre-trained sentence-transformer encoder + projection head
        to produce 640D embeddings with optional Matryoshka representations.
        """

        def __init__(
            self,
            encoder: nn.Module,
            embedding_dim: int = 640,
            matryoshka_dims: Optional[Tuple[int, ...]] = None,
            freeze_encoder: bool = False,
        ):
            super().__init__()
            self.encoder = encoder
            self.embedding_dim = embedding_dim
            self.matryoshka_dims = matryoshka_dims or (64, 128, 256, 640)

            # Get encoder output dimension
            encoder_dim = self._get_encoder_dim()

            # Projection head
            self.projection = ProjectionHead(
                input_dim=encoder_dim,
                output_dim=embedding_dim,
                matryoshka_dims=self.matryoshka_dims,
            )

            # Freeze encoder if specified
            if freeze_encoder:
                self._freeze_encoder()

        def _get_encoder_dim(self) -> int:
            """Detect encoder output dimension."""
            # Try common config attributes
            for attr in ["hidden_size", "dim", "d_model", "embedding_dim"]:
                if hasattr(self.encoder, attr):
                    return getattr(self.encoder, attr)
            if hasattr(self.encoder, "config"):
                cfg = self.encoder.config
                if hasattr(cfg, "hidden_size"):
                    return cfg.hidden_size
            # Fallback: MiniLM-L6 has 384
            return 384

        def _freeze_encoder(self):
            """Freeze encoder parameters."""
            for param in self.encoder.parameters():
                param.requires_grad = False

        def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Encode text tokens -> 640D embeddings.

            Args:
                input_ids: (batch, seq_len)
                attention_mask: (batch, seq_len)

            Returns:
                (batch, embedding_dim) normalized embeddings
            """
            # Get encoder output (use CLS token or mean pooling)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask

            # Project to embedding_dim
            return self.projection(pooled)

        def encode_matryoshka(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> List[torch.Tensor]:
            """Encode with Matryoshka representations at multiple granularities."""
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask

            return self.projection.forward_matryoshka(pooled)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass returning 640D embeddings."""
            return self.encode(input_ids, attention_mask)

        def get_num_params(self) -> dict:
            """Count trainable and total parameters."""
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            encoder_params = sum(p.numel() for p in self.encoder.parameters())
            projection_params = sum(p.numel() for p in self.projection.parameters())
            return {
                "total": total,
                "trainable": trainable,
                "encoder": encoder_params,
                "projection": projection_params,
            }

    class DistillationLoss(nn.Module):
        """
        Combined loss for knowledge distillation of embeddings.

        Components:
        - MSE loss: L2 distance to teacher embeddings
        - Cosine loss: 1 - cosine_similarity( student, teacher )
        - Matryoshka loss: same losses at each sub-dimension
        """

        def __init__(
            self,
            mse_weight: float = 1.0,
            cosine_weight: float = 0.5,
            matryoshka_weight: float = 0.3,
            matryoshka_dims: Optional[Tuple[int, ...]] = None,
        ):
            super().__init__()
            self.mse_weight = mse_weight
            self.cosine_weight = cosine_weight
            self.matryoshka_weight = matryoshka_weight
            self.matryoshka_dims = matryoshka_dims or (64, 128, 256, 640)
            self.mse_loss = nn.MSELoss()

        def forward(
            self,
            student_embeddings: torch.Tensor,
            teacher_embeddings: torch.Tensor,
        ) -> Tuple[torch.Tensor, dict]:
            """
            Args:
                student_embeddings: (batch, 640)
                teacher_embeddings: (batch, teacher_dim) - already normalized

            Returns:
                total_loss, loss_dict with individual components
            """
            losses = {}

            # MSE loss on full embedding (project teacher down first won't work,
            # we project student up or use same dim)
            # Since teacher might have different dim, we use cosine loss primarily
            losses["mse"] = self.mse_loss(student_embeddings, student_embeddings)  # placeholder
            losses["cosine"] = (
                1 - F.cosine_similarity(student_embeddings, student_embeddings, dim=-1)
            ).mean()  # placeholder

            # For real KD: teacher should be projected to same dim
            # We'll handle this in the training loop with proper teacher projection
            total = 0.0
            return total, losses

    class ProjectionDistillationLoss(nn.Module):
        """
        Knowledge distillation loss when teacher has different embedding dimension.

        Strategy: Learn a shared projection space. The student projects to 640D,
        and we minimize cosine distance + MSE in teacher's space.
        """

        def __init__(
            self,
            student_dim: int = 640,
            teacher_dim: int = 384,
            mse_weight: float = 1.0,
            cosine_weight: float = 1.0,
            matryoshka_weight: float = 0.3,
            matryoshka_dims: Optional[Tuple[int, ...]] = None,
        ):
            super().__init__()
            self.mse_weight = mse_weight
            self.cosine_weight = cosine_weight
            self.matryoshka_weight = matryoshka_weight
            self.matryoshka_dims = matryoshka_dims or (64, 128, 256, 640)

            # Learnable alignment: project student to teacher space for loss
            self.align_student = nn.Linear(student_dim, teacher_dim, bias=False)
            # Or project teacher to student space
            self.align_teacher = nn.Linear(teacher_dim, student_dim, bias=False)
            self.mse_loss = nn.MSELoss()

        def forward(
            self,
            student_embeddings: torch.Tensor,
            teacher_embeddings: torch.Tensor,
        ) -> Tuple[torch.Tensor, dict]:
            """
            Args:
                student_embeddings: (batch, 640) normalized
                teacher_embeddings: (batch, 384) normalized

            Returns:
                total_loss, loss_dict
            """
            losses = {}

            # Align teacher to student space
            teacher_aligned = self.align_teacher(teacher_embeddings)
            teacher_aligned = F.normalize(teacher_aligned, p=2, dim=-1)

            # Cosine loss (main KD signal)
            cosine_sim = F.cosine_similarity(student_embeddings, teacher_aligned, dim=-1)
            losses["cosine"] = (1 - cosine_sim).mean()

            # MSE loss
            losses["mse"] = self.mse_loss(student_embeddings, teacher_aligned)

            # Matryoshka losses (at sub-dimensions)
            matryoshka_loss = torch.tensor(0.0, device=student_embeddings.device)
            for d in self.matryoshka_dims:
                student_sub = F.normalize(student_embeddings[:, :d], p=2, dim=-1)
                teacher_sub = F.normalize(teacher_aligned[:, :d], p=2, dim=-1)
                sub_cos = (1 - F.cosine_similarity(student_sub, teacher_sub, dim=-1)).mean()
                matryoshka_loss = matryoshka_loss + sub_cos
            matryoshka_loss = matryoshka_loss / len(self.matryoshka_dims)
            losses["matryoshka"] = matryoshka_loss

            # Total
            total = (
                self.cosine_weight * losses["cosine"]
                + self.mse_weight * losses["mse"]
                + self.matryoshka_weight * losses["matryoshka"]
            )
            losses["total"] = total

            return total, losses
