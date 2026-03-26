# SPECS_OMNIMODAL_RAG.md — Omnimodal RAG Ingestion for M2M Vector Search

**Date:** 2026-03-21  
**Version:** 1.0  
**Hardware Target:** RTX 3090 (24GB CUDA) + RX 6650XT (8GB Vulkan) + 32GB RAM  
**M2M Latent Dim:** 640

---

## 1. SOTA Landscape — Multimodal Embedding Models

### 1.1 Model Comparison Matrix

| Model | Modalities | Dim | Params | VRAM (Inference) | License | HF Weights |
|-------|-----------|-----|--------|-------------------|---------|------------|
| **ImageBind** (Meta, 2023) | Image, Text, Audio, Video, Depth, Thermal, IMU | 1024 | ~1.2B | ~8GB | CC-BY-NC 4.0 ❌ non-commercial | `facebook/imagebind_huge` |
| **SigLIP-SO400M** (Google, 2023) | Image, Text | 1152 | ~400M | ~4GB | Apache 2.0 ✅ | `google/siglip-so400m-patch14-384` |
| **CLIP ViT-L/14@336** (OpenAI) | Image, Text | 768 | ~428M | ~3GB | MIT ✅ | `openai/clip-vit-large-patch14-336` |
| **Nomic-Embed-Vision v1.5** | Image, Text (shared space w/ text v1.5) | 768 | ~340M | ~3GB | Apache 2.0 ✅ | `nomic-ai/nomic-embed-vision-v1.5` |
| **Nomic-Embed-Text v1.5** | Text | 768 | ~137M | ~1GB | Apache 2.0 ✅ | `nomic-ai/nomic-embed-text-v1.5` |
| **BGE-M3** (BAAI, 2024) | Text (dense+sparse+ColBERT) | 1024 | ~568M | ~3GB | MIT ✅ | `BAAI/bge-m3` |
| **jina-embeddings-v3** (Jina AI, 2024) | Text | 1024 | ~570M | ~3GB | Apache 2.0 ✅ | `jinaai/jina-embeddings-v3` |
| **ColPali v1.2** (Vidore, 2024) | Document pages (as images) → multi-vector | 128 per token | ~3B | ~12GB | Apache 2.0 ✅ | `vidore/colpali-v1.2` |
| **ColQwen2.5** (Vidore, 2025) | Document pages → multi-vector | 128 per token | ~8B | ~20GB | Apache 2.0 ✅ | `vidore/colqwen2.5-v1` |
| **Ultravox v0.5** (Fixie, 2024) | Audio, Text | VLM latent | ~7B | ~16GB | Apache 2.0 ✅ | `fixie-ai/ultravox-v0_5` |

### 1.2 Key Observations

- **No single model covers all modalities well.** ImageBind is the broadest (6 modalities) but is non-commercial and research-grade.
- **Nomic-Embed is the best pragmatic choice**: text + vision in a shared embedding space, Apache 2.0, small enough for 8GB VRAM, and has strong MTEB scores (~62 MTEB, better than OpenAI CLIP ViT-B/16 at ~44).
- **ColPali/ColQwen2 are SOTA for document retrieval** but use multi-vector (ColBERT-style), not single-vector. Requires different search strategy.
- **BGE-M3** is the best text-only retriever for multilingual and long-document tasks (8192 tokens).
- **Audio embedding** remains underserved. Ultravox is capable but heavy. CLAP (Columbia, 2023) is lighter (~350M) but less maintained.

### 1.3 Benchmark Scores (from published papers/cards)

| Model | MTEB | MSCOCO I2T R@1 | Flickr30k I2T R@1 |
|-------|------|-----------------|---------------------|
| Nomic-Embed-Vision v1.5 | 62.28 | ~72.0 (DataComp) | ~88.5 |
| SigLIP-SO400M | N/A (vision model) | ~77.7 (IN1k 0-shot) | N/A |
| CLIP ViT-L/14@336 | N/A | ~75.3 | ~87.0 |
| BGE-M3 | ~64 (MTEB) | N/A | N/A |
| jina-embeddings-v3 | ~68 (MTEB) | N/A | N/A |

**Sources:** HuggingFace model cards (nomic-ai, google, BAAI, jinaai), CVPR 2023 papers.

---

## 2. Recommended Architecture

### 2.1 Design Decision: Modular Encoder + Unified Projection

**Chosen approach:** Use **best-in-class per-modality encoders** with a **learned projection head** to map all modalities into M2M's 640D space.

**Why not a single omnimodal model?**
- No single model is both commercially licensed AND covers all modalities
- Per-modality specialization gives better quality per modality
- Easier to swap/upgrade individual encoders
- Fine-tuning only requires a small projection layer

### 2.2 Pipeline Architecture

```
Input → Modality Router → Modality Encoder → ProjectionHead → Normalize → M2M Splat (μ, α, κ) → Index
```

### 2.3 Encoder Selection

| Modality | Primary Encoder | Backup | Dim → 640D Projection |
|----------|----------------|--------|----------------------|
| **Text** | `BAAI/bge-m3` (1024D) | `nomic-ai/nomic-embed-text-v1.5` (768D) | Linear: 1024→640 |
| **Image** | `nomic-ai/nomic-embed-vision-v1.5` (768D) | `google/siglip-so400m-patch14-384` (1152D) | Linear: 768→640 |
| **Audio** | `laion/larger_clap_general` (768D) | `ImageBind audio encoder` (1024D) | Linear: 768→640 |
| **PDF/Document** | `vidore/colpali-v1.2` → mean-pool to single vector | Nomic-Embed-Vision on page images | Linear: 128×N→640 |
| **Video** | Sample frames → ImageBind/Nomic Vision per-frame → temporal avg | CLIP frame sampling | Same as image |

### 2.4 Projection Layer Design

```python
class OmnimodalProjection(nn.Module):
    """Projects modality-specific embeddings into unified 640D M2M space."""
    
    def __init__(self, modality_dims: dict, output_dim: int = 640):
        super().__init__()
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
            )
            for modality, dim in modality_dims.items()
        })
    
    def forward(self, embeddings: torch.Tensor, modality: str) -> torch.Tensor:
        h = self.projections[modality](embeddings)
        return F.normalize(h, p=2, dim=-1)
```

**Training strategy for projection layers:** Contrastive alignment using multimodal pairs (text-image, text-audio) so that semantically equivalent content across modalities maps close in 640D space.

---

## 3. Training Plan

### 3.1 What Needs Training

Only the **projection layers** need training (encoders stay frozen). This is extremely lightweight:
- 4 projection heads: ~2.5M total parameters
- Fits comfortably in <1GB VRAM
- Can be trained on CPU if needed

### 3.2 Training Strategy: Contrastive Alignment

**Objective:** InfoNCE loss on modality pairs to align them in the unified 640D space.

```
L = -log(exp(sim(z_text, z_image^+) / τ) / Σ exp(sim(z_text, z_image^i) / τ))
```

**Datasets:**
| Pair Type | Dataset | Size | Source |
|-----------|---------|------|--------|
| Text-Image | MS COCO 2017 captions | 118K pairs | HuggingFace |
| Text-Image | LAION-400M (subset) | 1M pairs (sampled) | HuggingFace |
| Text-Audio | AudioCaps | 50K pairs | HuggingFace |
| Text-Audio | Clotho | 7K pairs | HuggingFace |
| Text-PDF | ViDoRe train | 127K pairs | HuggingFace |
| Text-Text (anchor) | MSCOCO duplicate captions | 118K | HuggingFace |

### 3.3 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 (projections only) |
| Batch size | 256 (gradient checkpointing) |
| Epochs | 10 |
| Temperature (τ) | 0.07 |
| Mixed precision | BF16 |
| Hardware | RTX 3090 (encoders ~8GB, projections negligible) |
| **Estimated training time** | **2-4 hours** |

### 3.4 Feasibility on RTX 3090

- **BGE-M3 encoder**: ~3GB VRAM
- **Nomic Vision encoder**: ~3GB VRAM
- **CLAP audio encoder**: ~2GB VRAM
- **Total frozen encoders**: ~8GB
- **Projection heads**: <0.1GB
- **Activations + gradients**: ~4GB
- **Total**: ~12GB — **fits comfortably in 24GB**

### 3.5 Validation Strategy

1. **Cross-modal retrieval accuracy**: text→image (MSCOCO), text→audio (AudioCaps), text→document (ViDoRe)
2. **Modality gap measurement**: CIDEr distance between modality centroids (should decrease after training)
3. **M2M retrieval quality**: After ingesting multimodal splats, run queries of each type and measure recall@k

---

## 4. Benchmark Plan

### 4.1 Benchmark Suite

| Benchmark | Metric | Task | Expected Baseline (untrained proj) | Target (after training) |
|-----------|--------|------|-------------------------------------|------------------------|
| **MSCOCO I2T** | Recall@1, R@5, R@10 | Image→Text retrieval | ~55% R@1 | ~70% R@1 |
| **MSCOCO T2I** | Recall@1, R@5, R@10 | Text→Image retrieval | ~50% R@1 | ~65% R@1 |
| **AudioCaps** | Recall@1, R@5, R@10 | Audio→Text retrieval | ~45% R@1 | ~55% R@1 |
| **ViDoRe** | Recall@1, R@5, R@10 | Document→Query retrieval | ~60% R@1 | ~75% R@1 |
| **MTEB (text-only)** | NDCG@10 | Text retrieval | ~64 (BGE-M3 native) | ~60 (with projection) |
| **MTEB (STSEncoder)** | Spearman | Semantic similarity | ~0.80 | ~0.77 |

### 4.2 SOTA Comparisons

Compare against published results for:
- **ImageBind** zero-shot cross-modal retrieval (as upper bound reference)
- **ColPali** on ViDoRe (as document retrieval SOTA)
- **BGE-M3** on MTEB (as text retrieval SOTA)
- **Nomic-Embed** on image-text benchmarks

### 4.3 Testing Methodology

1. **Ingestion test**: Encode 10K items per modality → store as M2M splats → measure ingestion throughput
2. **Retrieval accuracy**: 1K queries per modality → recall@1/5/10 against 100K splat index
3. **Cross-modal gap**: Measure cosine similarity between matched vs unmatched cross-modal pairs
4. **Latency**: Query latency on RTX 3090 (CUDA) and RX 6650XT (Vulkan) for different index sizes

---

## 5. Feasibility Assessment

### 5.1 Hardware Reality

| Task | RTX 3090 (24GB) | RX 6650XT (8GB) | CPU (32GB RAM) |
|------|------------------|-------------------|----------------|
| **Text encoding (BGE-M3)** | ✅ 3GB, fast | ⚠️ via Vulkan (PyTorch Vulkan support limited) | ✅ slow but works |
| **Image encoding (Nomic Vision)** | ✅ 3GB, fast | ⚠️ limited PyTorch Vulkan | ✅ slow but works |
| **Audio encoding (CLAP)** | ✅ 2GB, fast | ❌ unsupported | ✅ slow |
| **Document encoding (ColPali)** | ✅ ~12GB, moderate | ❌ too large | ❌ too slow |
| **Projection training** | ✅ ~12GB total | ❌ | ✅ possible but slow |
| **M2M indexing/search** | ✅ excellent | ✅ via Vulkan backend | ✅ good |
| **ColQwen2.5 encoding** | ✅ ~20GB, slow | ❌ | ❌ |

### 5.2 Timeline Estimates (Realistic)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Text + Image MVP** | 1 week | BGE-M3 + Nomic Vision → 640D projection → M2M ingestion + search |
| **Phase 2: Audio modality** | 1 week | CLAP encoder → projection alignment → audio-text retrieval |
| **Phase 3: Document modality** | 1-2 weeks | ColPali integration → page-level document retrieval |
| **Phase 4: Training alignment** | 1 week | Contrastive training of projection layers on all modalities |
| **Phase 5: Benchmarks** | 1 week | Full benchmark suite, comparison with SOTA |
| **Phase 6: Optimization** | 1-2 weeks | Quantization, batching, pipeline optimization |
| **Total** | **6-8 weeks** | Full omnimodal RAG system |

### 5.3 Honest Assessment

**What's achievable in 1 week:**
- Text + Image ingestion working end-to-end with simple linear projection
- Basic cross-modal retrieval (text→image, image→text)
- No contrastive training yet (just random projection or identity mapping)
- Quality will be lower than SOTA but functional

**What's achievable in 1 month:**
- All 4 modalities (text, image, audio, document) ingesting
- Trained projection layers with contrastive alignment
- Benchmarks showing reasonable performance
- Full integration with existing M2M codebase

**What requires 3+ months:**
- Competitive SOTA results (narrowing the gap to specialized models)
- Video ingestion with temporal modeling
- Custom fine-tuning of encoders (not just projections)
- Production-ready pipeline with error handling, streaming, etc.

### 5.4 Known Limitations

1. **Projection layer quality**: A simple linear projection will degrade from native encoder quality. The contrastive training mitigates but doesn't eliminate this.
2. **ColPali multi-vector**: ColPali produces per-patch token embeddings, not a single vector. Mean-pooling loses the ColBERT advantage. A better approach would be M2M multi-splat per page (one splat per patch), but this increases index size ~100x.
3. **Audio quality**: CLAP is less mature than vision-language models. Audio retrieval quality will be the weakest modality.
4. **Video**: No dedicated video encoder. Frame-sampling + image encoder is a rough approximation.

---

## 6. Implementation Plan

### Phase 1: Text + Image MVP (Week 1)

**New files:**
```
src/m2m/omnimodal/
├── __init__.py
├── registry.py           # Modality encoder registry
├── encoders/
│   ├── text_encoder.py   # BGE-M3 wrapper
│   ├── vision_encoder.py # Nomic Vision wrapper
│   ├── audio_encoder.py  # CLAP wrapper
│   └── document_encoder.py # ColPali wrapper
├── projection.py         # OmnimodalProjection module
├── router.py             # Modality detection + routing
└── ingestion.py          # OmnimodalIngestion pipeline
```

**Integration with existing M2M:**
- `OmnimodalIngestion` produces 640D vectors → feeds into existing `M2MEngine.store()`
- Reuses existing `M2MConfig` with `latent_dim=640`
- Works with existing CUDA/Vulkan/CPU backends for search
- Compatible with existing LangChain retriever interface

### Phase 2: Audio + Document (Week 2)

- Add CLAP encoder for audio files (mp3, wav)
- Add ColPali encoder for PDF documents (page-as-image)
- Modality auto-detection from file extensions/mimetypes

### Phase 3: Projection Training (Week 3)

- Contrastive training script using MSCOCO + AudioCaps + ViDoRe
- Evaluate modality gap before/after
- Save trained projection weights

### Phase 4: Benchmarks + Polish (Week 4)

- Automated benchmark runner
- Generate comparison report vs SOTA
- Pipeline optimization (batched encoding, async ingestion)

---

## 7. Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single vs multi-encoder | **Multi-encoder + projection** | Better quality per modality, commercially licensed, upgradeable |
| Train encoders or freeze | **Freeze encoders, train projections** | Vastly cheaper (~2.5M vs ~1B params), 2-4 hours vs days |
| Document encoding | **ColPali with mean-pool** | SOTA document retrieval, single-vector compatible |
| Audio encoding | **CLAP** | Best open audio-text alignment, lightweight |
| Dimension | **640D** | Matches existing M2M config, proven adequate |
| Training data | **Public datasets only** | Commercial use safe, reproducible |

---

## 8. References

- [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665) — Girdhar et al., CVPR 2023
- [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) — Zhai et al., 2023
- [Nomic Embed Vision v1.5](https://arxiv.org/abs/2406.18587) — Nomic AI, 2024
- [BGE-M3: Multi-Functionality, Multi-Linguality, Multi-Granularity](https://arxiv.org/abs/2402.03216) — BAAI, 2024
- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) — Faysse et al., 2024
- [CLAP: Learning Audio Concepts from Natural Language Supervision](https://arxiv.org/abs/2208.00355) — Wu et al., 2023
- [ImageBind GitHub](https://github.com/facebookresearch/ImageBind)
- [Nomic Embed Vision HF Card](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- [ColPali v1.2 HF Card](https://huggingface.co/vidore/colpali-v1.2)
- [BGE-M3 HF Card](https://huggingface.co/BAAI/bge-m3)
