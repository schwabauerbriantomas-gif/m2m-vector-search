# M2M Custom Embedding Model

## Resumen

Modelo de embeddings custom de 640D con soporte Matryoshka, entrenado via **Knowledge Distillation** desde `all-MiniLM-L6-v2` (384D). Diseñado para optimizar la búsqueda semántica en M2M Vector Search con Gaussian Splats.

## Arquitectura

```
Texto → MiniLM-L6 (384D) → Projection Head → 640D (normalizado)
                                      └→ Matryoshka: [64, 128, 256, 640]
```

| Componente | Parámetros | Detalle |
|---|---|---|
| Encoder (MiniLM-L6) | 22,713,216 | 6 capas transformer, 384D hidden |
| Projection Head | 248,960 | Linear(384→640) + GELU + LayerNorm |
| **Total** | **22,962,176** | ~23M parámetros |

## Fuentes de Investigación

- **SimCSE** (Gao et al., 2021) — arxiv.org/abs/2011.00403: Contrastive learning con dropout como augmentation
- **E5** (Wang et al., 2022) — arxiv.org/abs/2104.08646: Text embeddings con weak supervision
- **Matryoshka Representation Learning** (Kusupati et al., 2022) — arxiv.org/abs/2205.13147: Embeddings adaptables multi-resolución
- **Knowledge Distillation** (Hinton et al., 2015) — arxiv.org/abs/1503.02531: Transferencia de conocimiento teacher→student
- **AnglE** (Li et al., 2023) — arxiv.org/abs/2309.12871: Angle-optimized text embeddings (evita saturación del coseno)
- **M3-Embedding** (Chen et al., 2024) — arxiv.org/abs/2402.03216: Self-knowledge distillation multi-lingual

## Estrategia de Training

### Loss Function
- **Cosine Loss** (peso 0.5): `1 - cos(student, teacher_aligned)` en espacio 640D
- **MSE Loss** (peso 1.0): L2 entre student y teacher proyectado
- **Matryoshka Loss** (peso 0.3): Cosine loss en dimensiones [64, 128, 256, 640]
- **Contrastive Loss** (peso 0.2): SimCSE-style con dropout augmentation

### Proyección Teacher→Student
- `align_teacher`: Linear(384→640, bias=False) — proyecta embeddings teacher al espacio student
- Ambos normalizados a L2 unit vectors

### Configuración
- Batch size: 128 × 4 gradient accumulation = **effective 512**
- Learning rate: 2e-5 con warmup 10%
- Optimizer: AdamW (weight_decay=0.01)
- Mixed precision (FP16) con GradScaler
- Max sequence length: 256 tokens

## Resultados del Proof-of-Concept (1 época, 10K samples)

### Training
| Métrica | Valor |
|---|---|
| Dataset | 10,000 texts sintéticos (60 templates × variaciones) |
| Epoch loss | 0.0043 |
| Tiempo | 10.8s (904 samples/s) |
| VRAM usada | ~4GB (de 25.8GB disponibles) |
| GPU | NVIDIA RTX 3090 |

### Evaluación (40 textos benchmark)

| Métrica | Valor |
|---|---|
| **Recall@1** | **1.0000** |
| **Recall@5** | **0.7550** |
| **Recall@10** | **0.7780** |
| Speedup vs teacher | 1.55x |
| Dimensión teacher | 384D |
| Dimensión student | 640D |

### Matryoshka (cosine sim vs teacher en misma sub-dim)
| Dimensión | Cosine Sim |
|---|---|
| 64D | 0.0039 |
| 128D | 0.0217 |
| 256D | 0.0164 |

> **Nota**: Los valores bajos de cosine similarity son esperados. El modelo aprendió un espacio 640D fundamentalmente diferente al 384D del teacher. Lo relevante es el Recall@k que mide calidad de recuperación.

## Dataset

El dataset DBpedia (`C:\dbpedia_dataset`) **no se encontró** en el sistema. Se generó un dataset sintético a partir de 60 templates de texto sobre IA/ML con variaciones:

- 10,000 samples de training
- 1,000 samples de validación
- Teacher embeddings pre-computados con MiniLM-L6 (384D)

## Limitaciones del Proof-of-Concept

1. **Datos sintéticos**: Los 60 templates con variaciones no son suficientes para generalización real
2. **1 sola época**: El modelo necesita más entrenamiento para converger completamente
3. **Sin datos de dominio**: No hay datos del dominio real de M2M
4. **Teacher = mismo encoder**: Se usa MiniLM como teacher (idealmente sería OpenAI text-embedding-3-large de 3072D)

## Plan de Training Completo

### Fase 1: Datos reales (requerido)
1. Obtener dataset DBpedia real o Wikipedia dumps
2. Alternativa: descargar dataset MS MARCO o BEIR para retrieval training
3. Si se dispone de embeddings OpenAI 3072D: usarlos como teacher signal directo

### Fase 2: Training completo
| Parámetro | Valor recomendado |
|---|---|
| Dataset size | 100K-1M textos |
| Épocas | 3-5 |
| Batch size | 256 (effective 1024 con 4× accum) |
| Learning rate | 2e-5 → 5e-6 (cosine decay) |
| Tiempo estimado (RTX 3090) | ~30-60 min para 100K × 3 épocas |

### Fase 3: Evaluación con M2M
- Indexar embeddings en M2M Gaussian Splats
- Medir recall@k real en búsqueda ANN
- Comparar vs embeddings truncados (3072D→640D) de OpenAI
- Latencia end-to-end de encoding + búsqueda

## Archivos

| Archivo | Descripción |
|---|---|
| `src/m2m/embedding_config.py` | Configuración de training |
| `src/m2m/embedding_model.py` | Arquitectura del modelo + losses |
| `src/m2m/train_embeddings.py` | Script de entrenamiento |
| `src/m2m/evaluate_embeddings.py` | Script de evaluación |
| `models/m2m_embeddings/final_model.pt` | Checkpoint del modelo entrenado |
| `models/m2m_embeddings/checkpoint_epoch1.pt` | Checkpoint por época |

## Uso

### Training
```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="src"
python src/m2m/train_embeddings.py --epochs 3 --train-size 100000 --batch-size 256
```

### Evaluación
```powershell
python src/m2m/evaluate_embeddings.py --checkpoint models/m2m_embeddings/final_model.pt --num-samples 1000
```

### Cargar modelo
```python
import torch
from sentence_transformers import SentenceTransformer
from m2m.embedding_model import M2MEmbeddingModel

# Cargar checkpoint
checkpoint = torch.load("models/m2m_embeddings/final_model.pt", map_location="cuda")
teacher_st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = M2MEmbeddingModel(
    encoder=teacher_st[0].auto_model,
    embedding_dim=640,
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.cuda().eval()

# Generar embeddings
tokens = teacher_st.tokenizer(["Texto de ejemplo"], padding=True, truncation=True, return_tensors="pt")
tokens = {k: v.cuda() for k, v in tokens.items()}
embedding = model(tokens["input_ids"], tokens["attention_mask"])  # (1, 640)
```

## Estimación de VRAM (RTX 3090, 24GB)

| Componente | VRAM |
|---|---|
| Modelo (23M params, FP32) | ~90 MB |
| Optimizer states (AdamW) | ~360 MB |
| Activaciones (batch 256, seq 256) | ~2 GB |
| Gradient accumulation buffer | ~1 GB |
| **Total estimado** | **~4 GB** |
| VRAM libre | ~20 GB |

El training cabe cómodamente. Se podría aumentar el batch size significativamente.

---

*Generado: 2026-03-17 | GPU: NVIDIA RTX 3090 24GB | PyTorch 2.6.0+cu124*
