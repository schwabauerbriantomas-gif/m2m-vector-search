# M2M Vector Search Engine

[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-passed-brightgreen.svg)](#-benchmarks)

> **Machine-to-Memory (M2M) Engine & Gaussian Splat Vector Store**
>
> A vector database with hierarchical retrieval for local-first applications.
> Now available in two explicit flavors: The minimal **SQLite-style** for Edge, and the **Advanced Agent-style** for intelligence.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Two Modes of Operation](#-two-modes-of-operation)
  - [SimpleVectorDB (Edge / "SQLite" approach)](#1-simplevectordb)
  - [AdvancedVectorDB (Agentic approach)](#2-advancedvectordb)
- [Architecture & 3-Tier Memory](#-architecture)
- [Benchmarks](#-benchmarks)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [License](#-license)

---

## 🎯 Overview

**M2M Vector Search** is a vector database built on Gaussian Splats with hierarchical retrieval (HRM2). Designed originally for generative exploration and Self-Organized Criticality (SOC), it has been refined to offer production-ready profiles for both sheer speed (Edge computing) and complex reasoning (Agents).

### Core Engine Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Retrieval (HRM2)** | Two-level clustering (Level 1 Coarse, Level 2 Fine) for sub-millisecond searches. |
| **Gaussian Splats** | Full latent representation (μ, α, κ). |
| **Local-First** | No cloud dependencies, pure local Python/PyTorch logic. |
| **GPU Acceleration** | Optional true Vulkan compute shader acceleration for MoE routers. |

---

## 🌐 Omnimodal & Multimodal Ready

M2M does not care about the *source* of your vectors; it seamlessly stores and routes high-dimensional embeddings from any modality. By pairing M2M with state-of-the-art embedding models, you can achieve true omnimodal retrieval.

### Supported Data Formats & Best Practices
| Modality | Recommended Embedding Model | Format / Best Practice |
|----------|-----------------------------|------------------------|
| **Text** | OpenAI `text-embedding-3`, BGE, `all-MiniLM` | Chunks of Markdown, raw string text, JSON blobs. Normalize text before embedding. |
| **Images** | OpenAI CLIP, SigLIP | `.png`, `.jpg`, `.webp`. Normalize vectors to S^D space to perfectly fit M2M's spherical mapping. |
| **Audio** | ImageBind, Whisper-based encoders | `.wav`, `.mp3`. Embed 5-second acoustic slices to match natural HRM2 clustering. |
| **Video** | VideoMAE, ImageBind | Frame aggregations or temporal tokens. Group frames as "splat clusters". |
| **Spatial/3D**| PointNet++, 3D Gaussian Splatting | Pass raw splat features (μ, α, κ) directly for 3D routing applications. |
| **Telemetry**| Time2Vec | Server logs, IoT sensory data encoded to hyperspheres. |

---

## 🌓 Two Modes of Operation

We realized that applications need completely different things. Standard RAG needs a blazing fast, "dumb" vector store. Autonomous agents need exploratory latent spaces. Thus, M2M provides two compatible, interchangeable interfaces:

### 1. SimpleVectorDB 
*"The SQLite of Vector DBs"*

Designed for raw edge computing and pure embedding retrieval. It strips away all advanced mechanics (generative sampling, memory tiering, entropy tracking) to maximize throughput and minimize RAM/VRAM footprint.

**Best for**: RAG workflows, embedding lookup, static local vector caches.

```python
import torch
from m2m import SimpleVectorDB

# Zero-configuration initialization
db = SimpleVectorDB(device='cpu')

# Add embeddings dynamically
db.add(torch.randn(10000, 640))

# Blazing fast hierarchical search
results = db.search(torch.randn(1, 640), k=10)

# Save/Load your index instantly
# db.load("vector_cache.bin")
```

### 2. AdvancedVectorDB 
*"The Cognitive Latent Space"*

Designed for Autonomous Agents. Enables the **3-Tier Memory Manager** (VRAM -> RAM -> SSD), **Langevin Dynamics** for generative vector exploration, and **Self-Organized Criticality (SOC)** to passively consolidate redundant memory.

**Best for**: Long-running Agents, dynamic memory systems, associative reasoning.

```python
import torch
from m2m import AdvancedVectorDB

# Initialize Full Cognitive Suite
agent_db = AdvancedVectorDB(device='cuda')
agent_db.add(torch.randn(50000, 640))

# 1. Standard Search
nearest = agent_db.search(torch.randn(1, 640), k=10)

# 2. Generative Latent Exploration
# Uses Underdamped Langevin Dynamics to explicitly walk the energy manifold
creative_samples = agent_db.generate(query=torch.randn(1, 640), n_steps=20)

# 3. Consolidate Memory via Self-Organized Criticality
# Automatically removes near-duplicate or useless splats based on access frequency
removed_count = agent_db.consolidate(threshold=0.85)
```

> **Note**: Both systems utilize the same underlying `SplatStore` and `HRM2Engine`. An index built and persisted in `SimpleVectorDB` can be loaded natively into `AdvancedVectorDB`, and vice-versa!

---

## 🔗 Integrations

M2M natively supports the industry-standard frameworks for building RAG applications and Agentic workflows.

### LangChain Integration

```python
from langchain.vectorstores import M2MVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = M2MVectorStore(
    embedding_function=embeddings.embed_query,
    splat_capacity=100000,
    enable_vulkan=True
)

vectorstore.add_texts(["Document 1", "Document 2"])
results = vectorstore.similarity_search("Query", k=5)
```

### LlamaIndex Integration

```python
from llamaindex import VectorStoreIndex, SimpleDirectoryReader
from m2m.integrations.llamaindex import M2MVectorStore

documents = SimpleDirectoryReader("./docs").load_data()

vectorstore = M2MVectorStore(latent_dim=640, max_splats=100000, enable_vulkan=True)
index = VectorStoreIndex.from_documents(documents, vector_store=vectorstore)

query_engine = index.as_query_engine()
response = query_engine.query("Your search query")
```

---

## 🏗 Architecture

![Architecture](./assets/chart_architecture.png)

### The 3-Tier Memory Hierarchy (Advanced Mode)

| Tier | Storage | Latency | Use Case |
|------|---------|---------|----------|
| **Hot** | VRAM | ~0.1ms | Active queries, highly recurrent context. |
| **Warm** | RAM | ~0.5ms | Cached HRM2 embeddings, mid-term context. |
| **Cold** | SSD | ~10ms | Long-term persisted cold storage. |

### Component Breakdown
- **M2MEngine**: The main router and orchestrator.
- **SplatStore**: `splats.py` handles the physical tensors and GPU tracking.
- **HRM2Engine**: `hrm2_engine.py` builds the two-level K-Means lookup tree.

---

## 📊 Benchmarks

![Benchmark Comparison](./assets/chart_benchmark_comparison.png)

### Test Configuration

| Parameter | Value |
|-----------|-------|
| **CPU** | Dual Core Local Edge Device |
| **RAM** | 2GB Available |
| **Vectors** | 10,000 (sklearn fallback) |
| **Dimensions**| 640D |

### Results

| System | Avg Latency | Throughput | Speedup |
|--------|-------------|------------|---------|
| **Linear Scan** | 55.66ms | 17.97 QPS | 1.0x (baseline) |
| **M2M CPU** | 212.10ms | 4.71 QPS | 0.3x |
| **M2M Vulkan** | **98.54ms** | **10.15 QPS** | **0.6x** |

*(Reproduce with `python benchmarks/run_benchmark.py --dataset sklearn --n-splats 10000 --n-queries 100 --k 10`)*

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+

### From Source

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -r requirements.txt
```

Verify your installation:
```bash
python scripts/validate_project.py
```

---

## 📄 License & References

Licensed under the **AGPLv3**.

- **Methodology Conclusions**: [METHODOLOGY_CONCLUSIONS.md](METHODOLOGY_CONCLUSIONS.md)
- **Config Guide**: [CONFIG_RAG.md](CONFIG_RAG.md)

---
*M2M: Machine-to-Memory*
