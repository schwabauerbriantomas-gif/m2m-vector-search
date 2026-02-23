# M2M Training Data Lake

**High-performance Gaussian Splat Storage & Training Data Lake for AI Systems**

 [![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
 [![License](https://img.shields.io/badge/license-Apache%202.0-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**M2M Training Data Lake** transforms the M2M Vector Search core into a high-performance active PyTorch `IterableDataset` capable of scaling dynamically across VRAM, RAM, and SSD.

Instead of statically serving embeddings like traditional vector databases, M2M leverages **Gaussian Splats** and **Langevin Dynamics** to stream massive amounts of embeddings directly into Deep Learning training loops, generating stochastic data augmentations on the fly. 

### 💡 Why M2M Data Lake?
- **Streaming Massive Data:** Load millions of vectors directly into standard PyTorch training scripts from cold storage (SSD) to VRAM without exploding memory.
- **Langevin Data Augmentation:** The data lake is alive; it uses energy surfaces to alter and sample vectors at training time. The model never sees the exact same embedding twice.
- **SOC Importance Sampling:** Train faster by letting the Self-Organized Criticality (SOC) controller feed only the most "concentrated" ($\kappa$) and salient embeddings, skipping redundant data.

---

## 🌟 Features

### Storage & Optimization
- **Tier-Aware Streaming:** Background prefetching threads seamlessly load data from Cold (SSD) $\rightarrow$ Warm (RAM) $\rightarrow$ Hot (VRAM).
- **Consolidation:** Automatic redundancy elimination via Self-Organized Criticality.
- **Gaussian Representations:** Stores mean ($\mu$), concentration ($\kappa$), and precision ($\alpha$) rather than simple points.

### PyTorch Integration
- **`M2MDataLake` Dataset:** Native PyTorch `IterableDataset` exported instantly via `m2m.export_to_dataloader()`.
- **Zero-Copy Generation:** Bypasses RAM bottlenecks where possible, feeding generated states directly via PyTorch tensors on CUDA.

---

## 🏗 Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                    PyTorch Training Loop                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐  │
│  │   Llama / MLP   │ │    Optimizer    │ │   Criterion  │  │
│  └─────────────────┘ └─────────────────┘ └──────────────┘  │
└───────────────────────────▲────────────────────────────────┘
                            │ (Streaming Tensors)
┌───────────────────────────┴────────────────────────────────┐
│               M2M Iterable Dataset (Data Lake)             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐  │
│  │  Tier-Aware     │ │ Langevin Generat│ │   SOC Import │  │
│  │  Prefetching    │ │ ive Augmentation│ │   Sampling   │  │
│  └─────────────────┘ └─────────────────┘ └──────────────┘  │
└───────────────────────────▲────────────────────────────────┘
                            │ (Tiered Storage)
┌───────────────────────────┴────────────────────────────────┐
│                   M2M Memory Engine                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐  │
│  │   Hot (VRAM)    │ │   Warm (RAM)    │ │  Cold (SSD)  │  │
│  └─────────────────┘ └─────────────────┘ └──────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmarks

*Validated with 10,000 real-world high-dimensional structured embeddings (digits projected to S^639):*

| Hardware & Mode | Standard Training (SOC) | Generative Training (Langevin) |
| :--- | :--- | :--- |
| **CPU (Throughput)** | ~49,368 splats/sec | ~34,993 splats/sec |
| **Vulkan GPU (Throughput)** | ~35,801 splats/sec | **~38,059 splats/sec** |

*Data Ingest Speed into memory tiers:* **~80,260 splats/sec**

*Note: Standard iteration is heavily memory bound, making CPU faster for pure iteration. Generative Langevin dynamics require high numerical compute, where Vulkan acceleration shines, bypassing CPU bottlenecks.*

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
git checkout feature/data-lake
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### M2M Data Lake in 10 lines of code

```python
import torch
import torch.nn as nn
from m2m import M2MConfig, create_m2m

# 1. Initialize M2M Data Lake
m2m = create_m2m(M2MConfig(device='cuda'))

# 2. Ingest your dataset (embeddings)
dataset_embeddings = torch.randn(100000, 640).cuda()
m2m.add_splats(dataset_embeddings)

# 3. Export to an Iterable PyTorch DataLoader
dataloader = m2m.export_to_dataloader(
    batch_size=256, 
    generate_samples=True,  # Enables Langevin generative augmentation
    importance_sampling=True # Sorts stream by SOC concentration
)

# 4. Standard PyTorch Training
model = nn.Linear(640, 10).cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(5):
    for batch in dataloader:
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
```

See the `/examples` directory for complete working examples, such as `examples/m2m_training_loop.py` and `examples/validate_data_lake.py`.

---

## 🤝 Contributing
We welcome contributions! Please see the issue tracker for feature proposals and bug reports. 

## 📄 License
Licensed under the Apache License, Version 2.0 (the "License"). You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
