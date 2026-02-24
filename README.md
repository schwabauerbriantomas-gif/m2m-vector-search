# M2M Vector Search Engine

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![LangChain Integration](https://img.shields.io/badge/RAG-LangChain_Native-purple.svg)](#langchain-integration)
[![Performance](https://img.shields.io/badge/CPU_QPS-52.73-orange.svg)](#performance-benchmarks-real-data)

**M2M (Machine-to-Memory)** is a high-performance vector search engine designed for strictly local, resource-efficient enterprise AI applications. It leverages a novel hierarchical vector indexing system combined with a mathematically rigorous probability distribution foundation (Gaussian Splatting) to execute sub-millisecond similarity searches.

---

## System Architecture

M2M discards traditional single-layer flat vector arrays (like standard FAISS baseline) or monolithic RAM-heavy graphs (HNSW) in favor of a dynamic, multi-tier system explicitly built for memory constraint environments.

### Core Modules

```mermaid
graph TD;
    A[Application Layer: LangChain / gRPC] --> B(M2M Core Engine);
    B --> C[SplatStore: μ, α, κ];
    B --> D[HRM2 Clustering Engine];
    B --> E[SOC Consolidation Controller];
    C --> F[(3-Tier Memory Manager)];
    F --> G[VRAM: Hot Storage];
    F --> H[RAM: Warm Storage];
    F --> I[SSD: Cold Storage];
```

| Component | Responsible File | Architectural Role | Complexity |
|-----------|------------------|--------------------|------------|
| **SplatStore** | `splats.py` | Distributes embedding storage storing Gaussian Splats parameters $(\mu, \alpha, \kappa)$. Captures vector density and semantic shape. | $O(1)$ ins |
| **HRM2 Engine** | `hrm2_engine.py` | Recursive Hierarchical Region Merging. Segregates latent space semantically. | $O(\log N)$ q |
| **MemoryManager**| `memory.py` | 3-tier memory bus intercepting frequency hits to swap tensors between GPU ↔ RAM ↔ NVMe dynamically. | - |
| **SOC** | `splats.py` | Self-Organized Criticality controller. Unsupervised background merging of tightly clustered Splats to reduce total parameter count natively. | $O(N)$ bk |
| **Riemann Geom** | `geometry.py` | Mathematical boundaries for normalizing and calculating distance along spherical manifolds $S^{d-1}$ ensuring directional context over purely Euclidean $L_2$. | - |

### Computational Flow
1. **Hierarchical Region Merging (HRM2)**
   - M2M partitions the embedding space using coarse and fine quantization cells recursively, transforming standard $O(N)$ linear scans into **$O(\log N)$** logarithmic retrieval pathways.
2. **Fluid 3-Tier Memory Lifecycle**
   - **Hot (VRAM):** Highest query probability space.
   - **Warm (RAM):** Infrequent but active semantic boundaries.
   - **Cold (SSD/NVMe):** Archival storage with transparent swap mechanisms.
3. **Distribution-Aware Representations (Gaussian Splats)**
   - Vectors are stored not as point estimates, but as distributions defined by parameters $(\mu, \alpha, \kappa)$, capturing the semantic shape and certainty bounds of the ingested data.

---

## Performance Benchmarks (Real Data)

All benchmarks use the **scikit-learn Handwritten Digits dataset** (1,797 real images projected to 640D embedding space, upsampled to 10,000 samples). No synthetic or random data is used for reported metrics. Results are fully reproducible by running `python examples/validate_data_lake.py`.

### Hardware & Environment

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen (x86_64) |
| **GPU** | AMD Radeon RX 6650 XT |
| **OS** | Windows 10 |
| **Python** | 3.12 |
| **PyTorch** | CPU tensors (PyTorch does not support Vulkan as a device backend) |
| **Vulkan** | Custom GLSL compute shaders via PyVulkan (`vulkan_compute.py`) |

### Backend Architecture

> [!IMPORTANT]
> M2M supports two compute backends. **PyTorch tensors always reside on CPU** (or CUDA if available). When `device='vulkan'` is set, GPU-accelerated distance calculations run through custom Vulkan compute shaders (`moe.comp`), not through PyTorch.

| Setting | Tensor Storage | GPU Compute | Use Case |
|---------|---------------|-------------|----------|
| `device='cpu'` | CPU RAM | None | Maximum compatibility |
| `device='cuda'` | GPU VRAM | PyTorch CUDA | NVIDIA GPUs |
| `device='vulkan'` | CPU RAM | **Vulkan GPU shaders** | Any Vulkan-capable GPU (AMD, Intel, NVIDIA) |

### Benchmark Results

> Dataset: **10,000 real-world embeddings** (sklearn digits → 640D), **1,000 queries**, $K = 10$.

| Metric | Linear Scan | M2M CPU | M2M Vulkan GPU |
|--------|-------------|---------|----------------|
| **Avg Latency** | 93.53 ms | **18.92 ms** | 23.22 ms |
| **Throughput (QPS)** | 10.70 | **48.97** | 43.07 |
| **Speedup vs Linear** | 1.0x | **4.6x** | 4.0x |
| **P95 Latency** | — | **24.10 ms** | 27.93 ms |
| **P99 Latency** | — | **30.44 ms** | 34.03 ms |
| **Ingest Rate** | — | 890 splats/s | **1,046 splats/s** |

![Benchmark Results](assets/benchmark_results.png)

### Data Lake Training Metrics

| Metric | CPU | Vulkan GPU |
|--------|-----|------------|
| **Standard Training** (SOC Importance Sampling) | **49,651 splats/s** | 48,784 splats/s |
| **Generative Training** (Langevin Augmentation) | **38,667 splats/s** | 38,534 splats/s |
| **Standard Loss** | 2.3029 | 2.3037 |
| **Generative Loss** | 2.3035 | 2.3027 |

![Data Lake Metrics](assets/benchmark_data_lake.png)

> [!NOTE]
> **Vulkan retrieval vs CPU:** The retrieval path uses Vulkan persistent GPU buffers for distance computation (via `VulkanMoERouter` with pre-allocated buffers — **0.82ms per GPU dispatch**). However, the coarse cluster probing and candidate collection still run on CPU, adding CPU→GPU→CPU round-trip latency per query. For ingest workloads, Vulkan provides a consistent **+18% throughput advantage**. Future work will batch multiple queries into a single GPU dispatch to amortize the round-trip cost.

### Reproducing Benchmarks

```bash
# Run both CPU and Vulkan benchmarks with real data
python examples/validate_data_lake.py

# CPU only
python examples/validate_data_lake.py --cpu

# Vulkan GPU only
python examples/validate_data_lake.py --vulkan

# Generate charts from results
python scripts/generate_charts.py

# Results saved to data_lake_real_metrics.json
```

---

## LangChain Integration

M2M natively supports integration into your existing Retrieval-Augmented Generation (RAG) pipelines as an official LangChain `VectorStore`. 

### Quick Start with LangChain

```python
from integrations.langchain import M2MVectorStore
from langchain_core.embeddings import FakeEmbeddings # Replace with HuggingFace, OpenAI, etc.
from m2m import M2MConfig

# 1. Provide M2M constraints
config = M2MConfig(device='cpu', latent_dim=128, max_splats=100000)

# 2. Initialize Vector Store Wrapper
vectorstore = M2MVectorStore(
    embeddings=FakeEmbeddings(size=128), 
    config=config
)

# 3. Add chunked documentation (handles normalization and UUID metadata transparently)
documents = ["M2M provides highly scalable memory architecture.", "Utilize LangChain natively."]
vectorstore.add_texts(documents)

# 4. Perform ultra-low latency semantic queries
results = vectorstore.similarity_search("How is memory architecture handled?", k=1)
for doc in results:
    print(doc.page_content)
```

---

## Usage Scenarios

M2M is built strictly for developers executing retrieval operations under operational constraints constraints:
- **Edge AI Devices:** Run complex vector inferences directly on limited machinery.
- **Privacy-First RAG Local Agents:** Provide external memory to LLMs completely air-gapped from cloud environments.
- **Dynamic Data Lakes:** Instantly ingest, organize and query changing real-time metrics through the SOC (Self-Organized Criticality) consolidation engine.

---

## Development Setup

The project enforces `pytest` suites and strict dataset validation metrics before merging.

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search
pip install -r requirements.txt

# Execute regression and integration proofs
python scripts/validate_project.py
python scripts/validate_real_datasets.py
python tests/test_langchain.py
```

### Reporting Issues and Contributing
M2M is undergoing continuous R&D. If you encounter PyTorch/Vulkan backend issues when swapping devices, or wish to contribute optimization kernels in PyTorch C++ or HLSL, please consult `CONTRIBUTING.md` and submit an issue with your hardware execution traces.
