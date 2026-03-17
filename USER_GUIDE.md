# M2M Vector Search - Complete User Guide
**Version:** 2.0.0 | **Date:** 2026-03-13

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/schwabauerbriantomas-gif/m2m-vector-search.git
cd m2m-vector-search

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage (30 seconds)

```python
from m2m import SimpleVectorDB
import numpy as np

# 1. Initialize database
db = SimpleVectorDB(latent_dim=768, mode='standard')

# 2. Add documents
vectors = np.random.randn(100, 768).astype(np.float32)
metadata = [{'category': 'tech', 'source': 'blog'} for _ in range(100)]
documents = [f'Document content {i}' for i in range(100)]

db.add(
    ids=[f'doc{i}' for i in range(100)],
    vectors=vectors,
    metadata=metadata,
    documents=documents
)

# 3. Search
query = np.random.randn(768).astype(np.float32)
results = db.search(query, k=10, include_metadata=True)

for result in results:
    print(f"ID: {result.id}, Score: {result.score:.4f}")
    print(f"Metadata: {result.metadata}")
    print(f"Document: {result.document}\n")
```

---

## Operation Modes

### Edge Mode (Fastest)
```python
db = SimpleVectorDB(latent_dim=768, mode='edge')
```
- No persistence
- No WAL
- Minimal overhead
- Best for: Temporary storage, testing, edge devices

### Standard Mode (Recommended)
```python
db = SimpleVectorDB(
    latent_dim=768,
    mode='standard',
    storage_path='./data/m2m'
)
```
- Full CRUD operations
- WAL + SQLite metadata
- Document storage
- Best for: Production use, persistent storage

### EBM Mode (Advanced)
```python
db = AdvancedVectorDB(
    latent_dim=768,
    enable_soc=True,
    enable_energy_features=True
)
```
- All Standard features
- Energy-based features
- Self-Organized Criticality
- Exploration & knowledge gaps
- Best for: Research, adaptive systems, autonomous agents

---

## CRUD Operations

### Create (Add)
```python
# Single document
db.add(
    ids=['doc1'],
    vectors=vector[np.newaxis, :],
    metadata={'category': 'tech'},
    documents=['Document text']
)

# Batch insert
db.add(
    ids=['doc1', 'doc2', 'doc3'],
    vectors=vectors,  # Shape: [3, 768]
    metadata=[{'cat': 'a'}, {'cat': 'b'}, {'cat': 'c'}],
    documents=['Text 1', 'Text 2', 'Text 3']
)

# Auto-generated IDs
db.add(vectors=vectors)  # IDs auto-generated as UUIDs
```

### Read (Search)
```python
# Basic search
results = db.search(query, k=10)

# With metadata filter
results = db.search(
    query,
    k=10,
    filter={'category': {'$eq': 'tech'}},
    include_metadata=True
)

# Filter operators
filter = {
    'category': {'$eq': 'tech'},      # Equal
    'year': {'$gte': 2020},           # Greater than or equal
    'score': {'$lt': 100},            # Less than
    'status': {'$ne': 'deleted'}      # Not equal
}

# With energy information (EBM mode)
result = db.search_with_energy(query, k=10)
print(f"Query energy: {result.query_energy}")
print(f"Confidence: {result.total_confidence}")
```

### Update
```python
# Update metadata
db.update('doc1', metadata={'category': 'updated'})

# Update vector
db.update('doc1', vector=new_vector)

# Update document text
db.update('doc1', document='Updated text')

# Upsert (create if doesn't exist)
db.update('new_doc', vector=vector, upsert=True)
```

### Delete
```python
# Soft delete (recommended)
db.delete(id='doc1')

# Hard delete (permanent)
db.delete(id='doc1', hard=True)

# Delete by filter
db.delete(filter={'category': {'$eq': 'deprecated'}}, hard=True)

# Batch delete
db.delete(ids=['doc1', 'doc2', 'doc3'])
```

---

## Advanced Features (EBM Mode)

### Energy Analysis
```python
from m2m import AdvancedVectorDB

db = AdvancedVectorDB(latent_dim=768, enable_energy_features=True)

# Get vector energy
energy = db.get_energy(vector)
print(f"Energy: {energy}")

# Search with energy
result = db.search_with_energy(query, k=10)
print(f"Query energy: {result.query_energy}")
print(f"Uncertainty regions: {len(result.uncertainty_regions)}")
```

### Knowledge Gaps
```python
# Find gaps in knowledge
gaps = db.find_knowledge_gaps(n=5)

for gap in gaps:
    print(f"Gap at: {gap.center}")
    print(f"Energy: {gap.energy}")
    print(f"Radius: {gap.radius}")
```

### Exploration Suggestions
```python
# Get exploration suggestions
suggestions = db.suggest_exploration(n=3)

for suggestion in suggestions:
    print(f"Topic: {suggestion.topic}")
    print(f"Reason: {suggestion.reason}")
    print(f"Priority: {suggestion.priority}")
```

### Self-Organized Criticality
```python
# Check system state
criticality = db.check_criticality()
print(f"State: {criticality.state}")  # SUBCRITICAL/CRITICAL/SUPERCRITICAL

# Trigger avalanche (reorganization)
avalanche = db.trigger_avalanche()
print(f"Avalanche size: {avalanche.size}")

# Relax system
relax_result = db.relax(iterations=10)
print(f"Energy delta: {relax_result.energy_delta}")
```

---

## Persistence

### Save/Load
```python
# Save database
db.save('./my_database')

# Load database
db = SimpleVectorDB(latent_dim=768, mode='standard')
db.load('./my_database')
```

### WAL (Write-Ahead Log)
```python
# WAL is automatic in standard/ebm modes
db = SimpleVectorDB(
    latent_dim=768,
    mode='standard',
    storage_path='./data',
    enable_wal=True
)

# All operations are logged
db.add(...)  # Logged to WAL
db.update(...)  # Logged to WAL
db.delete(...)  # Logged to WAL

# Manual checkpoint
db.storage.checkpoint()
```

---

## Performance Tuning

### LSH Configuration
```python
from m2m import SimpleVectorDB
from m2m.lsh_index import LSHConfig

# Custom LSH config
db = SimpleVectorDB(
    latent_dim=768,
    enable_lsh_fallback=True,
    lsh_threshold=0.15  # Silhouette threshold
)

# LSH activates automatically for uniform distributions
# Default: 15 tables, 18 bits, 50 probes
```

### Memory Management
```python
# 3-tier memory (AdvancedVectorDB only)
db = AdvancedVectorDB(
    latent_dim=768,
    enable_3_tier_memory=True  # VRAM/RAM/SSD
)

# Manual consolidation
db.consolidate(threshold=0.5)
```

### Batch Operations
```python
# Batch insert (faster than individual)
vectors = np.random.randn(10000, 768).astype(np.float32)
ids = [f'doc{i}' for i in range(10000)]

db.add(ids=ids, vectors=vectors)  # Single batch call
```

---

## Integration Examples

### LangChain
```python
# Check integrations folder
# File: integrations/langchain.py

from m2m.integrations.langchain import M2MVectorStore
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = M2MVectorStore(
    embedding=embeddings,
    latent_dim=1536
)

# Use with LangChain
from langchain.chain import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
```

### Custom Embeddings
```python
from sentence_transformers import SentenceTransformer
from m2m import SimpleVectorDB

model = SentenceTransformer('all-MiniLM-L6-v2')
db = SimpleVectorDB(latent_dim=384, mode='standard')

# Encode and store
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = model.encode(texts)

db.add(
    ids=[f'doc{i}' for i in range(len(texts))],
    vectors=embeddings,
    documents=texts
)

# Search
query_embedding = model.encode(["search query"])
results = db.search(query_embedding[0], k=5)
```

---

## REST API (Server Mode)

### Start Server
```bash
# Using uvicorn
cd m2m-vector-search
uvicorn m2m.api.coordinator_api:app --host 0.0.0.0 --port 8000
```

### Client Usage
```python
from m2m import M2MClient

# Connect to server
client = M2MClient(host='localhost', port=8000)

# Create collection
collection = client.create_collection(
    name='documents',
    dimension=768
)

# Add vectors
collection.add(
    vectors=vectors,
    ids=['doc1', 'doc2'],
    metadata=[{'cat': 'a'}, {'cat': 'b'}]
)

# Search
results = collection.search(vector=query, k=10)

# Energy API (EBM mode)
energy_map = collection.get_energy_map(center=query, radius=1.0)
gaps = collection.find_knowledge_gaps()
```

---

## Troubleshooting

### Common Issues

**1. "No module named 'm2m'"**
```bash
# Ensure you're in the project directory
cd m2m-vector-search
pip install -e .
```

**2. "LSH fallback activated"**
```
This is normal - M2M automatically uses LSH for uniform distributions.
Performance remains good (11-12ms/query).
```

**3. "Vulkan acceleration disabled"**
```
Vulkan requires:
- Vulkan SDK installed
- Compatible GPU
- vulkan Python package

CPU mode works fine for most use cases.
```

**4. Memory issues with large datasets**
```python
# Use batch operations
db.add(ids=ids_batch, vectors=vectors_batch)

# Enable 3-tier memory
db = AdvancedVectorDB(latent_dim=768, enable_3_tier_memory=True)

# Periodic consolidation
db.consolidate()
```

---

## Best Practices

### 1. Choose the Right Mode
- **Edge**: Testing, development, edge devices
- **Standard**: Production, persistence needed
- **EBM**: Research, adaptive systems

### 2. Batch Operations
```python
# Good: Batch insert
db.add(ids=all_ids, vectors=all_vectors)

# Bad: Individual inserts
for id, vec in zip(ids, vectors):
    db.add(ids=[id], vectors=vec[np.newaxis, :])
```

### 3. Use Metadata Wisely
```python
# Good: Structured metadata
metadata = {
    'category': 'tech',
    'year': 2024,
    'source': 'blog',
    'author': 'john'
}

# Bad: Unstructured metadata
metadata = {'data': 'category:tech,year:2024'}
```

### 4. Filter Early
```python
# Good: Filter in search
results = db.search(query, k=10, filter={'category': 'tech'})

# Bad: Filter after search
results = db.search(query, k=100)
filtered = [r for r in results if r.metadata.get('category') == 'tech']
```

### 5. Monitor Performance
```python
stats = db.get_stats()
print(f"Total docs: {stats['total_documents']}")
print(f"Active docs: {stats['active_documents']}")
print(f"LSH active: {stats['lsh_active']}")
```

---

## Performance Benchmarks

### Hardware
- CPU: AMD Ryzen 5 3400G
- RAM: 32GB
- Mode: Edge (CPU-only)

### Results
```
Dataset: 1,000 vectors (128D)
Ingestion: 0.30s (3,322 vectors/sec)
Search (k=10): 11.6ms/query (86 queries/sec)
Memory: 0.13MB for 1,000 vectors
```

### Scaling
```
10K vectors: ~3s ingestion, 12ms/query
100K vectors: ~30s ingestion, 15ms/query
1M vectors: ~5min ingestion, 20ms/query (estimated)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│           SimpleVectorDB                 │
│  ┌────────────────────────────────────┐ │
│  │      M2MEngine                      │ │
│  │  ┌──────────────┬────────────────┐ │ │
│  │  │ SplatStore   │  HRM2Engine    │ │ │
│  │  │  (μ, α, κ)   │  (coarse/fine) │ │ │
│  │  └──────────────┴────────────────┘ │ │
│  │  ┌────────────────────────────────┐│ │
│  │  │ EnergyFunction                 ││ │
│  │  │  E_splats + E_geom + E_comp    ││ │
│  │  └────────────────────────────────┘│ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ LSH Index (fallback)               │ │
│  │  CrossPolytope (15 tables)         │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘

AdvancedVectorDB adds:
┌─────────────────────────────────────────┐
│  ┌────────────────────────────────────┐ │
│  │ EBM Layer                          │ │
│  │  - EBMEnergy                       │ │
│  │  - EBMExploration                  │ │
│  │  - SOCEngine                       │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │ 3-Tier Memory                      │ │
│  │  - Hot (VRAM)                      │ │
│  │  - Warm (RAM)                      │ │
│  │  - Cold (SSD)                      │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## API Reference

### SimpleVectorDB

**`__init__(latent_dim, mode='standard', storage_path=None, enable_wal=True)`**

**`add(ids, vectors, metadata=None, documents=None)`**
- Returns: int (number added)

**`search(query, k=10, filter=None, include_metadata=False, include_energy=False)`**
- Returns: List[DocResult] or Tuple

**`update(id, vector=None, metadata=None, document=None, upsert=False)`**
- Returns: UpdateResult

**`delete(id=None, ids=None, filter=None, hard=False)`**
- Returns: DeleteResult

**`save(path)`**

**`load(path)`**

**`get_stats()`**
- Returns: Dict

### AdvancedVectorDB (extends SimpleVectorDB)

**`search_with_energy(query, k=10)`**
- Returns: SearchResult

**`get_energy(vector)`**
- Returns: float

**`find_knowledge_gaps(n=5)`**
- Returns: List[EnergyRegion]

**`suggest_exploration(n=3)`**
- Returns: List[ExplorationSuggestion]

**`check_criticality()`**
- Returns: CriticalityReport

**`trigger_avalanche(seed_point=None)`**
- Returns: AvalancheResult

**`relax(iterations=10)`**
- Returns: RelaxationResult

**`consolidate(threshold=None)`**
- Returns: int (number removed)

---

## Conclusion

M2M Vector Search is a **production-ready, high-performance vector database** with unique EBM features. It offers:

✅ Full CRUD operations with metadata
✅ Automatic LSH fallback for uniform distributions
✅ Energy-Based Model for advanced use cases
✅ Self-Organized Criticality for adaptive systems
✅ REST API for server deployment
✅ LangChain/LlamaIndex integrations
✅ Excellent performance (3K+ vectors/sec, 11ms/query)

**Recommendation:** Use **Standard mode** for production, **EBM mode** for research/adaptive systems.

---

**Created by:** Alfred 🎩
**Date:** 2026-03-13
**Repository:** https://github.com/schwabauerbriantomas-gif/m2m-vector-search
