# M2M Vector Search - Configuration Reference
**Version:** 2.0.0 | **Last Updated:** March 13, 2026

---

## Recommended Configurations

### Development (Edge Mode)
```python
from m2m import SimpleVectorDB
import numpy as np

# Fast iteration, no persistence
db = SimpleVectorDB(
    latent_dim=768,
    mode='edge'
)

# Usage
vectors = np.random.randn(100, 768).astype(np.float32)
db.add(ids=[f'doc{i}' for i in range(100)], vectors=vectors)
results = db.search(vectors[0], k=10)
```

**When to use:**
- Development and testing
- Temporary storage
- Fast prototyping
- Edge devices

---

### Production (Standard Mode)
```python
from m2m import SimpleVectorDB
import numpy as np

# Full CRUD + persistence
db = SimpleVectorDB(
    latent_dim=768,
    mode='standard',
    storage_path='./data/m2m',
    enable_wal=True,
    max_splats=1000000
)

# Usage with metadata
metadata = [{'category': 'tech', 'source': 'blog'} for _ in range(100)]
documents = [f'Document content {i}' for i in range(100)]

db.add(
    ids=[f'doc{i}' for i in range(100)],
    vectors=vectors,
    metadata=metadata,
    documents=documents
)

# Search with filters
results = db.search(
    query,
    k=10,
    filter={'category': {'$eq': 'tech'}},
    include_metadata=True
)

# CRUD operations
db.update('doc1', metadata={'updated': True})
db.delete(id='doc2')
db.save('./backup')
```

**When to use:**
- Production deployment
- Persistent storage
- Full CRUD operations
- Document retrieval

---

### Research (EBM Mode)
```python
from m2m import AdvancedVectorDB
import numpy as np

# EBM features + SOC
db = AdvancedVectorDB(
    latent_dim=768,
    enable_soc=True,
    enable_energy_features=True,
    enable_3_tier_memory=True,
    soc_threshold=0.8
)

# Energy-aware operations
result = db.search_with_energy(query, k=10)
print(f"Query energy: {result.query_energy}")
print(f"Uncertainty: {len(result.uncertainty_regions)}")

# Knowledge gaps
gaps = db.find_knowledge_gaps(n=5)
for gap in gaps:
    print(f"Gap: {gap.center}, Energy: {gap.energy}")

# Self-organization
criticality = db.check_criticality()
print(f"State: {criticality.state}")

if criticality.state == 'CRITICAL':
    relax_result = db.relax(iterations=10)
    print(f"Energy delta: {relax_result.energy_delta}")

# Exploration suggestions
suggestions = db.suggest_exploration(n=3)
for suggestion in suggestions:
    print(f"Suggestion: {suggestion.topic}")
```

**When to use:**
- Research projects
- Adaptive systems
- Autonomous agents
- Knowledge discovery

---

## DBpedia Benchmark Configuration

### Validated Configuration
```python
# Configuration used in professional benchmark
config = {
    # Dataset
    "dataset": "DBpedia 1M",
    "embedding_model": "OpenAI text-embedding-3-large",
    "original_dimension": 3072,
    "truncated_dimension": 640,
    
    # Database
    "mode": "standard",
    "latent_dim": 640,
    "max_splats": 100000,
    "knn_k": 64,
    
    # LSH (auto-activated for uniform distributions)
    "enable_lsh_fallback": True,
    "lsh_threshold": 0.15,
    "lsh_tables": 15,
    "lsh_bits": 18,
    
    # Performance
    "batch_size": 1000,
    "enable_vulkan": False,  # CPU mode
    
    # Storage
    "storage_path": None,  # In-memory for benchmark
    "enable_wal": False
}
```

### Results
```
Ingestion: 1,528 docs/sec
Search (k=10): 104.91 queries/sec
Latency (mean): 9.53ms
Latency (P95): 10.08ms
Latency (P99): 12.01ms
```

---

## Performance Tuning

### LSH Parameters
```python
# For uniform distributions (silhouette < 0.15)
from m2m.lsh_index import LSHConfig

lsh_config = LSHConfig(
    dim=640,
    n_tables=15,      # More tables = better recall, more memory
    n_bits=18,        # More bits = better precision, slower
    n_probes=50,      # More probes = better recall, slower
    n_candidates=500  # More candidates = better recall, more compute
)

# Trade-offs:
# High recall: ↑ tables, ↑ probes, ↑ candidates
# High speed: ↓ tables, ↓ probes, ↓ candidates
# Balanced: default values
```

### Memory Management
```python
# For large datasets (>100K vectors)
db = AdvancedVectorDB(
    latent_dim=768,
    enable_3_tier_memory=True,  # VRAM/RAM/SSD
    max_splats=1000000
)

# Manual consolidation
db.consolidate(threshold=0.5)  # Remove low-frequency splats
```

### Batch Operations
```python
# Efficient batch ingestion
batch_size = 10000
for i in range(0, len(all_vectors), batch_size):
    batch = all_vectors[i:i+batch_size]
    batch_ids = all_ids[i:i+batch_size]
    db.add(ids=batch_ids, vectors=batch)
```

---

## Integration Examples

### LangChain
```python
# Note: Requires langchain package
from m2m.integrations.langchain import M2MVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

# Initialize
embeddings = OpenAIEmbeddings()
vectorstore = M2MVectorStore(
    embedding=embeddings,
    latent_dim=1536
)

# Add documents
docs = [Document(page_content="text", metadata={"source": "web"})]
vectorstore.add_documents(docs)

# Search
results = vectorstore.similarity_search("query", k=10)
```

### REST API
```python
# Start server
# $ uvicorn m2m.api.coordinator_api:app --host 0.0.0.0 --port 8000

# Client usage
import requests

# Create collection
requests.post('http://localhost:8000/collections', json={
    'name': 'documents',
    'dimension': 768
})

# Add vectors
requests.post('http://localhost:8000/collections/documents/add', json={
    'ids': ['doc1', 'doc2'],
    'vectors': vectors.tolist(),
    'metadata': [{'cat': 'a'}, {'cat': 'b'}]
})

# Search
response = requests.post('http://localhost:8000/collections/documents/search', json={
    'query': query.tolist(),
    'k': 10
})
```

---

## Environment Variables

```bash
# Optional environment configuration
export M2M_DEVICE=cpu              # cpu | cuda
export M2M_MAX_SPLATS=1000000      # Maximum splats
export M2M_STORAGE_PATH=./data     # Default storage path
export M2M_ENABLE_VULKAN=false     # GPU acceleration
export M2M_LOG_LEVEL=INFO          # DEBUG | INFO | WARNING | ERROR
```

---

## Hardware Recommendations

### Minimum (Development)
```
CPU: 4 cores
RAM: 8GB
Storage: 10GB SSD
Dataset: <100K vectors
```

### Recommended (Production)
```
CPU: 8+ cores
RAM: 32GB
Storage: 100GB SSD
Dataset: 100K-1M vectors
```

### High Performance (Large Scale)
```
CPU: 16+ cores
RAM: 64GB+
Storage: 500GB+ NVMe SSD
GPU: Vulkan-compatible (optional)
Dataset: 1M+ vectors
```

---

## Troubleshooting

### Issue: "LSH fallback activated"
**Solution:** Normal behavior for uniform distributions. Performance remains excellent.

### Issue: "Vulkan acceleration disabled"
**Solution:** Install Vulkan SDK and compatible GPU, or use CPU mode (works fine).

### Issue: "Out of memory"
**Solution:**
1. Enable 3-tier memory
2. Reduce batch size
3. Call `db.consolidate()` periodically

### Issue: "Slow ingestion"
**Solution:**
1. Use batch operations
2. Increase batch size
3. Disable WAL during bulk load

---

**Configuration Reference by:** Alfred 🎩
**Date:** March 13, 2026
**Validated with:** DBpedia 1M dataset
