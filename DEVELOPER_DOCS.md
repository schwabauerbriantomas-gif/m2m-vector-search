# M2M Vector Search - Developer Documentation
**Version:** 2.0.0 | **Architecture Overview**

---

## System Architecture

### Core Components

#### 1. SplatStore (`splats.py`)
**Purpose:** Gaussian Splat storage and retrieval

**Key Features:**
- Stores Gaussian Splats with (μ, α, κ) parameters
- In-memory numpy arrays for fast access
- Frequency tracking for hot/cold data
- Compaction for memory efficiency

**API:**
```python
class SplatStore:
    def __init__(self, config: M2MConfig)
    def add_splat(self, vector: np.ndarray) -> bool
    def find_neighbors(self, query, k, lod=2) -> Tuple[ndarray, ndarray, ndarray]
    def compact(self) -> None
    def entropy(self, x=None) -> float
```

**Implementation Details:**
- Pre-allocated arrays (max_splats)
- Active count tracking (n_active)
- LOD (Level of Detail) for progressive search

---

#### 2. HRM2Engine (`hrm2_engine.py`)
**Purpose:** Hierarchical Retrieval with 2-level clustering

**Key Features:**
- Coarse clustering (K-means)
- Fine clustering per coarse cluster
- O(log N) search complexity
- MiniBatchKMeans for performance

**API:**
```python
class HRM2Engine:
    def __init__(self, config: M2MConfig)
    def build_index(self, vectors: np.ndarray) -> None
    def search(self, query: np.ndarray, k: int) -> Tuple
```

**Algorithm:**
1. Cluster vectors into N_coarse clusters
2. Within each coarse cluster, create N_fine sub-clusters
3. Search: Find best coarse → Find best fine → Return top-k

---

#### 3. EnergyFunction (`energy.py`)
**Purpose:** Compute energy landscape E(x)

**Energy Formula:**
```
E(x) = w_splat * E_splats(x) 
     + w_geom * E_geom(x) 
     + w_comp * E_comp(x) 
     - T * H[q(x)]
```

**Components:**
- **E_splats:** Energy from Gaussian Splats (attraction)
- **E_geom:** Geometric energy (manifold constraints)
- **E_comp:** Compositional energy (semantic coherence)
- **H[q(x)]:** Entropy term (exploration bonus)

**API:**
```python
class EnergyFunction:
    def __init__(self, config: M2MConfig)
    def __call__(self, x: np.ndarray) -> np.ndarray
    def E_splats(self, x, splats) -> np.ndarray
    def E_geom(self, x) -> np.ndarray
    def E_comp(self, x) -> np.ndarray
```

---

#### 4. LSH Index (`lsh_index.py`)
**Purpose:** Approximate nearest neighbor for uniform distributions

**Algorithm:** Cross-Polytope LSH
- 15 hash tables
- 18 bits per hash
- 50 probes per query
- 500 candidates

**API:**
```python
class CrossPolytopeLSH:
    def __init__(self, config: LSHConfig)
    def index(self, vectors: np.ndarray) -> None
    def query(self, query: np.ndarray, k: int) -> Tuple[ndarray, ndarray]
```

**Activation:**
- Automatic when silhouette score < 0.15
- Indicates uniform distribution
- Provides fast approximate search

---

### EBM Layer

#### 5. EBMEnergy (`ebm/energy_api.py`)
**Purpose:** Energy-based model for uncertainty quantification

**Features:**
- Compute E(x) for any vector
- Track high-energy regions
- Calculate free energy F = E - T*S

**API:**
```python
class EBMEnergy:
    def energy(self, x: np.ndarray) -> float
    def free_energy(self) -> float
    def update_splats(self, mu, alpha, kappa) -> None
```

---

#### 6. EBMExploration (`ebm/exploration.py`)
**Purpose:** Guide exploration of knowledge space

**Features:**
- Find high-energy regions (uncertainty)
- Suggest exploration topics
- Boltzmann sampling

**API:**
```python
class EBMExploration:
    def find_high_energy_regions(self, topic_vector, n_regions) -> List[EnergyRegion]
    def find_knowledge_gaps(self, n_gaps) -> List[EnergyRegion]
    def suggest_exploration(self, current_knowledge, n_suggestions) -> List[ExplorationSuggestion]
    def boltzmann_sample(self, temperature) -> np.ndarray
```

---

#### 7. SOCEngine (`ebm/soc.py`)
**Purpose:** Self-Organized Criticality for adaptive reorganization

**States:**
- **SUBCRITICAL:** Stable, low energy
- **CRITICAL:** Transition point
- **SUPERCRITICAL:** High energy, needs relaxation

**API:**
```python
class SOCEngine:
    def check_criticality(self) -> CriticalityReport
    def trigger_avalanche(self, seed_point) -> AvalancheResult
    def relax(self, iterations) -> RelaxationResult
    def get_statistics(self) -> Dict
```

**Avalanche Dynamics:**
1. Trigger at high-energy point
2. Propagate to neighbors
3. Relax until stable
4. Return statistics

---

### Storage Layer

#### 8. M2MPersistence (`storage/persistence.py`)
**Purpose:** Persistent storage with WAL

**Features:**
- SQLite for metadata
- NumPy arrays for vectors
- Write-Ahead Log for durability
- Checkpoint/recovery

**API:**
```python
class M2MPersistence:
    def __init__(self, path, enable_wal=True)
    def save_vectors(self, vectors, ids) -> None
    def save_metadata(self, id, shard_idx, vector_idx, metadata, document) -> None
    def load_vectors(self) -> Tuple[ndarray, List[str]]
    def filter_by_metadata(self, filter_dict) -> List[str]
    def checkpoint(self) -> None
```

---

#### 9. WriteAheadLog (`storage/wal.py`)
**Purpose:** Durability for all operations

**Format:**
- msgpack binary format
- Sequential log
- Checksum validation

**API:**
```python
class WriteAheadLog:
    def append(self, operation: str, data: Dict) -> None
    def replay(self) -> List[Dict]
    def checkpoint(self) -> None
    def compact(self) -> None
```

---

### High-Level API

#### 10. SimpleVectorDB (`__init__.py`)
**Purpose:** User-friendly CRUD interface

**Modes:**
- **edge:** No persistence, minimal overhead
- **standard:** WAL + SQLite, full CRUD
- **ebm:** Standard + EBM features

**Internal Flow:**
```
add() → SplatStore.add_splat() → HRM2Engine.build_index()
                                      ↓
search() → HRM2Engine.search() or LSH.query()
                ↓
         Filter by metadata
                ↓
         Return DocResult objects
```

---

#### 11. AdvancedVectorDB (`__init__.py`)
**Purpose:** Full EBM feature set

**Additional Features:**
- Energy computation
- SOC monitoring
- Exploration suggestions
- Knowledge gap detection

**Internal Flow:**
```
search_with_energy() → SimpleVectorDB.search()
                              ↓
                    EBMEnergy.energy(query)
                              ↓
                    EBMExploration.find_high_energy_regions()
                              ↓
                    Return SearchResult with energy info
```

---

## Data Flow

### Insert Operation
```
User → SimpleVectorDB.add(ids, vectors, metadata)
  ↓
Check distribution (silhouette score)
  ↓
If uniform: Activate LSH
Else: Use HRM2
  ↓
SplatStore.add_splat(vectors)
  ↓
HRM2Engine.build_index() or LSH.index()
  ↓
M2MPersistence.save_vectors()
M2MPersistence.save_metadata()
  ↓
WriteAheadLog.append('add', data)
```

### Search Operation
```
User → SimpleVectorDB.search(query, k, filter)
  ↓
If LSH active:
  LSH.query(query, k)
Else:
  HRM2Engine.search(query, k)
  ↓
Filter results by metadata
  ↓
If include_metadata: Load from SQLite
  ↓
Return List[DocResult]
```

### EBM Search
```
User → AdvancedVectorDB.search_with_energy(query, k)
  ↓
SimpleVectorDB.search(query, k)
  ↓
EBMEnergy.energy(query) → query_energy
  ↓
For each result:
  EBMEnergy.energy(result.vector) → result.energy
  confidence = 1 / (1 + energy)
  ↓
EBMExploration.find_high_energy_regions(query)
  ↓
Return SearchResult(
  results, query_energy, 
  total_confidence, uncertainty_regions
)
```

---

## Performance Characteristics

### Time Complexity

| Operation | HRM2 | LSH | Notes |
|-----------|------|-----|-------|
| **Insert** | O(N log N) | O(N) | N = number of vectors |
| **Search** | O(log N) | O(1) | k nearest neighbors |
| **Update** | O(log N) | O(1) | Single vector |
| **Delete** | O(1) | O(1) | Mark as deleted |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| **SplatStore** | O(N * D) | N vectors, D dimensions |
| **HRM2 Index** | O(N) | Cluster assignments |
| **LSH Tables** | O(N * T) | T = number of tables (15) |
| **Metadata** | O(N * M) | M = avg metadata size |

### Memory Usage

```
Total = (N * D * 4 bytes)         # Vectors
      + (N * 3 * 4 bytes)         # Splats (μ, α, κ)
      + (N * T * 4 bytes)         # LSH hashes
      + (N * M bytes)             # Metadata

Example (10K vectors, 768D):
= 10K * 768 * 4 = 30MB (vectors)
+ 10K * 3 * 4 = 120KB (splats)
+ 10K * 15 * 4 = 600KB (LSH)
+ 10K * 100 = 1MB (metadata)
≈ 32MB total
```

---

## Configuration

### M2MConfig

```python
from m2m.config import M2MConfig

# Simple config (edge mode)
config = M2MConfig.simple(device='cpu')
config.latent_dim = 768
config.max_splats = 100000

# Advanced config (EBM mode)
config = M2MConfig.advanced(device='cpu')
config.enable_3_tier_memory = True
config.soc_threshold = 0.8
config.energy_splat_weight = 1.0
config.energy_geom_weight = 0.1
config.energy_comp_weight = 0.5

# GPU config (Vulkan)
config = M2MConfig(device='cuda', enable_vulkan=True)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 640 | Vector dimensionality |
| `max_splats` | 100000 | Maximum number of splats |
| `knn_k` | 64 | Default k for KNN |
| `enable_vulkan` | False | GPU acceleration |
| `enable_3_tier_memory` | False | VRAM/RAM/SSD tiers |
| `enable_lsh_fallback` | True | Auto LSH for uniform data |
| `lsh_threshold` | 0.15 | Silhouette threshold |
| `soc_threshold` | 0.8 | SOC activation threshold |
| `energy_splat_weight` | 1.0 | Splat energy weight |
| `energy_geom_weight` | 0.1 | Geometric energy weight |
| `energy_comp_weight` | 0.5 | Compositional energy weight |
| `temperature` | 1.0 | Boltzmann temperature |
| `langevin_steps` | 200 | Langevin dynamics steps |

---

## Extension Points

### Custom Energy Functions

```python
from m2m.energy import EnergyFunction

class CustomEnergy(EnergyFunction):
    def E_splats(self, x, splats):
        # Custom splat energy
        return super().E_splats(x, splats)
    
    def E_custom(self, x):
        # Additional energy term
        return my_custom_energy(x)
    
    def __call__(self, x):
        return (super().__call__(x) + 
                self.config.custom_weight * self.E_custom(x))
```

### Custom LSH

```python
from m2m.lsh_index import LSHConfig

# Custom LSH parameters
config = LSHConfig(
    dim=768,
    n_tables=20,      # More tables
    n_bits=24,        # More bits
    n_probes=100,     # More probes
    n_candidates=1000 # More candidates
)
```

### Custom Exploration

```python
from m2m.ebm.exploration import EBMExploration

class CustomExploration(EBMExploration):
    def suggest_exploration(self, current_knowledge, n_suggestions):
        # Custom exploration strategy
        suggestions = []
        # ... your logic
        return suggestions
```

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_splat_store.py

# Run with coverage
pytest tests/ --cov=m2m --cov-report=html
```

### Integration Tests

```bash
# Run integration test suite
python test_corrected.py
```

### Performance Benchmarks

```bash
# Run benchmarks
python benchmarks/benchmark_ingestion.py
python benchmarks/benchmark_search.py
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY setup.py .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "m2m.api.coordinator_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: m2m-vector-db
spec:
  replicas: 3
  selector:
    matchLabels:
      app: m2m
  template:
    spec:
      containers:
      - name: m2m
        image: m2m-vector-db:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
```

---

## Monitoring

### Health Check

```python
# Application-level health check
stats = db.get_stats()
assert stats['active_documents'] > 0
assert stats['lsh_active'] or stats['ebm_enabled']
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

insertions = Counter('m2m_insertions_total', 'Total insertions')
searches = Counter('m2m_searches_total', 'Total searches')
search_latency = Histogram('m2m_search_latency_seconds', 'Search latency')

@search_latency.time()
def search_with_metrics(query, k):
    searches.inc()
    return db.search(query, k)
```

---

## Contributing

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests

### Pull Request Process
1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit PR

### Code Review Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No breaking changes

---

**Created by:** Alfred 🎩
**Date:** 2026-03-13
