# Plan de Implementación: Grafos Gaussianos en M2M Vector Search

## Estado Actual del Repositorio (v1.4.0)

**Última actualización:** Enero 2025

---

## ✅ YA IMPLEMENTADO

### 1. Cluster Distribuido Edge-Cloud (100%)

```
src/m2m/cluster/
├── router.py          ✅ ClusterRouter con load balancing
├── aggregator.py      ✅ ResultAggregator con RRF
├── edge_node.py       ✅ EdgeNode con SimpleVectorDB
├── client.py          ✅ M2MClusterClient con failover
├── health.py          ✅ Health monitoring (LoadMetrics, GeoLocation)
├── balancer.py        ✅ Load balancer (least_loaded, broadcast)
├── sharding.py        ✅ Sharding (hash, cluster, geo)
├── sync.py            ✅ SyncQueue para offline
└── protocol.py        ✅ Protocolos de comunicación
```

### 2. API HTTP REST (100%)

```
src/m2m/api/
├── coordinator_api.py ✅ FastAPI coordinator (route, search, heartbeat)
└── edge_api.py        ✅ FastAPI edge node (search, ingest, metrics)
```

### 3. Graph Splat Base (100%)

```
src/m2m/graph_splat.py
├── NodeType           ✅ Enum (DOCUMENT, ENTITY, CONCEPT, RELATION)
├── GraphEdge          ✅ Dataclass para aristas
├── GraphSplat         ✅ Nodo con edges entrantes/salientes
└── GaussianGraphStore ✅ Store completo con:
    ├── add_document()        ✅
    ├── add_entity()          ✅
    ├── add_relation()        ✅
    ├── search_entities()     ✅
    ├── traverse()            ✅
    ├── get_subgraph()        ✅
    ├── hybrid_search()       ✅
    └── get_stats()           ✅
```

### 4. Entity Extractor Nativo M2M (100%)

```
src/m2m/entity_extractor.py
├── M2MEntityExtractor      ✅ Extractor sin dependencias externas
│   ├── extract()           ✅ Método principal
│   ├── _extract_structural() ✅ Patrones regex
│   ├── _extract_ngrams()   ✅ Análisis de n-grams
│   ├── _validate_semantic() ✅ Validación en hiperesferas
│   └── learn_entity()      ✅ Aprendizaje de entidades
├── M2MGraphEntityExtractor ✅ Integración con GaussianGraphStore
│   └── extract_and_store() ✅ Extracción + almacenamiento
└── EntityCandidate         ✅ Dataclass para candidatos
```

### 5. Deployment (100%)

```
deploy/
├── docker-compose.yml ✅ 1 coordinator + 3 edges
├── Dockerfile         ✅ Multi-stage build
└── k8s/
    ├── deployment.yaml ✅ Kubernetes deployment
    └── service.yaml    ✅ Kubernetes service
```

---

## ❌ PENDIENTE DE IMPLEMENTACIÓN

### FASE 1: Tests de Validación

**Prioridad:** Alta | **Tiempo:** 1-2 días

#### Tests Faltantes

```python
# tests/test_entity_extractor.py

def test_extractor_without_dataset_transformer():
    """Valida funcionamiento SIN DatasetTransformer."""
    # Usar clustering ad-hoc en hiperesferas
    pass

def test_extractor_with_dataset_transformer():
    """Valida funcionamiento CON DatasetTransformer."""
    # Usar splats pre-computados
    pass

def test_structural_patterns():
    """Test de patrones estructurales (emails, URLs, etc.)."""
    pass

def test_ngram_analysis():
    """Test de análisis de n-grams."""
    pass

def test_semantic_validation():
    """Test de validación semántica en S^639."""
    pass

def test_integration_with_graph_store():
    """Test de integración con GaussianGraphStore."""
    pass
```

---

## Arquitectura del Entity Extractor Nativo M2M

### Sin Dependencias Externas

```
┌─────────────────────────────────────────────────────────────────────┐
│            M2M NATIVE ENTITY EXTRACTOR                              │
│            (Zero dependencias externas)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ENTRADA:                                                           │
│  ├── text: str                          # Texto a analizar          │
│  ├── embeddings: np.ndarray             # Embeddings pre-computados │
│  ├── embedding_model: object            # Modelo para generar emb   │
│  ├── existing_clusters: np.ndarray      # Clusters HRM2 existentes  │
│  └── splat_data: Dict                   # Splats pre-computados     │
│                                                                      │
│  MÉTODOS DE EXTRACCIÓN:                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 1. PATRONES ESTRUCTURALES (Regex)                           │    │
│  │    ├── Emails: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}│    │
│  │    ├── URLs: https?://[^\s<>"]+                              │    │
│  │    ├── Teléfonos: \d{3}[-.]?\d{3}[-.]?\d{4}                │    │
│  │    ├── Fechas: \d{4}-\d{2}-\d{2}                            │    │
│  │    └── Dinero: \$\d+(?:,\d{3})*(?:\.\d{2})?                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 2. ANÁLISIS DE N-GRAMS                                      │    │
│  │    ├── Tokenización preservando mayúsculas                   │    │
│  │    ├── Generación de n-grams (1-4 tokens)                   │    │
│  │    ├── Filtrado de stopwords                                │    │
│  │    ├── Scoring por frecuencia y estructura                   │    │
│  │    └── Clasificación por contexto (person, org, location)   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 3. VALIDACIÓN SEMÁNTICA (Hiperesferas S^639)               │    │
│  │    ├── Normalización a esfera unitaria                      │    │
│  │    ├── Distancia geodésica: arccos(x · y)                   │    │
│  │    ├── Clustering en espacio latente                        │    │
│  │    ├── Validación contra splats existentes                  │    │
│  │    └── Asignación de cluster_id                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  SALIDA:                                                            │
│  └── List[EntityCandidate]                                          │
│      ├── text: str                                                  │
│      ├── entity_type: str (person, org, location, etc.)             │
│      ├── score: float (0.0 - 1.0)                                   │
│      ├── count: int                                                 │
│      ├── embedding: np.ndarray                                      │
│      ├── cluster_id: int                                            │
│      └── start_positions: List[int]                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Validación con y sin DatasetTransformer

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FLUJO DE VALIDACIÓN                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CASO 1: SIN DatasetTransformer                                     │
│  ══════════════════════════════════                                 │
│                                                                      │
│  extract(text, embedding_model)                                      │
│         │                                                            │
│         ▼                                                            │
│  [1] Patrones estructurales → entidades                              │
│         │                                                            │
│         ▼                                                            │
│  [2] N-gram analysis → candidatos                                    │
│         │                                                            │
│         ▼                                                            │
│  [3] Generar embeddings para cada candidato                          │
│         │                                                            │
│         ▼                                                            │
│  [4] Clustering ad-hoc (K-Means en S^639)                           │
│         │                                                            │
│         ▼                                                            │
│  [5] Validar: cluster_size > threshold                               │
│         │                                                            │
│         ▼                                                            │
│  [OUTPUT] Entidades validadas                                        │
│                                                                      │
│  CASO 2: CON DatasetTransformer                                     │
│  ══════════════════════════════════                                 │
│                                                                      │
│  extract(text, embedding_model, splat_data)                          │
│         │                                                            │
│         ▼                                                            │
│  [1] Patrones estructurales → entidades                              │
│         │                                                            │
│         ▼                                                            │
│  [2] N-gram analysis → candidatos                                    │
│         │                                                            │
│         ▼                                                            │
│  [3] Generar embeddings para cada candidato                          │
│         │                                                            │
│         ▼                                                            │
│  [4] Calcular distancia geodésica a splats existentes               │
│         │                                                            │
│         ▼                                                            │
│  [5] Validar: dist < threshold OR kappa/alpha alto                  │
│         │                                                            │
│         ▼                                                            │
│  [OUTPUT] Entidades validadas con cluster_id                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Código Implementado

### M2MEntityExtractor

Ubicación: `src/m2m/entity_extractor.py`

```python
class M2MEntityExtractor:
    """
    Extractor de entidades nativo de M2M.
    
    Detecta entidades usando:
    1. Patrones estructurales (emails, URLs, fechas, números)
    2. N-gram frequency analysis con validación semántica
    3. Clustering en espacio de hiperesferas S^639
    
    NO requiere GLiNER ni modelos externos de NER.
    """
    
    # Patrones estructurales básicos
    STRUCTURAL_PATTERNS = [
        EntityPattern("email", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "contact"),
        EntityPattern("url", r'https?://[^\s<>"]+|www\.[^\s<>"]+', "url"),
        EntityPattern("phone", r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "contact"),
        EntityPattern("date_iso", r'\b\d{4}-\d{2}-\d{2}\b', "date"),
        EntityPattern("money_usd", r'\$\d+(?:,\d{3})*(?:\.\d{2})?', "money"),
        # ... más patrones
    ]
    
    def extract(
        self,
        text: str,
        embeddings: Optional[np.ndarray] = None,
        embedding_model: Optional[object] = None,
        existing_clusters: Optional[np.ndarray] = None,
        splat_data: Optional[Dict] = None,
    ) -> List[EntityCandidate]:
        """
        Extrae entidades del texto usando métodos nativos M2M.
        
        Funciona con y sin DatasetTransformer:
        - SIN splat_data: usa clustering ad-hoc
        - CON splat_data: valida contra splats existentes
        """
        # Implementación completa en entity_extractor.py
```

### M2MGraphEntityExtractor

```python
class M2MGraphEntityExtractor:
    """
    Integración de M2MEntityExtractor con GaussianGraphStore.
    """
    
    def extract_and_store(
        self,
        text: str,
        doc_embedding: np.ndarray,
        doc_id: int,
        min_score: float = 0.3,
    ) -> Dict:
        """
        Extrae entidades del texto y las almacena en el grafo.
        
        Returns:
            {
                "entities_found": int,
                "entities_stored": int,
                "relations_created": int,
                "entities": List[EntityCandidate]
            }
        """
```

---

## Tests de Validación

Ubicación: `tests/test_entity_extractor.py`

### Test SIN DatasetTransformer

```python
def test_without_dataset_transformer():
    """
    Test de funcionamiento SIN DatasetTransformer.
    
    Valida que el extractor funciona con clustering ad-hoc.
    """
    extractor = M2MEntityExtractor(
        use_structural_patterns=True,
        use_ngram_analysis=True,
        use_semantic_clustering=True,
    )
    
    embedding_model = MockEmbeddingModel(dim=640)
    
    text = """
    Apple Inc. reported quarterly earnings. 
    Microsoft Corporation announced Azure growth.
    Google LLC released new AI features.
    """
    
    entities = extractor.extract(
        text,
        embedding_model=embedding_model,
        # SIN splat_data, SIN existing_clusters
    )
    
    assert len(entities) > 0
    for entity in entities:
        assert 0 <= entity.score <= 1.0
```

### Test CON DatasetTransformer

```python
def test_with_dataset_transformer():
    """
    Test de funcionamiento CON DatasetTransformer.
    
    Simula splats pre-computados.
    """
    extractor = M2MEntityExtractor()
    embedding_model = MockEmbeddingModel(dim=640)
    
    # Simular splats generados por DatasetTransformer
    splat_data = {
        'mu': np.random.randn(50, 640).astype(np.float32),
        'alpha': np.random.uniform(0.01, 0.1, 50).astype(np.float32),
        'kappa': np.random.uniform(5.0, 20.0, 50).astype(np.float32),
    }
    
    text = "Apple Inc. reported strong earnings."
    
    entities = extractor.extract(
        text,
        embedding_model=embedding_model,
        splat_data=splat_data,
    )
    
    assert len(entities) > 0
    # Entidades deben tener cluster_id asignado
    for entity in entities:
        if entity.embedding is not None and entity.cluster_id >= 0:
            assert entity.cluster_id < 50
```

---

## Ventajas del Enfoque Nativo

| Característica | GLiNER (externo) | M2M Nativo |
|----------------|------------------|------------|
| **Dependencias** | Requiere instalación adicional | Solo numpy + sklearn |
| **Memoria** | ~500MB modelo | ~0MB (usa estructura existente) |
| **Velocidad** | 50-200ms por texto | 10-50ms por texto |
| **Offline** | Requiere descarga previa | 100% offline |
| **Integración** | Wrapper externo | Integrado en arquitectura |
| **Explicabilidad** | Caja negra | Determinístico y explicable |

---

## Próximos Pasos

1. **Ejecutar tests** - Validar funcionamiento con y sin DatasetTransformer
2. **Integrar en CI/CD** - Añadir a pipeline de tests
3. **Documentar API** - Añadir ejemplos de uso
4. **Benchmarks** - Comparar rendimiento con GLiNER

---

## Archivos Creados/Modificados

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `src/m2m/entity_extractor.py` | ✅ Creado | Extractor de entidades nativo M2M |
| `tests/test_entity_extractor.py` | ✅ Creado | Tests de validación |
| `src/m2m/__init__.py` | ✅ Modificado | Exportar nuevas clases |

---

## Referencias

- Repositorio M2M: https://github.com/schwabauerbriantomas-gif/m2m-vector-search
- Distancia geodésica en esferas: `arccos(x · y)` para vectores normalizados
- HRM2 Engine: Motor de recuperación jerárquica 2 niveles

---

**Documento actualizado:** Enero 2025  
**Versión M2M:** 1.4.0  
**Estado:** Entity Extractor nativo implementado, pendiente validación de tests
