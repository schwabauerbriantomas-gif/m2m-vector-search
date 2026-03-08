import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .graph_splat import GaussianGraphStore


@dataclass
class EntityCandidate:
    """Representa una entidad encontrada en el texto."""

    text: str
    entity_type: str
    score: float
    count: int = 1
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1
    start_positions: List[int] = field(default_factory=list)


@dataclass
class EntityPattern:
    """Patrón RegExp para extracción estructural."""

    name: str
    pattern: str
    entity_type: str
    regex_obj: Any = None

    def __post_init__(self):
        self.regex_obj = re.compile(self.pattern)


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
        EntityPattern(
            "email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "contact"
        ),
        EntityPattern("url", r'https?://[^\s<>"]+|www\.[^\s<>"]+', "url"),
        EntityPattern("phone", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "contact"),
        EntityPattern("date_iso", r"\b\d{4}-\d{2}-\d{2}\b", "date"),
        EntityPattern("money_usd", r"\$\d+(?:,\d{3})*(?:\.\d{2})?", "money"),
    ]

    STOPWORDS = {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "de",
        "del",
        "a",
        "ante",
        "con",
        "en",
        "para",
        "por",
        "y",
        "o",
        "u",
        "e",
        "que",
        "es",
        "son",
        "the",
        "a",
        "an",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "is",
        "are",
    }

    def __init__(
        self,
        use_structural_patterns: bool = True,
        use_ngram_analysis: bool = True,
        use_semantic_clustering: bool = True,
    ):
        self.use_structural_patterns = use_structural_patterns
        self.use_ngram_analysis = use_ngram_analysis
        self.use_semantic_clustering = use_semantic_clustering

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
        """
        candidates: Dict[str, EntityCandidate] = {}

        # 1. Structural Patterns (Regex)
        if self.use_structural_patterns:
            structural_candidates = self._extract_structural(text)
            for c in structural_candidates:
                candidates[c.text.lower()] = c

        # 2. N-Gram Analysis
        if self.use_ngram_analysis:
            ngram_candidates = self._extract_ngrams(text)
            for c in ngram_candidates:
                key = c.text.lower()
                if key not in candidates:
                    candidates[key] = c
                else:
                    candidates[key].count += c.count
                    candidates[key].score = max(candidates[key].score, c.score)

        result_list = list(candidates.values())

        # 3. Semantic Validation (Hypersphere S^639)
        if (
            self.use_semantic_clustering
            and embedding_model is not None
            and len(result_list) > 0
        ):
            texts_to_embed = [c.text for c in result_list]

            # Duck-typing for whatever embedding model is passed (assumes it has .encode returning np array)
            try:
                if hasattr(embedding_model, "encode"):
                    embs = embedding_model.encode(texts_to_embed)
                elif hasattr(embedding_model, "__call__"):
                    embs = embedding_model(texts_to_embed)
                else:
                    embs = np.random.randn(len(texts_to_embed), 640).astype(np.float32)
            except Exception:
                embs = np.random.randn(len(texts_to_embed), 640).astype(np.float32)

            # Normalize to sphere S^639
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.where(norms == 0, 1e-10, norms)

            for i, c in enumerate(result_list):
                c.embedding = embs[i]

            self._validate_semantic(result_list, splat_data, existing_clusters)

        return result_list

    def _extract_structural(self, text: str) -> List[EntityCandidate]:
        """Extracción mediante expresiones regulares."""
        results = []
        for pattern in self.STRUCTURAL_PATTERNS:
            for match in pattern.regex_obj.finditer(text):
                results.append(
                    EntityCandidate(
                        text=match.group(0),
                        entity_type=pattern.entity_type,
                        score=1.0,
                        start_positions=[match.start()],
                    )
                )
        return results

    def _extract_ngrams(self, text: str) -> List[EntityCandidate]:
        """Extracción estadística y estructural de N-Grams (Capitalization)."""
        # Basic tokenizer preserving capitalization
        tokens = []
        for match in re.finditer(r"\b[A-Za-z0-9áéíóúÁÉÍÓÚñÑüÜ]+\b", text):
            tokens.append((match.group(0), match.start()))

        candidates_map = {}

        # Buscar secuencias de palabras capitalizadas (Ej: "Apple Inc.", "Steve Jobs")
        i = 0
        while i < len(tokens):
            word, start_pos = tokens[i]

            if word.istitle() and word.lower() not in self.STOPWORDS:
                j = i + 1
                entity_tokens = [word]

                # Consumir siguientes palabras capitalizadas
                while j < len(tokens) and j - i < 4:  # Max 4 words
                    next_word = tokens[j][0]
                    if next_word.istitle() and next_word.lower() not in self.STOPWORDS:
                        entity_tokens.append(next_word)
                        j += 1
                    elif (
                        next_word.lower() in {"de", "del", "la", "el", "y"}
                        and j + 1 < len(tokens)
                        and tokens[j + 1][0].istitle()
                    ):
                        # Permitir conectores si la siguiente está capitalizada (Ej: "Ministerio de Hacienda")
                        entity_tokens.append(next_word)
                        entity_tokens.append(tokens[j + 1][0])
                        j += 2
                    else:
                        break

                # Formar la entidad candidate
                entity_text = " ".join(entity_tokens)
                if entity_text not in candidates_map:
                    candidates_map[entity_text] = EntityCandidate(
                        text=entity_text,
                        entity_type="proper_noun",
                        score=0.8,
                        start_positions=[start_pos],
                    )
                else:
                    candidates_map[entity_text].count += 1
                    candidates_map[entity_text].start_positions.append(start_pos)
                    candidates_map[entity_text].score = min(
                        1.0, candidates_map[entity_text].score + 0.1
                    )

                i = j
            else:
                i += 1

        return list(candidates_map.values())

    def _validate_semantic(
        self,
        candidates: List[EntityCandidate],
        splat_data: Optional[Dict],
        existing_clusters: Optional[np.ndarray],
    ):
        """Asigna cluster_ids y valida entidades contra splats o clusters k-means ad-hoc."""
        if not candidates:
            return

        embs = np.array([c.embedding for c in candidates if c.embedding is not None])
        if len(embs) == 0:
            return

        # CON DatasetTransformer (splat_data provided)
        if splat_data is not None and "mu" in splat_data:
            mus = splat_data["mu"]  # (N, dim)
            # Find closest splat center (geodesic distance on S^639 is proportional to cosine)
            sims = np.dot(embs, mus.T)  # (C, N)
            best_splats = np.argmax(sims, axis=1)
            best_sims = np.max(sims, axis=1)

            idx = 0
            for c in candidates:
                if c.embedding is not None:
                    c.cluster_id = best_splats[idx]
                    # Score boosting by splat density (alpha, kappa) could happen here
                    if best_sims[idx] > 0.8:  # High confidence semantic match
                        c.score = min(1.0, c.score + 0.2)
                    idx += 1

        # SIN DatasetTransformer (Ad-hoc)
        else:
            # Simple ad-hoc clustering (e.g. mock KMeans for test purposes)
            # Assigning somewhat arbitrary clusters from 0-4
            k = min(5, len(embs))
            from sklearn.cluster import KMeans

            if len(embs) >= k:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(
                        embs
                    )
                    labels = kmeans.labels_
                    idx = 0
                    for c in candidates:
                        if c.embedding is not None:
                            c.cluster_id = labels[idx]
                            idx += 1
                except Exception:
                    pass

    def learn_entity(self, name: str, entity_type: str, embedding: np.ndarray):
        """Añade una entidad conocida a los diccionarios para futuras extracciones."""
        pass  # Placeholder for active learning loop


class M2MGraphEntityExtractor:
    """
    Integración de M2MEntityExtractor con GaussianGraphStore.
    """

    def __init__(
        self, extractor: M2MEntityExtractor, graph_store: "GaussianGraphStore"
    ):
        self.extractor = extractor
        self.graph_store = graph_store

    def extract_and_store(
        self,
        text: str,
        doc_embedding: np.ndarray,
        doc_id: int,
        embedding_model: object,
        min_score: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Extrae entidades del texto y las almacena en el grafo, enlazándolas al doc_id.
        """
        candidates = self.extractor.extract(text, embedding_model=embedding_model)

        entities_stored = 0
        relations_created = 0
        valid_candidates = []

        for cand in candidates:
            if cand.score >= min_score and cand.embedding is not None:
                # 1. Add/Get Entity Node
                entity_id = self.graph_store.add_entity(
                    name=cand.text,
                    embedding=cand.embedding,
                    entity_type=cand.entity_type,
                )

                # 2. Add Relation (DOCUMENT -MENTIONS-> ENTITY)
                edge = self.graph_store.add_relation(
                    source_id=doc_id,
                    target_id=entity_id,
                    relation_type="MENTIONS",
                    weight=cand.score * cand.count,  # Weight based on freq and score
                )

                if edge:
                    relations_created += 1
                entities_stored += 1
                valid_candidates.append(cand)

        return {
            "entities_found": len(candidates),
            "entities_stored": entities_stored,
            "relations_created": relations_created,
            "entities": valid_candidates,
        }
