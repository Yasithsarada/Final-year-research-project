import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from app.core.config import settings
from app.models.resume import NormalizedSkill

logger = logging.getLogger(__name__)

# Lazy imports for optional scientific packages
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except (ImportError, MemoryError, Exception) as e:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(f"Could not load sentence-transformers: {str(e)}. Falling back to regex/Jaccard matching.")

class SkillNormalizer:
    """Normalizes extracted skill strings into a canonical taxonomy.
    
    Uses exact string matching and semantic embeddings for robust normalization.
    """

    def __init__(self):
        self.taxonomy: Dict[str, Dict[str, Any]] = {}
        self.alias_to_canonical: Dict[str, Tuple[str, str]] = {} # alias_lower -> (canonical, category)
        self.model: Optional[Any] = None
        self.taxonomy_embeddings: List[np.ndarray] = []
        self.taxonomy_keys: List[Tuple[str, str, str]] = [] # list of (alias/canonical, canonical, category)
        
        # Load taxonomy from JSON
        self._load_taxonomy()
        
        # Lazy initialize embeddings
        self._initialized_embeddings = False

    def _load_taxonomy(self):
        """Loads taxonomy data from JSON file."""
        try:
            if settings.TAXONOMY_PATH.exists():
                with open(settings.TAXONOMY_PATH, "r", encoding="utf-8") as f:
                    self.taxonomy = json.load(f)
            else:
                logger.warning(f"Taxonomy file not found at {settings.TAXONOMY_PATH}. Using empty taxonomy.")
                self.taxonomy = {}
                
            # Build fast lookup map for exact matching
            for key, val in self.taxonomy.items():
                canonical = val["canonical"]
                category = val["category"]
                aliases = val.get("aliases", [])
                
                # Register canonical name
                self.alias_to_canonical[canonical.lower()] = (canonical, category)
                
                # Register aliases
                for alias in aliases:
                    self.alias_to_canonical[alias.lower()] = (canonical, category)
                    
        except Exception as e:
            logger.error(f"Failed to load skill taxonomy: {str(e)}")
            self.taxonomy = {}

    def _initialize_embeddings(self):
        """Lazy loader for Sentence Transformer model and taxonomy embeddings."""
        if self._initialized_embeddings:
            return
            
        if not HAS_SENTENCE_TRANSFORMERS:
            self._initialized_embeddings = True
            return
            
        try:
            logger.info(f"Loading sentence-transformer model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            # Pre-compute embeddings for all taxonomy keys (canonical and aliases)
            texts_to_embed = []
            for key, val in self.taxonomy.items():
                canonical = val["canonical"]
                category = val["category"]
                
                # Add canonical
                texts_to_embed.append(canonical)
                self.taxonomy_keys.append((canonical, canonical, category))
                
                # Add aliases
                for alias in val.get("aliases", []):
                    if alias != canonical:
                        texts_to_embed.append(alias)
                        self.taxonomy_keys.append((alias, canonical, category))
            
            if texts_to_embed:
                logger.info(f"Embedding {len(texts_to_embed)} taxonomy entries...")
                # Compute embeddings in a batch
                embeddings = self.model.encode(texts_to_embed, show_progress_bar=False)
                self.taxonomy_embeddings = [np.array(emb) for emb in embeddings]
                
            self._initialized_embeddings = True
            logger.info("Semantic skill mapping embeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer embeddings: {str(e)}")
            self._initialized_embeddings = True # Set to true to avoid repeated failures

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Helper to calculate cosine similarity between two vectors."""
        dot_prod = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_prod / (norm_v1 * norm_v2))

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Token-based similarity fallback for exact matches and string distance."""
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def normalize_skill(self, raw_skill: str) -> NormalizedSkill:
        """Normalizes a single skill string to a canonical representation."""
        if not raw_skill:
            return NormalizedSkill(original="", canonical="Unknown", category="Unknown", similarity_score=0.0)
            
        clean_raw = raw_skill.strip()
        raw_lower = clean_raw.lower()
        
        # Tier 1: Exact Match Lookup (Case-insensitive)
        if raw_lower in self.alias_to_canonical:
            canonical, category = self.alias_to_canonical[raw_lower]
            return NormalizedSkill(
                original=clean_raw,
                canonical=canonical,
                category=category,
                similarity_score=1.0
            )
            
        # Tier 2: Semantic Matching using sentence-transformers
        # Initialize embeddings if not already done
        self._initialize_embeddings()
        
        if HAS_SENTENCE_TRANSFORMERS and self.model and len(self.taxonomy_embeddings) > 0:
            try:
                # Embed the single raw skill
                raw_emb = self.model.encode(clean_raw, show_progress_bar=False)
                
                best_idx = -1
                best_score = -1.0
                
                # Compute similarity with all taxonomy embeddings
                for idx, tax_emb in enumerate(self.taxonomy_embeddings):
                    score = self._cosine_similarity(raw_emb, tax_emb)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        
                if best_idx != -1 and best_score >= settings.SKILL_SIMILARITY_THRESHOLD:
                    _, canonical, category = self.taxonomy_keys[best_idx]
                    return NormalizedSkill(
                        original=clean_raw,
                        canonical=canonical,
                        category=category,
                        similarity_score=round(best_score, 3)
                    )
            except Exception as e:
                logger.error(f"Semantic match error for '{clean_raw}': {str(e)}")
                
        # Tier 3: Jaccard/Fuzzy Similarity Fallback
        best_fuzzy_key = None
        best_fuzzy_score = 0.0
        
        for tax_key, (canonical, category) in self.alias_to_canonical.items():
            score = self._jaccard_similarity(clean_raw, tax_key)
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_key = tax_key
                
        if best_fuzzy_key and best_fuzzy_score >= 0.5:
            canonical, category = self.alias_to_canonical[best_fuzzy_key]
            return NormalizedSkill(
                original=clean_raw,
                canonical=canonical,
                category=category,
                similarity_score=round(best_fuzzy_score, 3)
            )

        # Fallback: Skill is unknown but registered
        return NormalizedSkill(
            original=clean_raw,
            canonical=clean_raw, # keep original as canonical
            category="Other/General",
            similarity_score=0.0
        )

    def normalize_all(self, raw_skills: List[str]) -> List[NormalizedSkill]:
        """Utility to normalize list of skills, removing duplicates."""
        normalized = []
        seen_canonical = set()
        
        for skill in raw_skills:
            norm = self.normalize_skill(skill)
            if norm.canonical not in seen_canonical:
                normalized.append(norm)
                seen_canonical.add(norm.canonical)
                
        return normalized
