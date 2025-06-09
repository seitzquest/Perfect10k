import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple, Any
import pickle
import hashlib
from functools import lru_cache
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
import os

from core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing semantic embeddings."""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self.cache_dir = "/tmp/perfect10k_embeddings"
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding by text hash."""
        cache_file = os.path.join(self.cache_dir, f"{text_hash}.pkl")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")
        return None
    
    def _cache_embedding(self, text_hash: str, embedding: np.ndarray):
        """Cache embedding to disk."""
        cache_file = os.path.join(self.cache_dir, f"{text_hash}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text."""
        if not text or not text.strip():
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2
        
        # Create hash for caching
        text_normalized = text.lower().strip()
        text_hash = hashlib.md5(text_normalized.encode()).hexdigest()
        
        # Try to get from cache first
        cached_embedding = self._get_cached_embedding(text_hash)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        self._load_model()
        try:
            embedding = self.model.encode(text_normalized, convert_to_numpy=True)
            
            # Cache the result
            self._cache_embedding(text_hash, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
            return np.zeros(384)  # Return zero vector as fallback
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        self._load_model()
        
        # Normalize texts
        normalized_texts = [text.lower().strip() if text else "" for text in texts]
        
        # Check cache for each text
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(normalized_texts):
            if not text:
                embeddings.append(np.zeros(384))
                continue
                
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cached_embedding = self._get_cached_embedding(text_hash)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
                
                # Fill in the placeholders and cache new embeddings
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    
                    # Cache the new embedding
                    text = normalized_texts[idx]
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self._cache_embedding(text_hash, embedding)
                    
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                # Fill remaining placeholders with zero vectors
                for idx in uncached_indices:
                    if embeddings[idx] is None:
                        embeddings[idx] = np.zeros(384)
        
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Reshape to 2D arrays for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find most similar embeddings to query."""
        if not candidate_embeddings:
            return []
        
        try:
            # Calculate similarities
            similarities = []
            query_emb = query_embedding.reshape(1, -1)
            
            for i, candidate in enumerate(candidate_embeddings):
                if candidate is not None and len(candidate) > 0:
                    candidate_emb = candidate.reshape(1, -1)
                    similarity = cosine_similarity(query_emb, candidate_emb)[0][0]
                    similarities.append((i, float(similarity)))
                else:
                    similarities.append((i, 0.0))
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []


class PlaceSemanticService:
    """Service for semantic analysis and matching of places."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.place_type_embeddings = {}
        self._initialize_place_type_embeddings()
    
    def _initialize_place_type_embeddings(self):
        """Initialize embeddings for common place types."""
        place_types = {
            "park": "green space with trees, grass, peaceful, nature, recreation",
            "lake": "water body, peaceful, scenic, nature, reflection, swimming",
            "river": "flowing water, nature, peaceful, scenic, banks, bridges",
            "forest": "trees, nature, hiking, peaceful, wildlife, green",
            "cafe": "coffee, food, indoor, social, urban, cozy, meeting place",
            "restaurant": "food, dining, social, indoor, cuisine, service",
            "museum": "culture, art, history, education, indoor, exhibits",
            "library": "books, quiet, study, education, indoor, peaceful",
            "playground": "children, recreation, outdoor, play, family",
            "beach": "water, sand, recreation, nature, swimming, sunny",
            "mountain": "hiking, nature, elevation, scenic, outdoor, adventure",
            "trail": "walking, hiking, nature, outdoor, path, exercise",
            "garden": "plants, flowers, peaceful, nature, beauty, cultivation",
            "viewpoint": "scenic, vista, elevated, nature, photography, panoramic",
            "bridge": "crossing, water, architecture, scenic, connection",
            "historic_site": "history, culture, heritage, architecture, education",
            "shopping_center": "retail, indoor, commercial, variety, urban",
            "church": "spiritual, architecture, peaceful, community, historic",
            "stadium": "sports, events, large, outdoor, recreation, crowds",
            "hospital": "medical, healthcare, urban, essential, emergency"
        }
        
        # Generate embeddings for place type descriptions
        descriptions = list(place_types.values())
        embeddings = self.embedding_service.generate_batch_embeddings(descriptions)
        
        for (place_type, _), embedding in zip(place_types.items(), embeddings):
            self.place_type_embeddings[place_type] = embedding
    
    def classify_place_type(self, place_description: str, osm_tags: Dict[str, Any] = None) -> str:
        """Classify place type based on description and OSM tags."""
        # First, try to classify based on OSM tags
        if osm_tags:
            osm_type = self._classify_from_osm_tags(osm_tags)
            if osm_type:
                return osm_type
        
        # Fall back to semantic classification
        if place_description:
            return self._classify_from_description(place_description)
        
        return "unknown"
    
    def _classify_from_osm_tags(self, osm_tags: Dict[str, Any]) -> Optional[str]:
        """Classify place based on OSM tags."""
        # Map common OSM tags to our place types
        tag_mappings = {
            ("leisure", "park"): "park",
            ("natural", "water"): "lake",
            ("waterway", "river"): "river",
            ("landuse", "forest"): "forest",
            ("natural", "forest"): "forest",
            ("amenity", "cafe"): "cafe",
            ("amenity", "restaurant"): "restaurant",
            ("tourism", "museum"): "museum",
            ("amenity", "library"): "library",
            ("leisure", "playground"): "playground",
            ("natural", "beach"): "beach",
            ("natural", "peak"): "mountain",
            ("highway", "path"): "trail",
            ("leisure", "garden"): "garden",
            ("tourism", "viewpoint"): "viewpoint",
            ("man_made", "bridge"): "bridge",
            ("historic", "*"): "historic_site",
            ("shop", "*"): "shopping_center",
            ("amenity", "place_of_worship"): "church",
            ("leisure", "stadium"): "stadium",
            ("amenity", "hospital"): "hospital"
        }
        
        for (key, value), place_type in tag_mappings.items():
            if key in osm_tags:
                if value == "*" or osm_tags[key] == value:
                    return place_type
        
        return None
    
    def _classify_from_description(self, description: str) -> str:
        """Classify place based on semantic description."""
        if not description:
            return "unknown"
        
        # Generate embedding for description
        desc_embedding = self.embedding_service.generate_embedding(description)
        
        # Find most similar place type
        best_similarity = -1
        best_type = "unknown"
        
        for place_type, type_embedding in self.place_type_embeddings.items():
            similarity = self.embedding_service.calculate_similarity(desc_embedding, type_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = place_type
        
        # Only return classification if similarity is above threshold
        if best_similarity > 0.3:
            return best_type
        
        return "unknown"
    
    def generate_place_description(self, name: str, place_type: str, osm_tags: Dict[str, Any] = None) -> str:
        """Generate a comprehensive description for a place."""
        description_parts = []
        
        # Add name
        if name:
            description_parts.append(name)
        
        # Add place type description
        if place_type in self.place_type_embeddings:
            # Use our predefined descriptions
            type_descriptions = {
                "park": "green space with trees and grass",
                "lake": "peaceful water body",
                "river": "flowing water with natural banks",
                "forest": "dense area of trees and nature",
                "cafe": "cozy place for coffee and food",
                # Add more as needed
            }
            if place_type in type_descriptions:
                description_parts.append(type_descriptions[place_type])
        
        # Add relevant OSM tag information
        if osm_tags:
            relevant_tags = ["description", "name:en", "amenity", "leisure", "natural", "tourism"]
            for tag in relevant_tags:
                if tag in osm_tags and osm_tags[tag] not in description_parts:
                    description_parts.append(str(osm_tags[tag]))
        
        return " ".join(description_parts)
    
    def match_user_preferences(
        self, 
        user_query: str, 
        places: List[Dict[str, Any]],
        threshold: float = 0.3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Match places against user preferences."""
        if not user_query or not places:
            return []
        
        # Generate embedding for user query
        query_embedding = self.embedding_service.generate_embedding(user_query)
        
        # Generate embeddings for all places
        place_descriptions = []
        for place in places:
            # Create comprehensive description for each place
            desc = self.generate_place_description(
                place.get("name", ""),
                place.get("place_type", ""),
                place.get("tags", {})
            )
            place_descriptions.append(desc)
        
        place_embeddings = self.embedding_service.generate_batch_embeddings(place_descriptions)
        
        # Calculate similarities
        matches = []
        for place, embedding in zip(places, place_embeddings):
            similarity = self.embedding_service.calculate_similarity(query_embedding, embedding)
            if similarity >= threshold:
                matches.append((place, similarity))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def cluster_places(self, places: List[Dict[str, Any]], n_clusters: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster places based on semantic similarity."""
        if len(places) < n_clusters:
            # If we have fewer places than clusters, each place gets its own cluster
            return {i: [place] for i, place in enumerate(places)}
        
        # Generate embeddings for all places
        place_descriptions = []
        for place in places:
            desc = self.generate_place_description(
                place.get("name", ""),
                place.get("place_type", ""),
                place.get("tags", {})
            )
            place_descriptions.append(desc)
        
        embeddings = self.embedding_service.generate_batch_embeddings(place_descriptions)
        
        # Perform k-means clustering
        try:
            embeddings_array = np.array(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Group places by cluster
            clusters = {}
            for place, label in zip(places, cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(place)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to cluster places: {e}")
            # Return all places in one cluster as fallback
            return {0: places}