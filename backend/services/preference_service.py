from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uuid
import numpy as np
from datetime import datetime
import logging

from models.models import UserPreference, User
from models.schemas import PreferenceResponse, PreferenceCreate
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class PreferenceService:
    """Service for managing user preferences and semantic matching."""
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService()
    
    async def create_preference(
        self,
        user_id: uuid.UUID,
        description: str,
        weight: float = 1.0
    ) -> PreferenceResponse:
        """Create a new user preference with embedding."""
        try:
            # Generate embedding for the preference description
            embedding = self.embedding_service.generate_embedding(description)
            
            # Create preference record
            preference = UserPreference(
                user_id=user_id,
                description=description,
                embedding=self._serialize_embedding(embedding),
                weight=weight
            )
            
            self.db.add(preference)
            self.db.commit()
            self.db.refresh(preference)
            
            return PreferenceResponse(
                id=preference.id,
                description=preference.description,
                weight=preference.weight,
                created_at=preference.created_at,
                is_active=preference.is_active
            )
            
        except Exception as e:
            logger.error(f"Failed to create preference: {e}")
            self.db.rollback()
            raise
    
    async def get_user_preferences(self, user_id: uuid.UUID) -> List[PreferenceResponse]:
        """Get all preferences for a user."""
        try:
            preferences = (
                self.db.query(UserPreference)
                .filter(UserPreference.user_id == user_id)
                .order_by(UserPreference.created_at.desc())
                .all()
            )
            
            return [
                PreferenceResponse(
                    id=pref.id,
                    description=pref.description,
                    weight=pref.weight,
                    created_at=pref.created_at,
                    is_active=pref.is_active
                )
                for pref in preferences
            ]
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return []
    
    async def update_preference(
        self,
        preference_id: uuid.UUID,
        user_id: uuid.UUID,
        description: str,
        weight: float
    ) -> Optional[PreferenceResponse]:
        """Update a user preference."""
        try:
            preference = (
                self.db.query(UserPreference)
                .filter(
                    UserPreference.id == preference_id,
                    UserPreference.user_id == user_id
                )
                .first()
            )
            
            if not preference:
                return None
            
            # Update fields
            preference.description = description
            preference.weight = weight
            
            # Regenerate embedding if description changed
            new_embedding = self.embedding_service.generate_embedding(description)
            preference.embedding = self._serialize_embedding(new_embedding)
            
            self.db.commit()
            self.db.refresh(preference)
            
            return PreferenceResponse(
                id=preference.id,
                description=preference.description,
                weight=preference.weight,
                created_at=preference.created_at,
                is_active=preference.is_active
            )
            
        except Exception as e:
            logger.error(f"Failed to update preference: {e}")
            self.db.rollback()
            return None
    
    async def delete_preference(self, preference_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Delete a user preference."""
        try:
            preference = (
                self.db.query(UserPreference)
                .filter(
                    UserPreference.id == preference_id,
                    UserPreference.user_id == user_id
                )
                .first()
            )
            
            if not preference:
                return False
            
            self.db.delete(preference)
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete preference: {e}")
            self.db.rollback()
            return False
    
    async def toggle_preference(
        self,
        preference_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> Optional[PreferenceResponse]:
        """Toggle a preference active/inactive."""
        try:
            preference = (
                self.db.query(UserPreference)
                .filter(
                    UserPreference.id == preference_id,
                    UserPreference.user_id == user_id
                )
                .first()
            )
            
            if not preference:
                return None
            
            preference.is_active = not preference.is_active
            self.db.commit()
            self.db.refresh(preference)
            
            return PreferenceResponse(
                id=preference.id,
                description=preference.description,
                weight=preference.weight,
                created_at=preference.created_at,
                is_active=preference.is_active
            )
            
        except Exception as e:
            logger.error(f"Failed to toggle preference: {e}")
            self.db.rollback()
            return None
    
    async def get_combined_user_preferences(self, user_id: uuid.UUID) -> Optional[str]:
        """Get combined user preferences as a single query string."""
        try:
            active_preferences = (
                self.db.query(UserPreference)
                .filter(
                    UserPreference.user_id == user_id,
                    UserPreference.is_active == True
                )
                .all()
            )
            
            if not active_preferences:
                return None
            
            # Combine preferences weighted by their weight values
            weighted_descriptions = []
            for pref in active_preferences:
                # Repeat description based on weight (rounded to integer)
                repeat_count = max(1, int(round(pref.weight)))
                weighted_descriptions.extend([pref.description] * repeat_count)
            
            return " ".join(weighted_descriptions)
            
        except Exception as e:
            logger.error(f"Failed to get combined preferences: {e}")
            return None
    
    async def get_user_preference_embedding(self, user_id: uuid.UUID) -> Optional[np.ndarray]:
        """Get combined preference embedding for a user."""
        try:
            combined_preferences = await self.get_combined_user_preferences(user_id)
            
            if not combined_preferences:
                return None
            
            return self.embedding_service.generate_embedding(combined_preferences)
            
        except Exception as e:
            logger.error(f"Failed to get user preference embedding: {e}")
            return None
    
    async def find_similar_preferences(
        self,
        user_id: uuid.UUID,
        query: str,
        threshold: float = 0.5
    ) -> List[PreferenceResponse]:
        """Find user preferences similar to a query."""
        try:
            # Get user's preferences
            user_preferences = await self.get_user_preferences(user_id)
            
            if not user_preferences:
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Get embeddings for user preferences
            similar_preferences = []
            for pref in user_preferences:
                # Get stored embedding
                db_pref = (
                    self.db.query(UserPreference)
                    .filter(UserPreference.id == pref.id)
                    .first()
                )
                
                if db_pref and db_pref.embedding:
                    pref_embedding = self._deserialize_embedding(db_pref.embedding)
                    similarity = self.embedding_service.calculate_similarity(
                        query_embedding, pref_embedding
                    )
                    
                    if similarity >= threshold:
                        similar_preferences.append((pref, similarity))
            
            # Sort by similarity and return
            similar_preferences.sort(key=lambda x: x[1], reverse=True)
            return [pref for pref, _ in similar_preferences]
            
        except Exception as e:
            logger.error(f"Failed to find similar preferences: {e}")
            return []
    
    async def suggest_preferences_from_routes(
        self,
        user_id: uuid.UUID,
        limit: int = 5
    ) -> List[str]:
        """Suggest new preferences based on user's route history."""
        try:
            from models.models import Route
            
            # Get user's recent routes with matched places
            recent_routes = (
                self.db.query(Route)
                .filter(
                    Route.user_id == user_id,
                    Route.matched_places.isnot(None)
                )
                .order_by(Route.created_at.desc())
                .limit(10)
                .all()
            )
            
            if not recent_routes:
                return []
            
            # Extract place types and descriptions from matched places
            place_descriptions = []
            for route in recent_routes:
                if route.matched_places:
                    for place in route.matched_places:
                        if isinstance(place, dict):
                            place_type = place.get("type", "")
                            place_name = place.get("name", "")
                            if place_type:
                                place_descriptions.append(f"{place_type} like {place_name}")
            
            # Generate suggestions based on common patterns
            if place_descriptions:
                # Use embedding similarity to cluster similar place types
                embeddings = self.embedding_service.generate_batch_embeddings(place_descriptions)
                
                # For now, return the most common unique descriptions
                unique_descriptions = list(set(place_descriptions))
                return unique_descriptions[:limit]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to suggest preferences: {e}")
            return []
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy embedding to bytes for database storage."""
        return embedding.tobytes()
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Deserialize bytes back to numpy embedding."""
        return np.frombuffer(embedding_bytes, dtype=np.float32)