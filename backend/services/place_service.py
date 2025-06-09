from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any, Tuple
import uuid
import osmnx as ox
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import math

from models.models import Place
from models.schemas import PlaceResponse, PlaceSearchRequest
from services.embedding_service import PlaceSemanticService
from core.config import settings

logger = logging.getLogger(__name__)


class PlaceService:
    """Service for managing places and semantic place search."""
    
    def __init__(self, db: Session):
        self.db = db
        self.semantic_service = PlaceSemanticService()
        
    async def search_places(
        self,
        latitude: float,
        longitude: float,
        radius: float = 5000,
        query: Optional[str] = None,
        place_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[PlaceResponse]:
        """Search for places using semantic matching and geographical filtering."""
        try:
            # First, get places from database within radius
            db_places = await self._get_places_from_db(latitude, longitude, radius, place_types, limit * 2)
            
            # If we don't have enough places in DB, fetch from OSM
            if len(db_places) < limit:
                osm_places = await self._fetch_places_from_osm(latitude, longitude, radius)
                # Store new places in database
                for place_data in osm_places:
                    await self._store_place_in_db(place_data)
                
                # Re-query database
                db_places = await self._get_places_from_db(latitude, longitude, radius, place_types, limit * 2)
            
            # Convert to dict format for semantic processing
            places_dict = []
            for place in db_places:
                place_dict = {
                    "id": place.id,
                    "name": place.name,
                    "place_type": place.place_type,
                    "latitude": place.latitude,
                    "longitude": place.longitude,
                    "description": place.description,
                    "tags": place.tags or {},
                    "embedding": self._deserialize_embedding(place.embedding) if place.embedding else None
                }
                places_dict.append(place_dict)
            
            # Apply semantic filtering if query is provided
            if query:
                matches = self.semantic_service.match_user_preferences(query, places_dict)
                # Convert back to Place objects with similarity scores
                result_places = []
                for place_dict, similarity in matches[:limit]:
                    place_response = PlaceResponse(
                        id=place_dict["id"],
                        name=place_dict["name"],
                        place_type=place_dict["place_type"],
                        latitude=place_dict["latitude"],
                        longitude=place_dict["longitude"],
                        description=place_dict["description"],
                        tags=place_dict["tags"],
                        similarity_score=similarity
                    )
                    result_places.append(place_response)
                return result_places
            else:
                # Return places without semantic matching
                result_places = []
                for place in db_places[:limit]:
                    place_response = PlaceResponse(
                        id=place.id,
                        name=place.name,
                        place_type=place.place_type,
                        latitude=place.latitude,
                        longitude=place.longitude,
                        description=place.description,
                        tags=place.tags,
                        similarity_score=None
                    )
                    result_places.append(place_response)
                return result_places
                
        except Exception as e:
            logger.error(f"Failed to search places: {e}")
            return []
    
    async def get_nearby_places(
        self,
        latitude: float,
        longitude: float,
        radius: float = 5000,
        place_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[PlaceResponse]:
        """Get nearby places without semantic filtering."""
        return await self.search_places(
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            query=None,
            place_types=place_types,
            limit=limit
        )
    
    async def refresh_places_cache(
        self,
        latitude: float,
        longitude: float,
        radius: float = 10000
    ) -> int:
        """Refresh places cache for a given area by fetching from OSM."""
        try:
            # Fetch fresh data from OSM
            osm_places = await self._fetch_places_from_osm(latitude, longitude, radius)
            
            # Store/update places in database
            updated_count = 0
            for place_data in osm_places:
                if await self._store_place_in_db(place_data):
                    updated_count += 1
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to refresh places cache: {e}")
            return 0
    
    async def get_available_place_types(self) -> List[str]:
        """Get all available place types from the database."""
        try:
            types = self.db.query(Place.place_type).distinct().all()
            return [t[0] for t in types if t[0]]
        except Exception as e:
            logger.error(f"Failed to get place types: {e}")
            return []
    
    async def _get_places_from_db(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        place_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Place]:
        """Get places from database within specified radius."""
        try:
            # Calculate approximate lat/lon bounds for radius
            lat_range = radius / 111000  # Roughly 111km per degree of latitude
            lon_range = radius / (111000 * math.cos(math.radians(latitude)))
            
            query = self.db.query(Place).filter(
                Place.latitude.between(latitude - lat_range, latitude + lat_range),
                Place.longitude.between(longitude - lon_range, longitude + lon_range)
            )
            
            # Filter by place types if specified
            if place_types:
                query = query.filter(Place.place_type.in_(place_types))
            
            places = query.limit(limit).all()
            
            # Filter by exact distance
            filtered_places = []
            for place in places:
                distance = self._calculate_distance(latitude, longitude, place.latitude, place.longitude)
                if distance <= radius:
                    filtered_places.append(place)
            
            return filtered_places
            
        except Exception as e:
            logger.error(f"Failed to get places from database: {e}")
            return []
    
    async def _fetch_places_from_osm(
        self,
        latitude: float,
        longitude: float,
        radius: float
    ) -> List[Dict[str, Any]]:
        """Fetch places from OpenStreetMap."""
        try:
            # Define OSM tags for different place types
            osm_queries = {
                "leisure": ["park", "garden", "playground", "stadium"],
                "natural": ["water", "beach", "peak", "forest"],
                "amenity": ["cafe", "restaurant", "library", "hospital", "place_of_worship"],
                "tourism": ["museum", "viewpoint", "attraction"],
                "waterway": ["river"],
                "historic": True,  # Get all historic features
                "shop": True       # Get all shops
            }
            
            places = []
            point = (latitude, longitude)
            
            for tag_key, tag_values in osm_queries.items():
                try:
                    if tag_values is True:
                        # Get all features with this tag
                        tags = {tag_key: True}
                    else:
                        # Get features with specific values
                        tags = {tag_key: tag_values}
                    
                    # Fetch POIs from OSM
                    pois = ox.features_from_point(point, tags=tags, dist=radius)
                    
                    if not pois.empty:
                        for idx, poi in pois.iterrows():
                            place_data = self._process_osm_feature(poi)
                            if place_data:
                                places.append(place_data)
                                
                except Exception as e:
                    logger.warning(f"Failed to fetch OSM data for tag {tag_key}: {e}")
                    continue
            
            return places
            
        except Exception as e:
            logger.error(f"Failed to fetch places from OSM: {e}")
            return []
    
    def _process_osm_feature(self, poi) -> Optional[Dict[str, Any]]:
        """Process an OSM feature into our place format."""
        try:
            # Extract geometry
            geometry = poi.geometry
            if geometry is None:
                return None
            
            # Get centroid coordinates
            if hasattr(geometry, 'centroid'):
                centroid = geometry.centroid
                lat, lon = centroid.y, centroid.x
            elif hasattr(geometry, 'coords'):
                # For points
                coords = list(geometry.coords)[0]
                lat, lon = coords[1], coords[0]
            else:
                return None
            
            # Extract basic information
            name = poi.get('name', '') or poi.get('name:en', '') or 'Unnamed'
            osm_id = str(poi.name) if poi.name else None
            
            # Extract and clean tags
            tags = {}
            for key, value in poi.items():
                if pd.notna(value) and key not in ['geometry']:
                    tags[key] = value
            
            # Classify place type
            place_type = self.semantic_service.classify_place_type(name, tags)
            
            # Generate description
            description = self.semantic_service.generate_place_description(name, place_type, tags)
            
            return {
                "name": name,
                "place_type": place_type,
                "latitude": lat,
                "longitude": lon,
                "osm_id": osm_id,
                "osm_type": "way",  # Could be node, way, or relation
                "description": description,
                "tags": tags
            }
            
        except Exception as e:
            logger.warning(f"Failed to process OSM feature: {e}")
            return None
    
    async def _store_place_in_db(self, place_data: Dict[str, Any]) -> bool:
        """Store or update a place in the database."""
        try:
            # Check if place already exists (by OSM ID or location)
            existing_place = None
            if place_data.get("osm_id"):
                existing_place = self.db.query(Place).filter(
                    Place.osm_id == place_data["osm_id"]
                ).first()
            
            if not existing_place:
                # Check by approximate location
                lat_tolerance = 0.0001  # About 10 meters
                lon_tolerance = 0.0001
                existing_place = self.db.query(Place).filter(
                    Place.latitude.between(
                        place_data["latitude"] - lat_tolerance,
                        place_data["latitude"] + lat_tolerance
                    ),
                    Place.longitude.between(
                        place_data["longitude"] - lon_tolerance,
                        place_data["longitude"] + lon_tolerance
                    ),
                    Place.name == place_data["name"]
                ).first()
            
            # Generate embedding for the place
            embedding = self.semantic_service.embedding_service.generate_embedding(
                place_data["description"]
            )
            
            if existing_place:
                # Update existing place
                existing_place.name = place_data["name"]
                existing_place.place_type = place_data["place_type"]
                existing_place.description = place_data["description"]
                existing_place.tags = place_data["tags"]
                existing_place.embedding = self._serialize_embedding(embedding)
                existing_place.last_updated = datetime.utcnow()
            else:
                # Create new place
                new_place = Place(
                    name=place_data["name"],
                    place_type=place_data["place_type"],
                    latitude=place_data["latitude"],
                    longitude=place_data["longitude"],
                    osm_id=place_data.get("osm_id"),
                    osm_type=place_data.get("osm_type"),
                    description=place_data["description"],
                    tags=place_data["tags"],
                    embedding=self._serialize_embedding(embedding)
                )
                self.db.add(new_place)
            
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store place in database: {e}")
            self.db.rollback()
            return False
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy embedding to bytes for database storage."""
        return embedding.tobytes()
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Deserialize bytes back to numpy embedding."""
        return np.frombuffer(embedding_bytes, dtype=np.float32)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters using Haversine formula."""
        from haversine import haversine, Unit
        return haversine((lat1, lon1), (lat2, lon2), Unit.METERS)