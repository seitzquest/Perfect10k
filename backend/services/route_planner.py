from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import uuid
# Optional rasterio import
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    rasterio = None
from datetime import datetime, timedelta
import asyncio

from models.models import Route, User, RouteCache
from models.schemas import RouteResponse, RouteHistoryResponse
from core.route_algorithms import (
    MapLoader, RouteOptimizer, RouteConfig, TileValueFunction, 
    ElevationHandler, RouteExporter
)
from core.value_function import SpatialValueFunction, ValueConfig, ValueFunctionOptimizer
from core.config import settings
from services.place_service import PlaceService
from services.preference_service import PreferenceService
import osmnx as ox
import hashlib
import json


class RoutePlannerService:
    """Main service for route planning operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.place_service = PlaceService(db)
        self.preference_service = PreferenceService(db)
    
    async def plan_route(
        self,
        user_id: uuid.UUID,
        latitude: float,
        longitude: float,
        target_distance: int = 8000,
        tolerance: int = 1000,
        preference_query: Optional[str] = None,
        min_elevation_gain: Optional[int] = None,
        avoid_roads: bool = True
    ) -> RouteResponse:
        """
        Plan a new route with user preferences and constraints.
        """
        try:
            # Check cache first
            cached_route = await self._check_route_cache(
                latitude, longitude, target_distance, preference_query
            )
            if cached_route:
                return cached_route
            
            # Configure route planning
            config = RouteConfig(
                target_distance=target_distance,
                tolerance=tolerance,
                min_elevation_gain=min_elevation_gain,
                avoid_roads=avoid_roads
            )
            
            # Load map data
            point = (latitude, longitude)
            G, nodes, edges = MapLoader.load_map(point)
            
            # Find nearest node to start location
            start_node = ox.nearest_nodes(G, longitude, latitude)
            
            # Create advanced spatial value function
            bounds = {
                'minx': edges.bounds.minx.min(),
                'maxx': edges.bounds.maxx.max(),
                'miny': edges.bounds.miny.min(),
                'maxy': edges.bounds.maxy.max()
            }
            
            # Configure value function based on user preferences
            value_config = ValueConfig(
                spatial_resolution=50.0,  # 50m resolution for detailed analysis
                preference_weight=0.5 if preference_query else 0.1,
                elevation_weight=0.3 if min_elevation_gain else 0.1,
                accessibility_weight=0.4,
                diversity_weight=0.1
            )
            
            value_function = SpatialValueFunction(bounds, value_config)
            
            # Add accessibility constraints from graph
            value_function.add_accessibility_constraints(G)
            
            # Apply semantic preferences if provided
            matched_places = []
            if preference_query:
                matched_places = await self._apply_semantic_preferences(
                    latitude, longitude, preference_query, value_function
                )
            
            # Add elevation preferences if specified
            if min_elevation_gain and elevation_dataset:
                try:
                    # Load elevation data for the area (this would need proper implementation)
                    # For now, we'll use a moderate elevation preference
                    value_function.add_elevation_preferences(
                        elevation_data=None,  # Would load actual elevation grid
                        preference_type='moderate'
                    )
                except Exception as e:
                    print(f"Warning: Could not apply elevation preferences: {e}")
            
            # Get user's previous routes for diversity
            try:
                user_routes = await self.get_user_routes(user_id, page=1, size=5)
                if user_routes.routes:
                    previous_coords = []
                    for route in user_routes.routes:
                        if route.path_coordinates:
                            previous_coords.append(route.path_coordinates)
                    
                    if previous_coords:
                        value_function.add_diversity_bonus(previous_coords)
            except Exception as e:
                print(f"Warning: Could not apply diversity bonus: {e}")
            
            # Compute the final combined value function
            combined_values = value_function.compute_combined_value()
            
            # Initialize route optimizer
            optimizer = RouteOptimizer(config)
            
            # Load elevation data if needed
            elevation_dataset = None
            if min_elevation_gain and settings.SRTM_DATA_PATH and RASTERIO_AVAILABLE:
                try:
                    elevation_dataset = rasterio.open(settings.SRTM_DATA_PATH)
                except Exception:
                    pass  # Continue without elevation data
            
            # Find optimal route
            optimal_path = optimizer.find_optimal_route(
                G, start_node, value_function, elevation_dataset
            )
            
            # Calculate route metrics
            actual_distance = optimizer.calculate_path_length(G, optimal_path)
            coords = optimizer.path_to_coords_list(G, optimal_path)
            
            # Calculate elevation gain if elevation data available
            elevation_gain = None
            if elevation_dataset and coords:
                try:
                    _, elevations = ElevationHandler.get_elevation_profile(elevation_dataset, coords)
                    elevation_gain = ElevationHandler.calculate_elevation_gain(elevations)
                finally:
                    if elevation_dataset:
                        elevation_dataset.close()
            
            # Generate GPX export
            gpx_data = RouteExporter.export_to_gpx(G, optimal_path)
            
            # Save route to database
            db_route = Route(
                user_id=user_id,
                start_latitude=latitude,
                start_longitude=longitude,
                target_distance=target_distance,
                actual_distance=actual_distance,
                path_nodes=optimal_path,
                gpx_data=gpx_data,
                elevation_gain=elevation_gain,
                preference_query=preference_query,
                matched_places=matched_places
            )
            
            self.db.add(db_route)
            self.db.commit()
            self.db.refresh(db_route)
            
            # Cache the result
            await self._cache_route_result(
                latitude, longitude, target_distance, preference_query, db_route
            )
            
            # Convert coordinates for response
            path_coordinates = [[coord[0], coord[1]] for coord in coords]
            
            return RouteResponse(
                id=db_route.id,
                actual_distance=actual_distance,
                path_coordinates=path_coordinates,
                elevation_gain=elevation_gain,
                matched_places=matched_places,
                gpx_data=gpx_data,
                created_at=db_route.created_at
            )
            
        except Exception as e:
            # Log error here
            raise Exception(f"Route planning failed: {str(e)}")
    
    async def _apply_semantic_preferences(
        self,
        latitude: float,
        longitude: float,
        preference_query: str,
        value_function,
        radius: float = 5000
    ) -> List[Dict[str, Any]]:
        """Apply semantic preferences to value function."""
        try:
            # Search for places matching preferences
            places = await self.place_service.search_places(
                latitude=latitude,
                longitude=longitude,
                radius=radius,
                query=preference_query,
                limit=50
            )
            
            matched_places = []
            places_for_value_function = []
            similarity_scores = []
            
            # Collect place data for value function
            for place in places:
                if place.similarity_score and place.similarity_score > 0.3:  # Minimum threshold
                    places_for_value_function.append({
                        'latitude': place.latitude,
                        'longitude': place.longitude,
                        'name': place.name,
                        'type': place.place_type
                    })
                    similarity_scores.append(place.similarity_score)
                    
                    # Add to matched places list
                    matched_places.append({
                        "id": str(place.id),
                        "name": place.name,
                        "type": place.place_type,
                        "latitude": place.latitude,
                        "longitude": place.longitude,
                        "similarity_score": place.similarity_score
                    })
            
            # Apply places to value function using advanced spatial influence
            if places_for_value_function and isinstance(value_function, SpatialValueFunction):
                value_function.add_place_preferences(places_for_value_function, similarity_scores)
            elif places_for_value_function:
                # Fallback for legacy TileValueFunction
                for place in places_for_value_function:
                    place_value = 0.8 + (place.get('similarity_score', 0.5)) * 0.2
                    value_function.set_value(place['latitude'], place['longitude'], place_value)
            
            return matched_places
            
        except Exception as e:
            # Log warning and continue without preferences
            print(f"Warning: Failed to apply semantic preferences: {str(e)}")
            return []
    
    async def get_user_routes(
        self,
        user_id: uuid.UUID,
        page: int = 1,
        size: int = 20
    ) -> RouteHistoryResponse:
        """Get user's route history with pagination."""
        offset = (page - 1) * size
        
        # Get total count
        total = self.db.query(Route).filter(Route.user_id == user_id).count()
        
        # Get paginated routes
        routes = (
            self.db.query(Route)
            .filter(Route.user_id == user_id)
            .order_by(Route.created_at.desc())
            .offset(offset)
            .limit(size)
            .all()
        )
        
        # Convert to response format
        route_responses = []
        for route in routes:
            # Reconstruct coordinates from path_nodes if needed
            path_coordinates = []
            if route.path_nodes:
                # This would need the original graph to convert nodes back to coordinates
                # For now, we'll return empty coordinates or store them separately
                pass
            
            route_responses.append(RouteResponse(
                id=route.id,
                actual_distance=route.actual_distance,
                path_coordinates=path_coordinates,
                elevation_gain=route.elevation_gain,
                matched_places=route.matched_places,
                gpx_data=route.gpx_data,
                created_at=route.created_at
            ))
        
        return RouteHistoryResponse(
            routes=route_responses,
            total=total,
            page=page,
            size=size
        )
    
    async def get_route_by_id(self, route_id: uuid.UUID, user_id: uuid.UUID) -> Optional[RouteResponse]:
        """Get a specific route by ID."""
        route = (
            self.db.query(Route)
            .filter(Route.id == route_id, Route.user_id == user_id)
            .first()
        )
        
        if not route:
            return None
        
        # Convert to response format (similar to get_user_routes)
        return RouteResponse(
            id=route.id,
            actual_distance=route.actual_distance,
            path_coordinates=[],  # Would need graph reconstruction
            elevation_gain=route.elevation_gain,
            matched_places=route.matched_places,
            gpx_data=route.gpx_data,
            created_at=route.created_at
        )
    
    async def delete_route(self, route_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Delete a route."""
        route = (
            self.db.query(Route)
            .filter(Route.id == route_id, Route.user_id == user_id)
            .first()
        )
        
        if not route:
            return False
        
        self.db.delete(route)
        self.db.commit()
        return True
    
    async def export_route_gpx(self, route_id: uuid.UUID, user_id: uuid.UUID) -> Optional[str]:
        """Export route as GPX content."""
        route = (
            self.db.query(Route)
            .filter(Route.id == route_id, Route.user_id == user_id)
            .first()
        )
        
        if not route:
            return None
        
        return route.gpx_data
    
    async def _check_route_cache(
        self,
        latitude: float,
        longitude: float,
        distance: int,
        preference_query: Optional[str]
    ) -> Optional[RouteResponse]:
        """Check if a similar route exists in cache."""
        try:
            # Create cache key
            preference_hash = None
            if preference_query:
                preference_hash = hashlib.md5(preference_query.encode()).hexdigest()
            
            # Look for cached routes within reasonable distance (100m) and same preferences
            lat_tolerance = 0.001  # Roughly 100m
            lon_tolerance = 0.001
            
            cached_route = (
                self.db.query(RouteCache)
                .filter(
                    RouteCache.latitude.between(latitude - lat_tolerance, latitude + lat_tolerance),
                    RouteCache.longitude.between(longitude - lon_tolerance, longitude + lon_tolerance),
                    RouteCache.distance == distance,
                    RouteCache.preference_hash == preference_hash,
                    RouteCache.expires_at > datetime.utcnow()
                )
                .first()
            )
            
            if cached_route:
                # Update hit count
                cached_route.hit_count += 1
                self.db.commit()
                
                # Convert cached route to response
                route_data = cached_route.cached_route
                return RouteResponse(**route_data)
            
            return None
            
        except Exception:
            # If cache check fails, continue without cache
            return None
    
    async def _cache_route_result(
        self,
        latitude: float,
        longitude: float,
        distance: int,
        preference_query: Optional[str],
        route: Route
    ):
        """Cache a route result."""
        try:
            preference_hash = None
            if preference_query:
                preference_hash = hashlib.md5(preference_query.encode()).hexdigest()
            
            # Create cache entry
            cached_route = RouteCache(
                latitude=latitude,
                longitude=longitude,
                distance=distance,
                preference_hash=preference_hash,
                cached_route={
                    "id": str(route.id),
                    "actual_distance": route.actual_distance,
                    "path_coordinates": [],  # Would store actual coordinates
                    "elevation_gain": route.elevation_gain,
                    "matched_places": route.matched_places,
                    "gpx_data": route.gpx_data,
                    "created_at": route.created_at.isoformat()
                },
                expires_at=datetime.utcnow() + timedelta(hours=24)  # Cache for 24 hours
            )
            
            self.db.add(cached_route)
            self.db.commit()
            
        except Exception:
            # If caching fails, continue without caching
            pass