"""
Semantic Scoring System - Extensible proximity-based scoring for any semantic property
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString
from shapely.ops import unary_union
import numpy as np
from loguru import logger


@dataclass
class SemanticProperty:
    """Configuration for a semantic property (e.g., forest, cafe, beach)"""
    name: str
    description: str
    proximity_threshold: float  # meters - how close is "close enough"
    scoring_weight: float = 1.0  # relative importance (0.0 to 1.0)
    falloff_function: str = "exponential"  # "linear", "exponential", "gaussian"
    enabled: bool = True


class SemanticLabeler(ABC):
    """Abstract base class for labeling geographic features"""
    
    @abstractmethod
    def label_features(self, bbox_data: Dict) -> List[Dict]:
        """
        Extract and label features from geographic data.
        
        Args:
            bbox_data: Geographic data for a bounding box
            
        Returns:
            List of labeled features with geometry
        """
        pass
    
    @abstractmethod
    def get_feature_geometries(self, features: List[Dict]) -> List[Any]:
        """
        Convert features to Shapely geometries for distance calculations.
        
        Args:
            features: List of labeled features
            
        Returns:
            List of Shapely geometry objects
        """
        pass


class OSMSemanticLabeler(SemanticLabeler):
    """Labeler for OpenStreetMap-based semantic properties"""
    
    def __init__(self, osm_queries: Dict[str, str]):
        """
        Initialize with OSM Overpass queries for different feature types.
        
        Args:
            osm_queries: Dict mapping feature types to Overpass query templates
        """
        self.osm_queries = osm_queries
    
    def label_features(self, osm_data: Dict) -> List[Dict]:
        """Extract features from OSM GeoJSON data"""
        features = []
        
        if 'features' in osm_data:
            for feature in osm_data['features']:
                labeled_feature = {
                    'geometry': feature.get('geometry'),
                    'properties': feature.get('properties', {}),
                    'feature_id': feature.get('properties', {}).get('id'),
                    'tags': feature.get('properties', {}).get('tags', {})
                }
                features.append(labeled_feature)
        
        return features
    
    def get_feature_geometries(self, features: List[Dict]) -> List[Any]:
        """Convert GeoJSON geometries to Shapely objects"""
        geometries = []
        
        for feature in features:
            geom_data = feature.get('geometry')
            if not geom_data:
                continue
                
            try:
                shapely_geom = self._geojson_to_shapely(geom_data)
                if shapely_geom and not shapely_geom.is_empty:
                    geometries.append(shapely_geom)
            except Exception as e:
                logger.warning(f"Failed to convert geometry: {e}")
                continue
        
        return geometries
    
    def _geojson_to_shapely(self, geom_data: Dict) -> Optional[Any]:
        """Convert GeoJSON geometry to Shapely geometry"""
        geom_type = geom_data.get('type')
        coordinates = geom_data.get('coordinates', [])
        
        if not coordinates:
            return None
        
        try:
            if geom_type == 'Point':
                return Point(coordinates)
            elif geom_type == 'LineString':
                return LineString(coordinates)
            elif geom_type == 'Polygon':
                if len(coordinates) > 0:
                    exterior = coordinates[0]
                    holes = coordinates[1:] if len(coordinates) > 1 else []
                    return Polygon(exterior, holes)
            elif geom_type == 'MultiPolygon':
                polygons = []
                for poly_coords in coordinates:
                    if len(poly_coords) > 0:
                        exterior = poly_coords[0]
                        holes = poly_coords[1:] if len(poly_coords) > 1 else []
                        polygons.append(Polygon(exterior, holes))
                return MultiPolygon(polygons) if polygons else None
            elif geom_type == 'MultiLineString':
                return MultiLineString([LineString(line) for line in coordinates])
        except Exception as e:
            logger.warning(f"Error creating {geom_type} geometry: {e}")
            return None
        
        return None


class POISemanticLabeler(SemanticLabeler):
    """Labeler for POI-based semantic properties (cafes, restaurants, etc.)"""
    
    def __init__(self, poi_categories: List[str]):
        """
        Initialize with POI categories to look for.
        
        Args:
            poi_categories: List of POI categories (e.g., ['cafe', 'restaurant'])
        """
        self.poi_categories = poi_categories
    
    def label_features(self, poi_data: Dict) -> List[Dict]:
        """Extract POI features - to be implemented when POI data source is available"""
        # Placeholder for POI data processing
        # This would integrate with external POI APIs or databases
        return []
    
    def get_feature_geometries(self, features: List[Dict]) -> List[Any]:
        """Convert POI locations to Point geometries"""
        geometries = []
        
        for feature in features:
            lat = feature.get('lat')
            lon = feature.get('lon')
            if lat is not None and lon is not None:
                try:
                    geometries.append(Point(lon, lat))
                except Exception as e:
                    logger.warning(f"Failed to create POI point: {e}")
                    continue
        
        return geometries


class SemanticScorer:
    """Core semantic scoring system"""
    
    def __init__(self):
        self.properties: Dict[str, SemanticProperty] = {}
        self.labelers: Dict[str, SemanticLabeler] = {}
        self.feature_cache: Dict[str, List[Any]] = {}  # Cache geometries per property
        self.feature_bounds_cache: Dict[str, List[Tuple]] = {}  # Cache bounding boxes for spatial optimization
    
    def register_property(self, property_name: str, property_config: SemanticProperty, 
                         labeler: SemanticLabeler):
        """
        Register a new semantic property with its labeler.
        
        Args:
            property_name: Unique identifier for the property
            property_config: Configuration for the property
            labeler: Labeler to extract features for this property
        """
        self.properties[property_name] = property_config
        self.labelers[property_name] = labeler
        logger.info(f"Registered semantic property: {property_name}")
    
    def load_property_features(self, property_name: str, geographic_data: Dict):
        """
        Load and cache features for a specific property.
        
        Args:
            property_name: Name of the property to load
            geographic_data: Geographic data (e.g., OSM GeoJSON)
        """
        if property_name not in self.labelers:
            logger.warning(f"No labeler registered for property: {property_name}")
            return
        
        try:
            labeler = self.labelers[property_name]
            
            # Extract features using the labeler
            features = labeler.label_features(geographic_data)
            logger.info(f"Extracted {len(features)} features for {property_name}")
            
            # Convert to geometries and cache
            geometries = labeler.get_feature_geometries(features)
            self.feature_cache[property_name] = geometries
            
            # Build bounding box cache for spatial optimization
            bounds = []
            for geom in geometries:
                try:
                    if hasattr(geom, 'bounds'):
                        bounds.append(geom.bounds)  # (minx, miny, maxx, maxy)
                    else:
                        bounds.append(None)
                except Exception:
                    bounds.append(None)
            
            self.feature_bounds_cache[property_name] = bounds
            logger.info(f"Cached {len(geometries)} geometries and {len(bounds)} bounds for {property_name}")
            
        except Exception as e:
            logger.error(f"Failed to load features for {property_name}: {e}")
            self.feature_cache[property_name] = []
    
    def score_location(self, lat: float, lon: float, 
                      property_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate semantic scores for a location across all or specified properties.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            property_names: List of properties to score (None = all enabled properties)
            
        Returns:
            Dict mapping property names to scores (0.0 to 1.0)
        """
        location_point = Point(lon, lat)
        scores = {}
        
        # Determine which properties to score
        if property_names is None:
            property_names = [name for name, prop in self.properties.items() if prop.enabled]
        
        for property_name in property_names:
            if property_name not in self.properties:
                logger.warning(f"Unknown property: {property_name}")
                continue
            
            if property_name not in self.feature_cache:
                logger.warning(f"No cached features for property: {property_name}")
                scores[property_name] = 0.0
                continue
            
            property_config = self.properties[property_name]
            geometries = self.feature_cache[property_name]
            
            score = self._calculate_proximity_score(
                location_point, geometries, property_config
            )
            
            scores[property_name] = score
        
        return scores
    
    def _calculate_proximity_score(self, location: Point, geometries: List[Any], 
                                 property_config: SemanticProperty) -> float:
        """
        Calculate proximity score for a location to a set of geometries.
        
        Args:
            location: Point location to score
            geometries: List of Shapely geometries
            property_config: Configuration for the property
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not geometries:
            return 0.0
        
        min_distance = float('inf')
        
        # Get cached bounding boxes for spatial optimization
        bounds_list = self.feature_bounds_cache.get(property_config.name, [])
        
        # Find minimum distance using spatial optimization
        for i, geometry in enumerate(geometries):
            try:
                # Quick bounding box check first (much faster than geometric distance)
                if i < len(bounds_list) and bounds_list[i] is not None:
                    bounds = bounds_list[i]  # (minx, miny, maxx, maxy)
                    
                    # Quick check: if point is far from bounding box, skip expensive calculation
                    if self._point_far_from_bounds(location.x, location.y, bounds, property_config.proximity_threshold):
                        continue
                
                # Use geodesic-approximate distance (degrees to meters)
                distance = self._calculate_distance_meters(location, geometry)
                min_distance = min(min_distance, distance)
                
                # Early termination: if we're very close, no need to check other features
                if distance < 50:  # Very close (50m)
                    break
                    
            except Exception as e:
                logger.warning(f"Error calculating distance: {e}")
                continue
        
        if min_distance == float('inf'):
            return 0.0
        
        # Apply falloff function
        threshold = property_config.proximity_threshold
        weight = property_config.scoring_weight
        
        if min_distance <= 0:
            raw_score = 1.0
        else:
            # Calculate falloff based on function type - no hard cutoff for better scores!
            ratio = min_distance / threshold
            
            if property_config.falloff_function == "linear":
                # Linear falloff with gradual tail, capped at 0
                raw_score = max(0.0, 1.0 - ratio)
            elif property_config.falloff_function == "exponential":
                # Exponential falloff - more generous, always has some value
                raw_score = math.exp(-1 * ratio)  # Much more generous for urban areas
            elif property_config.falloff_function == "gaussian":
                raw_score = math.exp(-0.5 * (ratio * 2.0) ** 2)  # More generous gaussian
            else:
                # Default to exponential for better distance tolerance
                raw_score = math.exp(-1 * ratio)
        
        final_score = min(1.0, max(0.0, raw_score * weight))
        
        # Debug very small scores
        if final_score > 0 and final_score < 0.001:
            logger.debug(f"Small score calculated: raw={raw_score:.6f}, weight={weight}, final={final_score:.6f}")
        
        return final_score
    
    def _calculate_distance_meters(self, point: Point, geometry: Any) -> float:
        """
        Calculate approximate distance in meters using degree-to-meter conversion.
        
        Args:
            point: Point location
            geometry: Shapely geometry
            
        Returns:
            Distance in meters
        """
        # Get closest point on geometry
        if hasattr(geometry, 'distance'):
            distance_degrees = point.distance(geometry)
        else:
            # Fallback: distance to centroid
            distance_degrees = point.distance(geometry.centroid)
        
        # Convert degrees to meters (approximate)
        # 1 degree ≈ 111,000 meters at equator
        lat_factor = math.cos(math.radians(point.y))
        meters_per_degree = 111000 * lat_factor
        
        return distance_degrees * meters_per_degree
    
    def score_multiple_locations(self, locations: List[Tuple[float, float]], 
                               property_names: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """
        Score multiple locations efficiently.
        
        Args:
            locations: List of (lat, lon) tuples
            property_names: List of properties to score
            
        Returns:
            List of score dictionaries for each location
        """
        return [self.score_location(lat, lon, property_names) for lat, lon in locations]
    
    def get_property_info(self) -> Dict[str, Dict]:
        """Get information about all registered properties"""
        info = {}
        
        for name, prop in self.properties.items():
            info[name] = {
                'description': prop.description,
                'proximity_threshold': prop.proximity_threshold,
                'scoring_weight': prop.scoring_weight,
                'falloff_function': prop.falloff_function,
                'enabled': prop.enabled,
                'cached_features': len(self.feature_cache.get(name, []))
            }
        
        return info
    
    def clear_cache(self, property_name: Optional[str] = None):
        """Clear feature cache for a property or all properties"""
        if property_name:
            self.feature_cache.pop(property_name, None)
            self.feature_bounds_cache.pop(property_name, None)
            logger.info(f"Cleared cache for {property_name}")
        else:
            self.feature_cache.clear()
            self.feature_bounds_cache.clear()
            logger.info("Cleared all feature caches")
    
    def update_property_config(self, property_name: str, **kwargs):
        """Update configuration for an existing property"""
        if property_name not in self.properties:
            logger.warning(f"Property {property_name} not found")
            return
        
        prop = self.properties[property_name]
        
        for key, value in kwargs.items():
            if hasattr(prop, key):
                setattr(prop, key, value)
                logger.info(f"Updated {property_name}.{key} = {value}")
            else:
                logger.warning(f"Unknown property attribute: {key}")
    
    def _point_far_from_bounds(self, point_x: float, point_y: float, bounds: Tuple, threshold_m: float) -> bool:
        """
        Quick bounding box check to skip expensive distance calculations.
        
        Returns True if point is definitely too far from the bounding box.
        Uses approximate degree-to-meter conversion for speed.
        """
        minx, miny, maxx, maxy = bounds
        
        # Approximate conversion: 1 degree ≈ 111km
        # Add small buffer to account for approximation errors
        threshold_degrees = (threshold_m * 1.5) / 111000  # 1.5x buffer for safety
        
        # Check if point is outside expanded bounding box
        if (point_x < minx - threshold_degrees or 
            point_x > maxx + threshold_degrees or
            point_y < miny - threshold_degrees or 
            point_y > maxy + threshold_degrees):
            return True
        
        return False


def create_default_semantic_scorer() -> SemanticScorer:
    """Create a semantic scorer with default natural feature properties"""
    scorer = SemanticScorer()
    
    # Define OSM queries for natural features
    osm_queries = {
        'forests': """
            [out:json][timeout:30];
            (
                way["natural"="wood"](bbox);
                way["landuse"="forest"](bbox);
                relation["natural"="wood"](bbox);
                relation["landuse"="forest"](bbox);
            );
            out geom;
        """,
        'rivers': """
            [out:json][timeout:30];
            (
                way["waterway"="river"](bbox);
                way["waterway"="stream"](bbox);
                way["waterway"="canal"](bbox);
                relation["waterway"="river"](bbox);
                relation["waterway"="stream"](bbox);
            );
            out geom;
        """,
        'lakes': """
            [out:json][timeout:30];
            (
                way["natural"="water"](bbox);
                way["leisure"="swimming_pool"](bbox);
                relation["natural"="water"](bbox);
                way["water"](bbox);
            );
            out geom;
        """
    }
    
    # Create OSM labeler
    osm_labeler = OSMSemanticLabeler(osm_queries)
    
    # Register forest property
    scorer.register_property(
        'forests',
        SemanticProperty(
            name='forests',
            description='Proximity to forests, parks, and wooded areas',
            proximity_threshold=500.0,  # 500 meters
            scoring_weight=1.0,
            falloff_function='exponential'
        ),
        osm_labeler
    )
    
    # Register rivers property
    scorer.register_property(
        'rivers',
        SemanticProperty(
            name='rivers',
            description='Proximity to rivers, streams, and waterways',
            proximity_threshold=200.0,  # 200 meters
            scoring_weight=1.0,
            falloff_function='exponential'
        ),
        osm_labeler
    )
    
    # Register lakes property
    scorer.register_property(
        'lakes',
        SemanticProperty(
            name='lakes',
            description='Proximity to lakes, ponds, and water bodies',
            proximity_threshold=300.0,  # 300 meters
            scoring_weight=1.0,
            falloff_function='exponential'
        ),
        osm_labeler
    )
    
    logger.info("Created default semantic scorer with natural features")
    return scorer