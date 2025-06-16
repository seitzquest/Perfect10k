"""
Optimized Semantic Scoring System
High-performance implementation with spatial indexing, vectorization, and intelligent sampling.
"""

import math
import time
import json
import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict

# Heavy imports only when needed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from shapely.geometry import Point, Polygon, LineString
    from rtree import index
    
from loguru import logger


@dataclass
class OptimizedSemanticProperty:
    """Optimized configuration for semantic properties"""
    name: str
    description: str
    proximity_threshold: float
    scoring_weight: float = 1.0
    falloff_function: str = "exponential"
    enabled: bool = True
    # Optimization parameters
    early_termination_distance: float = 50.0  # meters - perfect score threshold
    max_search_distance: float = 2000.0  # meters - beyond this, score is 0
    spatial_index_enabled: bool = True


class SpatialIndex:
    """Spatial indexing using R-tree for fast proximity queries"""
    
    def __init__(self):
        self.index = None
        self.geometries = []
        self.centroids = None  # NumPy array of centroids for vectorized calculations
        self.bounds = []
        self._index_built = False
    
    def build_index(self, geometries: List[Any]):
        """Build spatial index from geometries"""
        try:
            from rtree import index
            
            self.geometries = geometries
            self.bounds = []
            centroids_list = []
            
            # Create R-tree index
            self.index = index.Index()
            
            for i, geom in enumerate(geometries):
                try:
                    if hasattr(geom, 'bounds'):
                        bounds = geom.bounds
                        self.bounds.append(bounds)
                        self.index.insert(i, bounds)
                        
                        # Extract centroid for vectorized calculations
                        if hasattr(geom, 'centroid'):
                            centroid = geom.centroid
                            centroids_list.append([centroid.x, centroid.y])
                        else:
                            # Fallback: calculate center of bounds
                            center_x = (bounds[0] + bounds[2]) / 2
                            center_y = (bounds[1] + bounds[3]) / 2
                            centroids_list.append([center_x, center_y])
                    else:
                        self.bounds.append(None)
                        centroids_list.append([0, 0])  # Fallback
                        
                except Exception as e:
                    logger.warning(f"Failed to index geometry {i}: {e}")
                    self.bounds.append(None)
                    centroids_list.append([0, 0])
            
            # Convert centroids to NumPy array for vectorized operations
            self.centroids = np.array(centroids_list) if centroids_list else np.array([])
            self._index_built = True
            
            logger.info(f"Built spatial index with {len(geometries)} geometries")
            
        except ImportError:
            logger.warning("rtree not available, falling back to linear search")
            self.geometries = geometries
            self._index_built = False
    
    def find_nearby_indices(self, lon: float, lat: float, radius_degrees: float) -> List[int]:
        """Find geometry indices within radius using spatial index"""
        if not self._index_built or self.index is None:
            # Fallback: return all indices for linear search
            return list(range(len(self.geometries)))
        
        # Query R-tree with bounding box
        bbox = (lon - radius_degrees, lat - radius_degrees,
                lon + radius_degrees, lat + radius_degrees)
        
        return list(self.index.intersection(bbox))
    
    def get_nearby_centroids_vectorized(self, lon: float, lat: float, 
                                      max_distance_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get nearby centroids using vectorized distance calculation.
        
        Returns:
            Tuple of (distances_meters, indices) for centroids within max_distance
        """
        if self.centroids is None or len(self.centroids) == 0:
            return np.array([]), np.array([])
        
        # Vectorized haversine distance calculation
        distances = haversine_vectorized(lat, lon, 
                                       self.centroids[:, 1], self.centroids[:, 0])
        
        # Filter by distance
        nearby_mask = distances <= max_distance_m
        nearby_distances = distances[nearby_mask]
        nearby_indices = np.where(nearby_mask)[0]
        
        return nearby_distances, nearby_indices


@lru_cache(maxsize=10000)
def haversine_cached(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Cached haversine distance calculation"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat_rad = math.radians(lat2 - lat1)
    dlon_rad = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat_rad/2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(dlon_rad/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def haversine_vectorized(lat1: float, lon1: float, 
                        lat2_array: np.ndarray, lon2_array: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance calculation using NumPy"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2_array)
    dlat_rad = np.radians(lat2_array - lat1)
    dlon_rad = np.radians(lon2_array - lon1)
    
    a = (np.sin(dlat_rad/2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * 
         np.sin(dlon_rad/2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


class OptimizedSemanticScorer:
    """High-performance semantic scorer with spatial indexing and vectorization"""
    
    def __init__(self, enable_vectorization: bool = True):
        self.properties: Dict[str, OptimizedSemanticProperty] = {}
        self.spatial_indexes: Dict[str, SpatialIndex] = {}
        self.feature_cache: Dict[str, List[Any]] = {}
        self.enable_vectorization = enable_vectorization
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'vectorized_queries': 0,
            'spatial_index_hits': 0,
            'cache_hits': 0,
            'avg_query_time_ms': 0.0
        }
    
    def register_property(self, property_name: str, property_config: OptimizedSemanticProperty):
        """Register a semantic property"""
        self.properties[property_name] = property_config
        logger.info(f"Registered optimized semantic property: {property_name}")
    
    def load_property_features(self, property_name: str, geometries: List[Any]):
        """Load and index features for a property"""
        if property_name not in self.properties:
            logger.warning(f"Property {property_name} not registered")
            return
        
        start_time = time.time()
        
        # Cache geometries
        self.feature_cache[property_name] = geometries
        
        # Build spatial index
        spatial_index = SpatialIndex()
        spatial_index.build_index(geometries)
        self.spatial_indexes[property_name] = spatial_index
        
        elapsed = time.time() - start_time
        logger.info(f"Loaded and indexed {len(geometries)} features for {property_name} in {elapsed:.3f}s")
    
    def score_location_optimized(self, lat: float, lon: float, 
                               property_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Optimized single location scoring"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        scores = {}
        
        if property_names is None:
            property_names = [name for name, prop in self.properties.items() if prop.enabled]
        
        for property_name in property_names:
            if property_name not in self.properties:
                continue
            
            score = self._calculate_proximity_score_optimized(lat, lon, property_name)
            scores[property_name] = score
        
        # Update performance stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['avg_query_time_ms'] = (
            (self.stats['avg_query_time_ms'] * (self.stats['total_queries'] - 1) + elapsed_ms) / 
            self.stats['total_queries']
        )
        
        return scores
    
    def score_multiple_locations_batch(self, locations: List[Tuple[float, float]], 
                                     property_names: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """Highly optimized batch scoring using vectorization"""
        if not locations:
            return []
        
        start_time = time.time()
        
        if property_names is None:
            property_names = [name for name, prop in self.properties.items() if prop.enabled]
        
        # Convert locations to numpy arrays for vectorization
        locations_array = np.array(locations)
        
        results = []
        
        # Process each property across all locations at once
        for property_name in property_names:
            if property_name not in self.properties or property_name not in self.spatial_indexes:
                # Fallback to individual scoring
                for lat, lon in locations:
                    if len(results) <= len(locations) - 1:
                        results.append({property_name: 0.0})
                continue
            
            property_scores = self._score_property_vectorized(locations_array, property_name)
            
            # Merge scores into results
            for i, score in enumerate(property_scores):
                if i >= len(results):
                    results.append({})
                results[i][property_name] = score
        
        # Ensure all results have all properties
        for result in results:
            for property_name in property_names:
                if property_name not in result:
                    result[property_name] = 0.0
        
        elapsed = time.time() - start_time
        self.stats['vectorized_queries'] += len(locations)
        logger.debug(f"Batch scored {len(locations)} locations in {elapsed:.3f}s ({elapsed/len(locations)*1000:.1f}ms per location)")
        
        return results
    
    def _score_property_vectorized(self, locations_array: np.ndarray, property_name: str) -> np.ndarray:
        """Score a single property across multiple locations using vectorization"""
        property_config = self.properties[property_name]
        spatial_index = self.spatial_indexes[property_name]
        
        if spatial_index.centroids is None or len(spatial_index.centroids) == 0:
            return np.zeros(len(locations_array))
        
        # Calculate distances from all locations to all feature centroids
        all_scores = np.zeros(len(locations_array))
        
        for i, (lat, lon) in enumerate(locations_array):
            # Use vectorized centroid calculation for quick approximation
            distances, indices = spatial_index.get_nearby_centroids_vectorized(
                lon, lat, property_config.max_search_distance
            )
            
            if len(distances) == 0:
                all_scores[i] = 0.0
                continue
            
            # Find minimum distance (best proximity)
            min_distance = np.min(distances)
            
            # Early termination for very close features
            if min_distance <= property_config.early_termination_distance:
                all_scores[i] = 1.0
                continue
            
            # Calculate score using falloff function
            score = self._apply_falloff_function(min_distance, property_config)
            all_scores[i] = score
        
        return all_scores
    
    def _calculate_proximity_score_optimized(self, lat: float, lon: float, property_name: str) -> float:
        """Optimized proximity score calculation for single location"""
        if property_name not in self.properties or property_name not in self.spatial_indexes:
            return 0.0
        
        property_config = self.properties[property_name]
        spatial_index = self.spatial_indexes[property_name]
        
        # Quick vectorized check using centroids first
        if self.enable_vectorization and spatial_index.centroids is not None:
            distances, indices = spatial_index.get_nearby_centroids_vectorized(
                lon, lat, property_config.max_search_distance
            )
            
            if len(distances) == 0:
                return 0.0
            
            min_distance = np.min(distances)
            
            # For very close features, return perfect score immediately
            if min_distance <= property_config.early_termination_distance:
                return 1.0
            
            # For moderate distances, do more precise calculation with actual geometries
            if min_distance <= property_config.proximity_threshold * 2:
                return self._precise_distance_calculation(lat, lon, property_name, indices[:5])  # Check top 5
            
            # For distant features, use centroid approximation
            return self._apply_falloff_function(min_distance, property_config)
        
        # Fallback to traditional calculation
        return self._calculate_proximity_score_traditional(lat, lon, property_name)
    
    def _precise_distance_calculation(self, lat: float, lon: float, 
                                    property_name: str, candidate_indices: np.ndarray) -> float:
        """Calculate precise distance to actual geometries for nearby candidates"""
        spatial_index = self.spatial_indexes[property_name]
        property_config = self.properties[property_name]
        
        min_distance = float('inf')
        
        try:
            from shapely.geometry import Point
            point = Point(lon, lat)
            
            for idx in candidate_indices:
                if idx < len(spatial_index.geometries):
                    geometry = spatial_index.geometries[idx]
                    
                    # Calculate precise distance
                    if hasattr(geometry, 'distance'):
                        distance_degrees = point.distance(geometry)
                        # Convert to meters
                        lat_factor = math.cos(math.radians(lat))
                        distance_meters = distance_degrees * 111000 * lat_factor
                        
                        min_distance = min(min_distance, distance_meters)
                        
                        # Early termination
                        if distance_meters <= property_config.early_termination_distance:
                            return 1.0
            
        except ImportError:
            # Fallback to cached haversine
            for idx in candidate_indices:
                if idx < len(spatial_index.centroids):
                    centroid_lon, centroid_lat = spatial_index.centroids[idx]
                    distance = haversine_cached(lat, lon, centroid_lat, centroid_lon)
                    min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0
        
        return self._apply_falloff_function(min_distance, property_config)
    
    def _calculate_proximity_score_traditional(self, lat: float, lon: float, property_name: str) -> float:
        """Traditional proximity calculation (fallback)"""
        if property_name not in self.feature_cache:
            return 0.0
        
        property_config = self.properties[property_name]
        geometries = self.feature_cache[property_name]
        
        if not geometries:
            return 0.0
        
        min_distance = float('inf')
        
        try:
            from shapely.geometry import Point
            point = Point(lon, lat)
            
            for geometry in geometries[:50]:  # Limit to first 50 for performance
                if hasattr(geometry, 'distance'):
                    distance_degrees = point.distance(geometry)
                    lat_factor = math.cos(math.radians(lat))
                    distance_meters = distance_degrees * 111000 * lat_factor
                    
                    min_distance = min(min_distance, distance_meters)
                    
                    if distance_meters <= property_config.early_termination_distance:
                        return 1.0
        
        except ImportError:
            logger.warning("Shapely not available for precise distance calculation")
            return 0.0
        
        if min_distance == float('inf'):
            return 0.0
        
        return self._apply_falloff_function(min_distance, property_config)
    
    def _apply_falloff_function(self, distance_meters: float, property_config: OptimizedSemanticProperty) -> float:
        """Apply falloff function to convert distance to score"""
        if distance_meters <= 0:
            return 1.0
        
        if distance_meters > property_config.max_search_distance:
            return 0.0
        
        threshold = property_config.proximity_threshold
        ratio = distance_meters / threshold
        
        if property_config.falloff_function == "linear":
            raw_score = max(0.0, 1.0 - ratio)
        elif property_config.falloff_function == "exponential":
            raw_score = math.exp(-ratio)
        elif property_config.falloff_function == "gaussian":
            raw_score = math.exp(-0.5 * (ratio * 2.0) ** 2)
        else:
            raw_score = math.exp(-ratio)  # Default to exponential
        
        return min(1.0, max(0.0, raw_score * property_config.scoring_weight))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'properties_loaded': len(self.properties),
            'spatial_indexes_built': len(self.spatial_indexes),
            'total_features': sum(len(cache) for cache in self.feature_cache.values()),
            'vectorization_enabled': self.enable_vectorization
        }
    
    def clear_cache(self, property_name: Optional[str] = None):
        """Clear caches"""
        if property_name:
            self.feature_cache.pop(property_name, None)
            self.spatial_indexes.pop(property_name, None)
        else:
            self.feature_cache.clear()
            self.spatial_indexes.clear()
        
        logger.info(f"Cleared cache for {property_name or 'all properties'}")


class IntelligentNodeSampler:
    """Smart node sampling strategies to reduce computation without losing quality"""
    
    def __init__(self):
        self.sampling_strategies = [
            'intersection_priority',
            'spatial_stratification', 
            'semantic_guided',
            'distance_weighted'
        ]
    
    def sample_nodes_intelligently(self, graph, center_lat: float, center_lon: float,
                                 radius_m: float, target_count: int = 1000,
                                 strategy: str = 'hybrid') -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        Intelligent node sampling using multiple strategies.
        
        Returns:
            Tuple of (locations, node_ids) for sampled nodes
        """
        start_time = time.time()
        
        # Get all nodes in radius first
        all_nodes = []
        for node_id, data in graph.nodes(data=True):
            lat, lon = data['y'], data['x']
            distance = haversine_cached(center_lat, center_lon, lat, lon)
            if distance <= radius_m:
                all_nodes.append((node_id, lat, lon, data, distance))
        
        logger.info(f"Found {len(all_nodes)} nodes in {radius_m}m radius")
        
        if len(all_nodes) <= target_count:
            locations = [(lat, lon) for _, lat, lon, _, _ in all_nodes]
            node_ids = [node_id for node_id, _, _, _, _ in all_nodes]
            return locations, node_ids
        
        if strategy == 'hybrid':
            sampled_nodes = self._hybrid_sampling(graph, all_nodes, target_count)
        elif strategy == 'intersection_priority':
            sampled_nodes = self._intersection_priority_sampling(graph, all_nodes, target_count)
        elif strategy == 'spatial_stratification':
            sampled_nodes = self._spatial_stratified_sampling(all_nodes, center_lat, center_lon, target_count)
        else:
            # Fallback to simple random sampling
            import random
            sampled_nodes = random.sample(all_nodes, target_count)
        
        locations = [(lat, lon) for _, lat, lon, _, _ in sampled_nodes]
        node_ids = [node_id for node_id, _, _, _, _ in sampled_nodes]
        
        elapsed = time.time() - start_time
        logger.info(f"Sampled {len(sampled_nodes)} nodes using {strategy} strategy in {elapsed:.3f}s")
        
        return locations, node_ids
    
    def _hybrid_sampling(self, graph, all_nodes: List[Tuple], target_count: int) -> List[Tuple]:
        """Hybrid sampling combining multiple strategies"""
        sampled = []
        
        # 1. High-priority intersections (30% of budget)
        intersection_budget = int(target_count * 0.3)
        intersections = [
            node_tuple for node_tuple in all_nodes 
            if graph.degree(node_tuple[0]) >= 3
        ]
        
        if intersections:
            import random
            intersection_sample = random.sample(
                intersections, 
                min(intersection_budget, len(intersections))
            )
            sampled.extend(intersection_sample)
        
        # 2. Spatial stratification (50% of budget)
        remaining_budget = target_count - len(sampled)
        stratified_budget = int(remaining_budget * 0.7)
        
        # Remove already sampled nodes
        remaining_nodes = [n for n in all_nodes if n not in sampled]
        if remaining_nodes:
            stratified_sample = self._spatial_stratified_sampling(
                remaining_nodes, 
                all_nodes[0][1], all_nodes[0][2],  # Use first node's coordinates as center approximation
                stratified_budget
            )
            sampled.extend(stratified_sample)
        
        # 3. Fill remainder with random sampling
        final_budget = target_count - len(sampled)
        if final_budget > 0:
            remaining_nodes = [n for n in all_nodes if n not in sampled]
            if remaining_nodes:
                import random
                random_sample = random.sample(
                    remaining_nodes,
                    min(final_budget, len(remaining_nodes))
                )
                sampled.extend(random_sample)
        
        return sampled[:target_count]
    
    def _intersection_priority_sampling(self, graph, all_nodes: List[Tuple], target_count: int) -> List[Tuple]:
        """Prioritize intersections and connectivity hubs"""
        # Sort by degree (connectivity)
        nodes_by_degree = sorted(
            all_nodes, 
            key=lambda x: graph.degree(x[0]), 
            reverse=True
        )
        
        return nodes_by_degree[:target_count]
    
    def _spatial_stratified_sampling(self, all_nodes: List[Tuple], 
                                   center_lat: float, center_lon: float, 
                                   target_count: int) -> List[Tuple]:
        """Sample nodes using spatial stratification"""
        # Create spatial grid
        grid_size = 8  # 8x8 grid
        samples_per_cell = max(1, target_count // (grid_size * grid_size))
        
        # Find bounds
        lats = [lat for _, lat, _, _, _ in all_nodes]
        lons = [lon for _, lon, _, _, _ in all_nodes]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        
        # Assign nodes to grid cells
        grid_cells = defaultdict(list)
        for node_tuple in all_nodes:
            _, lat, lon, _, _ = node_tuple
            
            grid_lat = int((lat - min_lat) / lat_step) if lat_step > 0 else 0
            grid_lon = int((lon - min_lon) / lon_step) if lon_step > 0 else 0
            
            grid_lat = min(grid_lat, grid_size - 1)
            grid_lon = min(grid_lon, grid_size - 1)
            
            grid_cells[(grid_lat, grid_lon)].append(node_tuple)
        
        # Sample from each cell
        sampled = []
        import random
        
        for cell_nodes in grid_cells.values():
            if cell_nodes:
                sample_size = min(samples_per_cell, len(cell_nodes))
                cell_sample = random.sample(cell_nodes, sample_size)
                sampled.extend(cell_sample)
        
        # If we're under target, fill with remaining nodes
        if len(sampled) < target_count:
            remaining = [n for n in all_nodes if n not in sampled]
            additional = random.sample(
                remaining, 
                min(target_count - len(sampled), len(remaining))
            )
            sampled.extend(additional)
        
        return sampled[:target_count]


def create_optimized_semantic_scorer() -> OptimizedSemanticScorer:
    """Create optimized semantic scorer with default properties"""
    scorer = OptimizedSemanticScorer(enable_vectorization=True)
    
    # Register optimized forest property
    scorer.register_property(
        'forests',
        OptimizedSemanticProperty(
            name='forests',
            description='Proximity to forests, parks, and wooded areas',
            proximity_threshold=500.0,
            scoring_weight=1.0,
            falloff_function='exponential',
            early_termination_distance=50.0,
            max_search_distance=1500.0
        )
    )
    
    # Register optimized rivers property
    scorer.register_property(
        'rivers',
        OptimizedSemanticProperty(
            name='rivers',
            description='Proximity to rivers, streams, and waterways',
            proximity_threshold=200.0,
            scoring_weight=1.0,
            falloff_function='exponential',
            early_termination_distance=25.0,
            max_search_distance=800.0
        )
    )
    
    # Register optimized lakes property
    scorer.register_property(
        'lakes',
        OptimizedSemanticProperty(
            name='lakes',
            description='Proximity to lakes, ponds, and water bodies',
            proximity_threshold=300.0,
            scoring_weight=1.0,
            falloff_function='exponential',
            early_termination_distance=30.0,
            max_search_distance=1000.0
        )
    )
    
    logger.info("Created optimized semantic scorer with high-performance features")
    return scorer