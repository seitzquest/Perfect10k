import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValueConfig:
    """Configuration for value function estimation."""
    spatial_resolution: float = 100.0  # meters per tile
    preference_weight: float = 0.4      # Weight for user preferences
    elevation_weight: float = 0.2       # Weight for elevation preferences
    accessibility_weight: float = 0.3   # Weight for path accessibility
    diversity_weight: float = 0.1       # Weight for route diversity
    smoothing_sigma: float = 1.0        # Gaussian smoothing parameter
    decay_distance: float = 500.0       # Distance decay for place influence (meters)


class SpatialValueFunction:
    """
    Advanced spatial value function that incorporates user preferences,
    elevation data, accessibility, and route diversity.
    """
    
    def __init__(self, bounds: Dict[str, float], config: ValueConfig = None):
        """
        Initialize spatial value function.
        
        Args:
            bounds: Geographic bounds {'minx', 'maxx', 'miny', 'maxy'}
            config: Value function configuration
        """
        self.bounds = bounds
        self.config = config or ValueConfig()
        
        # Calculate grid dimensions
        self._setup_spatial_grid()
        
        # Initialize value components
        self.base_values = np.ones(self.grid_shape) * 0.5  # Neutral base
        self.preference_values = np.zeros(self.grid_shape)
        self.elevation_values = np.zeros(self.grid_shape)
        self.accessibility_values = np.ones(self.grid_shape)
        self.diversity_values = np.ones(self.grid_shape)
        
        # Combined value function
        self.combined_values = None
        
    def _setup_spatial_grid(self):
        """Setup the spatial grid based on bounds and resolution."""
        # Calculate grid dimensions in meters
        width_meters = self._deg_to_meters(
            self.bounds['maxx'] - self.bounds['minx'], 
            (self.bounds['miny'] + self.bounds['maxy']) / 2
        )
        height_meters = self._deg_to_meters(
            self.bounds['maxy'] - self.bounds['miny'], 
            (self.bounds['miny'] + self.bounds['maxy']) / 2
        )
        
        # Grid dimensions
        self.grid_width = int(width_meters / self.config.spatial_resolution)
        self.grid_height = int(height_meters / self.config.spatial_resolution)
        self.grid_shape = (self.grid_height, self.grid_width)
        
        # Create coordinate mapping
        self.lat_coords = np.linspace(self.bounds['miny'], self.bounds['maxy'], self.grid_height)
        self.lon_coords = np.linspace(self.bounds['minx'], self.bounds['maxx'], self.grid_width)
        
        logger.info(f"Spatial grid initialized: {self.grid_shape} ({width_meters:.0f}x{height_meters:.0f}m)")
    
    def _deg_to_meters(self, degrees: float, latitude: float) -> float:
        """Convert degrees to meters at given latitude."""
        # Approximate conversion (more accurate methods exist)
        lat_rad = math.radians(latitude)
        meters_per_degree_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        meters_per_degree_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
        
        # Use average for simplicity
        return degrees * (meters_per_degree_lat + meters_per_degree_lon) / 2
    
    def _coords_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon coordinates to grid indices."""
        if not (self.bounds['miny'] <= lat <= self.bounds['maxy'] and 
                self.bounds['minx'] <= lon <= self.bounds['maxx']):
            return None, None
            
        row = int((lat - self.bounds['miny']) / (self.bounds['maxy'] - self.bounds['miny']) * self.grid_height)
        col = int((lon - self.bounds['minx']) / (self.bounds['maxx'] - self.bounds['minx']) * self.grid_width)
        
        # Clamp to grid bounds
        row = max(0, min(self.grid_height - 1, row))
        col = max(0, min(self.grid_width - 1, col))
        
        return row, col
    
    def _grid_to_coords(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to lat/lon coordinates."""
        lat = self.bounds['miny'] + (row / self.grid_height) * (self.bounds['maxy'] - self.bounds['miny'])
        lon = self.bounds['minx'] + (col / self.grid_width) * (self.bounds['maxx'] - self.bounds['minx'])
        return lat, lon
    
    def add_place_preferences(self, places: List[Dict[str, Any]], similarity_scores: List[float]):
        """
        Add user preference influence from matched places.
        
        Args:
            places: List of place dictionaries with 'latitude', 'longitude'
            similarity_scores: Similarity scores for each place (0-1)
        """
        if not places or not similarity_scores:
            return
            
        for place, similarity in zip(places, similarity_scores):
            if similarity <= 0:
                continue
                
            lat, lon = place['latitude'], place['longitude']
            center_row, center_col = self._coords_to_grid(lat, lon)
            
            if center_row is None or center_col is None:
                continue
            
            # Calculate influence radius in grid cells
            influence_radius_cells = int(self.config.decay_distance / self.config.spatial_resolution)
            
            # Apply Gaussian influence around the place
            self._add_gaussian_influence(
                self.preference_values,
                center_row, center_col,
                influence_radius_cells,
                similarity
            )
    
    def add_elevation_preferences(self, elevation_data: np.ndarray, preference_type: str = 'moderate'):
        """
        Add elevation preferences to the value function.
        
        Args:
            elevation_data: 2D array of elevation values matching grid
            preference_type: 'flat', 'moderate', 'hilly'
        """
        if elevation_data.shape != self.grid_shape:
            # Resize elevation data to match grid
            from scipy.ndimage import zoom
            scale_factors = (
                self.grid_shape[0] / elevation_data.shape[0],
                self.grid_shape[1] / elevation_data.shape[1]
            )
            elevation_data = zoom(elevation_data, scale_factors, order=1)
        
        # Calculate elevation gradients
        grad_y, grad_x = np.gradient(elevation_data)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        if preference_type == 'flat':
            # Prefer flat areas (low slope)
            self.elevation_values = 1.0 - np.clip(slope / np.max(slope), 0, 1)
        elif preference_type == 'moderate':
            # Prefer moderate slopes
            normalized_slope = slope / np.max(slope)
            self.elevation_values = 1.0 - np.abs(normalized_slope - 0.3) / 0.7
        elif preference_type == 'hilly':
            # Prefer hillier areas
            self.elevation_values = np.clip(slope / np.max(slope), 0, 1)
        
        # Apply smoothing
        self.elevation_values = gaussian_filter(self.elevation_values, self.config.smoothing_sigma)
    
    def add_accessibility_constraints(self, graph: nx.Graph):
        """
        Add accessibility constraints based on path density and connectivity.
        
        Args:
            graph: NetworkX graph of walkable paths
        """
        # Initialize accessibility grid
        self.accessibility_values = np.zeros(self.grid_shape)
        
        # Count nodes in each grid cell
        node_density = np.zeros(self.grid_shape)
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if 'y' in node_data and 'x' in node_data:
                lat, lon = node_data['y'], node_data['x']
                row, col = self._coords_to_grid(lat, lon)
                
                if row is not None and col is not None:
                    node_density[row, col] += 1
        
        # Convert density to accessibility values
        if np.max(node_density) > 0:
            # Normalize and apply non-linear scaling
            normalized_density = node_density / np.max(node_density)
            self.accessibility_values = np.sqrt(normalized_density)  # Square root for gentler scaling
        
        # Smooth accessibility values
        self.accessibility_values = gaussian_filter(self.accessibility_values, self.config.smoothing_sigma)
        
        # Ensure minimum accessibility (avoid completely inaccessible areas)
        self.accessibility_values = np.maximum(self.accessibility_values, 0.1)
    
    def add_diversity_bonus(self, previous_routes: List[List[Tuple[float, float]]], decay_factor: float = 0.7):
        """
        Add diversity bonus to encourage route variation.
        
        Args:
            previous_routes: List of previous route coordinates
            decay_factor: How much to penalize previously used areas
        """
        if not previous_routes:
            return
        
        # Initialize diversity values (start with full diversity)
        self.diversity_values = np.ones(self.grid_shape)
        
        for route in previous_routes:
            for lat, lon in route:
                row, col = self._coords_to_grid(lat, lon)
                
                if row is not None and col is not None:
                    # Apply penalty around this route point
                    influence_radius = int(200 / self.config.spatial_resolution)  # 200m radius
                    self._add_gaussian_influence(
                        self.diversity_values,
                        row, col,
                        influence_radius,
                        -0.3,  # Negative influence (penalty)
                        operation='multiply',
                        base_value=decay_factor
                    )
        
        # Ensure diversity values don't go below minimum
        self.diversity_values = np.maximum(self.diversity_values, 0.2)
    
    def _add_gaussian_influence(
        self, 
        target_array: np.ndarray, 
        center_row: int, 
        center_col: int, 
        radius: int, 
        strength: float,
        operation: str = 'add',
        base_value: float = 0.0
    ):
        """
        Add Gaussian influence to target array around a center point.
        
        Args:
            target_array: Array to modify
            center_row, center_col: Center of influence
            radius: Radius of influence in grid cells
            strength: Strength of influence
            operation: 'add' or 'multiply'
            base_value: Base value for multiply operation
        """
        # Create coordinate grids
        rows, cols = np.ogrid[:target_array.shape[0], :target_array.shape[1]]
        
        # Calculate distance from center
        distance_sq = (rows - center_row)**2 + (cols - center_col)**2
        
        # Create Gaussian kernel
        sigma = radius / 3.0  # 3-sigma rule
        gaussian = np.exp(-distance_sq / (2 * sigma**2))
        
        # Apply mask for radius
        mask = distance_sq <= radius**2
        influence = gaussian * mask * strength
        
        if operation == 'add':
            target_array += influence
        elif operation == 'multiply':
            multiplier = base_value + influence
            target_array *= np.maximum(multiplier, 0.1)  # Prevent negative values
    
    def compute_combined_value(self) -> np.ndarray:
        """
        Compute the combined value function from all components.
        
        Returns:
            Combined value function array
        """
        # Normalize all components to [0, 1]
        norm_preferences = self._normalize_array(self.preference_values)
        norm_elevation = self._normalize_array(self.elevation_values)
        norm_accessibility = self._normalize_array(self.accessibility_values)
        norm_diversity = self._normalize_array(self.diversity_values)
        
        # Weighted combination
        self.combined_values = (
            self.config.preference_weight * norm_preferences +
            self.config.elevation_weight * norm_elevation +
            self.config.accessibility_weight * norm_accessibility +
            self.config.diversity_weight * norm_diversity
        )
        
        # Apply final smoothing
        self.combined_values = gaussian_filter(self.combined_values, self.config.smoothing_sigma)
        
        # Ensure values are in [0, 1] range
        self.combined_values = np.clip(self.combined_values, 0, 1)
        
        return self.combined_values
    
    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        min_val = np.min(array)
        max_val = np.max(array)
        
        if max_val > min_val:
            return (array - min_val) / (max_val - min_val)
        else:
            return np.ones_like(array) * 0.5
    
    def get_value_at_coords(self, lat: float, lon: float) -> float:
        """
        Get value function at specific coordinates.
        
        Args:
            lat, lon: Coordinates
            
        Returns:
            Value at coordinates (0-1)
        """
        if self.combined_values is None:
            self.compute_combined_value()
        
        row, col = self._coords_to_grid(lat, lon)
        
        if row is None or col is None:
            return 0.5  # Neutral value for out-of-bounds
        
        return self.combined_values[row, col]
    
    def get_path_value(self, path_coords: List[Tuple[float, float]]) -> float:
        """
        Calculate average value along a path.
        
        Args:
            path_coords: List of (lat, lon) coordinates along path
            
        Returns:
            Average value along path
        """
        if not path_coords:
            return 0.5
        
        values = []
        for lat, lon in path_coords:
            value = self.get_value_at_coords(lat, lon)
            values.append(value)
        
        return np.mean(values)
    
    def get_gradient_at_coords(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Get value function gradient at coordinates.
        
        Args:
            lat, lon: Coordinates
            
        Returns:
            Gradient (dlat, dlon) pointing toward higher values
        """
        if self.combined_values is None:
            self.compute_combined_value()
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(self.combined_values)
        
        row, col = self._coords_to_grid(lat, lon)
        
        if row is None or col is None:
            return 0.0, 0.0
        
        # Convert grid gradients to coordinate gradients
        dlat = grad_y[row, col] * (self.bounds['maxy'] - self.bounds['miny']) / self.grid_height
        dlon = grad_x[row, col] * (self.bounds['maxx'] - self.bounds['minx']) / self.grid_width
        
        return dlat, dlon
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """
        Export value function data for visualization.
        
        Returns:
            Dictionary with grid data and metadata
        """
        if self.combined_values is None:
            self.compute_combined_value()
        
        return {
            'combined_values': self.combined_values.tolist(),
            'preference_values': self.preference_values.tolist(),
            'elevation_values': self.elevation_values.tolist(),
            'accessibility_values': self.accessibility_values.tolist(),
            'diversity_values': self.diversity_values.tolist(),
            'bounds': self.bounds,
            'grid_shape': self.grid_shape,
            'lat_coords': self.lat_coords.tolist(),
            'lon_coords': self.lon_coords.tolist()
        }


class ValueFunctionOptimizer:
    """
    Optimizer that uses value function to guide route planning.
    """
    
    def __init__(self, value_function: SpatialValueFunction):
        self.value_function = value_function
    
    def evaluate_route(self, route_coords: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Evaluate a route using the value function.
        
        Args:
            route_coords: List of (lat, lon) coordinates
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not route_coords:
            return {'total_value': 0.0, 'avg_value': 0.0, 'value_variance': 0.0}
        
        # Calculate values along route
        values = []
        for lat, lon in route_coords:
            value = self.value_function.get_value_at_coords(lat, lon)
            values.append(value)
        
        values = np.array(values)
        
        return {
            'total_value': np.sum(values),
            'avg_value': np.mean(values),
            'value_variance': np.var(values),
            'min_value': np.min(values),
            'max_value': np.max(values)
        }
    
    def suggest_route_improvements(
        self, 
        route_coords: List[Tuple[float, float]], 
        improvement_radius: float = 200.0
    ) -> List[Dict[str, Any]]:
        """
        Suggest improvements to a route based on value function.
        
        Args:
            route_coords: Current route coordinates
            improvement_radius: Radius to search for improvements (meters)
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        for i, (lat, lon) in enumerate(route_coords):
            current_value = self.value_function.get_value_at_coords(lat, lon)
            
            # Check nearby points for higher values
            best_value = current_value
            best_coords = None
            
            # Sample points in a radius around current point
            radius_deg = improvement_radius / 111000  # Rough conversion to degrees
            
            for dlat in np.linspace(-radius_deg, radius_deg, 5):
                for dlon in np.linspace(-radius_deg, radius_deg, 5):
                    test_lat = lat + dlat
                    test_lon = lon + dlon
                    
                    test_value = self.value_function.get_value_at_coords(test_lat, test_lon)
                    
                    if test_value > best_value:
                        best_value = test_value
                        best_coords = (test_lat, test_lon)
            
            # If we found a better point, suggest it
            if best_coords and best_value > current_value + 0.1:  # Minimum improvement threshold
                suggestions.append({
                    'route_index': i,
                    'current_coords': (lat, lon),
                    'suggested_coords': best_coords,
                    'current_value': current_value,
                    'suggested_value': best_value,
                    'improvement': best_value - current_value
                })
        
        return suggestions