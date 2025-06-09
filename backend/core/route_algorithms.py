import math
import numpy as np
import osmnx as ox
import networkx as nx
from haversine import haversine, Unit
from typing import List, Tuple, Optional, Dict, Any
import random
from shapely.geometry import LineString
import geopandas as gpd
from dataclasses import dataclass

# Optional rasterio import for elevation handling
try:
    import rasterio
    from rasterio.sample import sample_gen
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    rasterio = None


@dataclass
class RouteConfig:
    """Configuration for route planning algorithms."""
    target_distance: int = 8000  # 8km for 10k steps
    tolerance: int = 1000        # 1km tolerance
    min_elevation_gain: Optional[int] = None
    avoid_roads: bool = True
    max_complexity: int = 1000   # Max nodes for pathfinding
    resolution_factor: float = 0.7  # Fraction of edges to keep in resolution reduction


class MapLoader:
    """Handles loading and processing of OpenStreetMap data."""
    
    @staticmethod
    def load_map(point: Tuple[float, float], dist: int = 3500) -> Tuple[nx.MultiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Load OSM trail data for a given region.
        Args:
            point: User location (lat, long) used as center for local map
            dist: Distance in meters for bounding box
        Returns: The dataset as a NetworkX MultiGraph, the nodes geojson, the edges geojson
        """
        # Custom filter to only include walkable segments
        custom_filter = '["highway"~"path|track|footway|steps|bridleway|cycleway"]'
        
        # Load graph with walking network
        G = ox.graph_from_point(
            point, 
            dist=dist, 
            network_type='walk',
            custom_filter=custom_filter if True else None  # Use avoid_roads config
        )

        # Convert Graph to GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(G)
        edges = edges[["name", "geometry", "length"]]
        edges["name"] = edges["name"].apply(lambda x: x[0] if isinstance(x, list) else x)

        return G, nodes, edges


class TileValueFunction:
    """Legacy tile-based value function - use SpatialValueFunction for advanced features."""
    
    def __init__(self, bounds: Dict[str, float], zoom: int = 15):
        """
        Initialize tile-based value function.
        Args:
            bounds: Dict with keys 'minx', 'maxx', 'miny', 'maxy'
            zoom: Tile zoom level
        """
        self.zoom = zoom
        self.bounds = bounds
        
        # Calculate tile boundaries
        point_a = self._deg2num(bounds['miny'], bounds['minx'], zoom)
        point_b = self._deg2num(bounds['maxy'], bounds['maxx'], zoom)
        
        self.min_x = min(point_b[0], point_a[0])
        self.min_y = min(point_b[1], point_a[1])
        self.n_tiles_x = abs(point_b[0] - point_a[0])
        self.n_tiles_y = abs(point_b[1] - point_a[1])
        
        # Initialize random value function
        self.value = np.random.uniform(0, 1, (self.n_tiles_x, self.n_tiles_y)) * 0.3
    
    def _deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile numbers."""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)
    
    def _get_idx(self, lat: float, lon: float) -> Tuple[int, int]:
        """Map lat/lon to tile index."""
        x, y = self._deg2num(lat, lon, self.zoom)
        return int((x - self.min_x) % self.n_tiles_x), int((y - self.min_y) % self.n_tiles_y)
    
    def set_value(self, lat: float, lon: float, value: float):
        """Set value at specific location."""
        idx = self._get_idx(lat, lon)
        if 0 <= idx[0] < self.n_tiles_x and 0 <= idx[1] < self.n_tiles_y:
            self.value[idx] = value
    
    def get_value(self, lat: float, lon: float) -> float:
        """Get value at specific location."""
        idx = self._get_idx(lat, lon)
        if 0 <= idx[0] < self.n_tiles_x and 0 <= idx[1] < self.n_tiles_y:
            return self.value[idx]
        return 0.5  # Default neutral value


class ElevationHandler:
    """Handles elevation data processing."""
    
    @staticmethod
    def get_elevation_profile(dataset, coords: List[List[float]]) -> Tuple[List[float], List[float]]:
        """
        Get the elevation profile of a path segment.
        Args:
            dataset: The opened rasterio dataset for SRTM data
            coords: Path coordinates in [[lon, lat], [lon, lat], ...] format
        Returns: Distance (in m) and elevation (in m) vectors
        """
        if not RASTERIO_AVAILABLE:
            # Return flat elevation profile if rasterio not available
            distances = [i * 100 for i in range(len(coords))]  # Dummy distances
            elevations = [100.0] * len(coords)  # Flat 100m elevation
            return distances, elevations
        
        # Coordinates are [lon, lat], flip for rasterio
        coords_flipped = [[c[1], c[0]] for c in coords]
        
        # Query elevation for each point
        elevations = [e[0] for e in sample_gen(dataset, coords_flipped)]
        
        # Calculate cumulative distances
        distances = [0.0]
        for j in range(len(coords) - 1):
            dist = haversine(
                (coords[j][1], coords[j][0]), 
                (coords[j + 1][1], coords[j + 1][0]), 
                Unit.METERS
            )
            distances.append(distances[j] + dist)
        
        return distances, elevations
    
    @staticmethod
    def calculate_elevation_gain(elevations: List[float], threshold: float = 70) -> float:
        """Calculate total elevation gain with smoothing threshold."""
        last_pt = elevations[0]
        elev_gain = 0
        
        for elevation in elevations:
            if abs(elevation - last_pt) >= threshold:
                if elevation - last_pt > 0:
                    elev_gain += (elevation - last_pt)
                last_pt = elevation
        
        return elev_gain


class ResolutionReducer:
    """Handles complexity reduction through resolution methods - Legacy wrapper."""
    
    @staticmethod
    def reduce_graph_complexity(G: nx.MultiGraph, resolution_factor: float = 0.7) -> nx.MultiGraph:
        """
        Legacy method for backward compatibility.
        Use ResolutionManager from resolution_strategies for advanced reduction.
        """
        from core.resolution_strategies import ResolutionManager, ResolutionConfig, ResolutionStrategy
        
        config = ResolutionConfig(
            strategy=ResolutionStrategy.ADAPTIVE,
            reduction_factor=resolution_factor,
            preserve_connectivity=True
        )
        
        manager = ResolutionManager(config)
        return manager.reduce_complexity(G)


class RouteOptimizer:
    """Handles route optimization and pathfinding."""
    
    def __init__(self, config: RouteConfig):
        self.config = config
    
    def path_to_coords_list(self, G: nx.Graph, path: List) -> List[List[float]]:
        """Convert path to coordinate list."""
        coords_list = []
        for u, v in zip(path[:-1], path[1:]):
            data = G.get_edge_data(u, v)
            if data and "geometry" in data:
                # If geometry attribute exists, add all its coords
                xs, ys = data["geometry"].xy
                pts = [[ys[i], xs[i]] for i in range(len(xs))]
                coords_list += pts
            else:
                # Otherwise, straight line from node to node
                coords_list.append([G.nodes[u]["y"], G.nodes[u]["x"]])
        return coords_list
    
    def calculate_path_length(self, G: nx.Graph, path: List) -> float:
        """Calculate total path length."""
        total_dist = 0
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i + 1])
            if edge_data:
                total_dist += edge_data.get("length", 0)
        return total_dist
    
    def find_optimal_route(
        self, 
        G: nx.Graph, 
        start_node: int, 
        value_function = None,
        elevation_dataset = None
    ) -> List[int]:
        """
        Find optimal route using improved heuristic algorithm.
        Args:
            G: Graph to search
            start_node: Starting node ID
            value_function: Optional spatial value function
            elevation_dataset: Optional elevation data
        Returns: List of node IDs forming the route
        """
        target_dist = self.config.target_distance
        tolerance = self.config.tolerance
        
        # First, reduce complexity if graph is too dense using advanced resolution strategies
        if len(G.nodes()) > self.config.max_complexity:
            from core.resolution_strategies import ResolutionManager, ResolutionConfig, ResolutionStrategy
            
            resolution_config = ResolutionConfig(
                strategy=ResolutionStrategy.ADAPTIVE,
                target_complexity=self.config.max_complexity,
                reduction_factor=self.config.resolution_factor,
                preserve_connectivity=True
            )
            
            resolution_manager = ResolutionManager(resolution_config)
            G = resolution_manager.reduce_complexity(G)
        
        # Use advanced optimization if value function is available
        from core.value_function import SpatialValueFunction
        if isinstance(value_function, SpatialValueFunction):
            from core.advanced_optimizer import AdvancedRouteOptimizer, OptimizationConfig
            
            opt_config = OptimizationConfig(
                max_iterations=200,  # Reduce for real-time performance
                population_size=20,
                value_weight=0.5,
                distance_weight=0.4,
                diversity_weight=0.1
            )
            
            advanced_optimizer = AdvancedRouteOptimizer(self.config, opt_config)
            return advanced_optimizer.optimize_route(
                G, start_node, target_dist, value_function, method='hybrid'
            )
        
        # Find anchor points (1/3 of target distance)
        anchors = self._find_anchor_points(G, start_node, target_dist // 3, tolerance // 3)
        
        if not anchors:
            # Fallback to simple circular route
            return self._find_simple_circular_route(G, start_node, target_dist, tolerance)
        
        best_route = []
        best_error = float('inf')
        
        # Try multiple anchor points
        for anchor in list(anchors)[:10]:  # Limit to prevent excessive computation
            # Find dual points from this anchor
            duals = self._find_anchor_points(G, anchor, target_dist // 3, tolerance // 3)
            
            if not duals:
                continue
            
            # Try paths through dual points
            for dual in list(duals)[:5]:  # Limit dual points too
                try:
                    # Create route: start -> anchor -> dual -> start
                    route_parts = [
                        nx.shortest_path(G, start_node, anchor, weight="length"),
                        nx.shortest_path(G, anchor, dual, weight="length")[1:],
                        nx.shortest_path(G, dual, start_node, weight="length")[1:]
                    ]
                    
                    # Combine route parts
                    full_route = []
                    for part in route_parts:
                        full_route.extend(part)
                    
                    # Calculate route metrics
                    route_length = self.calculate_path_length(G, full_route)
                    distance_error = abs(route_length - target_dist)
                    
                    # Calculate diversity score
                    diversity = len(set(full_route)) / len(full_route)
                    
                    # Combined error metric
                    total_error = distance_error / diversity
                    
                    # Check elevation constraint if specified
                    if self.config.min_elevation_gain and elevation_dataset:
                        coords = self.path_to_coords_list(G, full_route)
                        _, elevations = ElevationHandler.get_elevation_profile(elevation_dataset, coords)
                        elev_gain = ElevationHandler.calculate_elevation_gain(elevations)
                        
                        if elev_gain < self.config.min_elevation_gain:
                            continue
                    
                    # Update best route if this is better
                    if total_error < best_error:
                        best_error = total_error
                        best_route = full_route
                        
                        # Early termination if good enough
                        if distance_error < tolerance:
                            break
                            
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        return best_route if best_route else self._find_simple_circular_route(G, start_node, target_dist, tolerance)
    
    def _find_anchor_points(self, G: nx.Graph, start_node: int, target_dist: float, tolerance: float) -> set:
        """Find potential anchor points at approximately target distance."""
        anchors = set()
        
        try:
            # Use single-source shortest path to find nodes at target distance
            distances = nx.single_source_dijkstra_path_length(G, start_node, cutoff=target_dist + tolerance, weight="length")
            
            for node, dist in distances.items():
                if target_dist - tolerance <= dist <= target_dist + tolerance:
                    anchors.add(node)
        except Exception:
            pass
        
        return anchors
    
    def _find_simple_circular_route(self, G: nx.Graph, start_node: int, target_dist: float, tolerance: float) -> List[int]:
        """Fallback method for simple circular route."""
        # Find the farthest reachable node
        try:
            distances = nx.single_source_dijkstra_path_length(G, start_node, weight="length")
            
            # Find node closest to half target distance
            half_target = target_dist / 2
            best_node = start_node
            best_diff = float('inf')
            
            for node, dist in distances.items():
                diff = abs(dist - half_target)
                if diff < best_diff:
                    best_diff = diff
                    best_node = node
            
            # Create route: start -> best_node -> start
            if best_node != start_node:
                path_out = nx.shortest_path(G, start_node, best_node, weight="length")
                path_back = nx.shortest_path(G, best_node, start_node, weight="length")[1:]
                return path_out + path_back
        except Exception:
            pass
        
        # Ultimate fallback: just return start node
        return [start_node]


class RouteExporter:
    """Handles route export functionality."""
    
    @staticmethod
    def export_to_gpx(G: nx.Graph, path: List[int]) -> str:
        """Export route to GPX format."""
        optimizer = RouteOptimizer(RouteConfig())
        coords = optimizer.path_to_coords_list(G, path)
        
        # Convert to proper coordinate format for GPX
        coords_lonlat = [(c[1], c[0]) for c in coords]
        
        # Create LineString geometry
        geom = LineString(coords_lonlat)
        
        # Create GeoDataFrame and export to GPX-like format
        gdf = gpd.GeoDataFrame([{"geometry": geom}])
        
        # For now, return a simple GPX string (could be enhanced with proper GPX library)
        gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Perfect10k">
  <trk>
    <name>Perfect10k Route</name>
    <trkseg>
"""
        
        for coord in coords:
            gpx_content += f'      <trkpt lat="{coord[0]}" lon="{coord[1]}"></trkpt>\n'
        
        gpx_content += """    </trkseg>
  </trk>
</gpx>"""
        
        return gpx_content