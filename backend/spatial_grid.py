"""
Simple Grid-Based Spatial Indexing for Perfect10k
Provides O(1) locality queries for efficient candidate retrieval.
"""

import math
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from loguru import logger


@dataclass
class GridCell:
    """A single cell in the spatial grid."""
    grid_x: int
    grid_y: int
    center_lat: float
    center_lon: float
    nodes: Set[int]
    
    def __post_init__(self):
        if not isinstance(self.nodes, set):
            self.nodes = set(self.nodes) if self.nodes else set()


@dataclass  
class SpatialBounds:
    """Spatial bounds for a geographic area."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


class SpatialGrid:
    """
    Simple grid-based spatial indexing for fast locality queries.
    
    Divides geographic space into uniform grid cells (~200m x 200m)
    for O(1) lookups and efficient neighbor queries.
    """
    
    def __init__(self, cell_size_meters: float = 200.0):
        """
        Initialize spatial grid.
        
        Args:
            cell_size_meters: Size of each grid cell in meters (~200m works well)
        """
        self.cell_size_meters = cell_size_meters
        self.grid: Dict[Tuple[int, int], GridCell] = {}
        self.bounds: Optional[SpatialBounds] = None
        
        # Derived parameters (will be set when first data is added)
        self.lat_per_meter = 1.0 / 111000.0  # Rough: 1 degree ≈ 111km
        self.lon_per_meter = None  # Will be calculated based on latitude
        self.grid_lat_size = None  # Grid cell size in latitude degrees
        self.grid_lon_size = None  # Grid cell size in longitude degrees
        
        logger.info(f"Initialized spatial grid with {cell_size_meters}m cell size")
    
    def add_nodes_from_graph(self, graph: nx.MultiGraph) -> None:
        """
        Populate the spatial grid from a NetworkX graph.
        
        Args:
            graph: NetworkX graph with nodes containing 'x' (lon) and 'y' (lat) attributes
        """
        if not graph.nodes:
            logger.warning("Cannot build spatial grid from empty graph")
            return
        
        # Calculate bounds from all nodes
        lats = [data['y'] for _, data in graph.nodes(data=True)]
        lons = [data['x'] for _, data in graph.nodes(data=True)]
        
        self.bounds = SpatialBounds(
            min_lat=min(lats), max_lat=max(lats),
            min_lon=min(lons), max_lon=max(lons)
        )
        
        # Calculate grid cell sizes based on center latitude
        center_lat = (self.bounds.min_lat + self.bounds.max_lat) / 2.0
        self.lon_per_meter = 1.0 / (111000.0 * math.cos(math.radians(center_lat)))
        
        self.grid_lat_size = self.cell_size_meters * self.lat_per_meter
        self.grid_lon_size = self.cell_size_meters * self.lon_per_meter
        
        logger.info(f"Grid parameters: lat_size={self.grid_lat_size:.6f}°, lon_size={self.grid_lon_size:.6f}°")
        
        # Add all nodes to grid
        nodes_added = 0
        for node_id, data in graph.nodes(data=True):
            lat, lon = data['y'], data['x']
            self.add_node(node_id, lat, lon)
            nodes_added += 1
        
        logger.info(f"Added {nodes_added} nodes to spatial grid across {len(self.grid)} cells")
        logger.info(f"Grid coverage: {self.bounds.min_lat:.4f} to {self.bounds.max_lat:.4f} lat, "
                   f"{self.bounds.min_lon:.4f} to {self.bounds.max_lon:.4f} lon")
    
    def add_node(self, node_id: int, lat: float, lon: float) -> Tuple[int, int]:
        """
        Add a node to the spatial grid.
        
        Args:
            node_id: Unique node identifier
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            Grid coordinates (grid_x, grid_y) where node was placed
        """
        if self.grid_lat_size is None or self.grid_lon_size is None:
            raise ValueError("Grid not initialized. Call add_nodes_from_graph() first.")
        
        # Calculate grid coordinates
        grid_x, grid_y = self._lat_lon_to_grid(lat, lon)
        
        # Create cell if it doesn't exist
        if (grid_x, grid_y) not in self.grid:
            cell_center_lat, cell_center_lon = self._grid_to_lat_lon(grid_x, grid_y)
            self.grid[(grid_x, grid_y)] = GridCell(
                grid_x=grid_x,
                grid_y=grid_y, 
                center_lat=cell_center_lat,
                center_lon=cell_center_lon,
                nodes=set()
            )
        
        # Add node to cell
        self.grid[(grid_x, grid_y)].nodes.add(node_id)
        return grid_x, grid_y
    
    def get_nearby_nodes(self, lat: float, lon: float, radius_meters: float) -> List[int]:
        """
        Get all nodes within a radius of a location.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_meters: Search radius in meters
            
        Returns:
            List of node IDs within the radius
        """
        if not self.grid:
            return []
        
        # Calculate how many grid cells to search
        radius_cells = math.ceil(radius_meters / self.cell_size_meters)
        center_grid_x, center_grid_y = self._lat_lon_to_grid(lat, lon)
        
        nearby_nodes = []
        
        # Search in a square around the center
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy
                
                if (grid_x, grid_y) in self.grid:
                    cell = self.grid[(grid_x, grid_y)]
                    
                    # Quick distance check using cell center
                    cell_distance = self._haversine_distance(
                        lat, lon, cell.center_lat, cell.center_lon
                    )
                    
                    # Include cell if center is within radius + cell diagonal
                    cell_diagonal = self.cell_size_meters * math.sqrt(2)
                    if cell_distance <= radius_meters + cell_diagonal:
                        nearby_nodes.extend(cell.nodes)
        
        return nearby_nodes
    
    def get_cell_at_location(self, lat: float, lon: float) -> Optional[GridCell]:
        """
        Get the grid cell containing a specific location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            GridCell if exists, None otherwise
        """
        if not self.grid:
            return None
        
        grid_x, grid_y = self._lat_lon_to_grid(lat, lon)
        return self.grid.get((grid_x, grid_y))
    
    def get_neighboring_cells(self, lat: float, lon: float, radius_cells: int = 1) -> List[GridCell]:
        """
        Get neighboring grid cells around a location.
        
        Args:
            lat: Center latitude
            lon: Center longitude  
            radius_cells: How many cells in each direction to include
            
        Returns:
            List of neighboring GridCells
        """
        if not self.grid:
            return []
        
        center_grid_x, center_grid_y = self._lat_lon_to_grid(lat, lon)
        neighboring_cells = []
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy
                
                if (grid_x, grid_y) in self.grid:
                    neighboring_cells.append(self.grid[(grid_x, grid_y)])
        
        return neighboring_cells
    
    def _lat_lon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon coordinates to grid coordinates."""
        if self.bounds is None:
            raise ValueError("Grid bounds not set")
        
        # Normalize to grid origin
        norm_lat = lat - self.bounds.min_lat
        norm_lon = lon - self.bounds.min_lon
        
        # Convert to grid coordinates
        grid_x = int(norm_lon / self.grid_lon_size)
        grid_y = int(norm_lat / self.grid_lat_size)
        
        return grid_x, grid_y
    
    def _grid_to_lat_lon(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to lat/lon (cell center)."""
        if self.bounds is None:
            raise ValueError("Grid bounds not set")
        
        # Calculate cell center
        lat = self.bounds.min_lat + (grid_y + 0.5) * self.grid_lat_size
        lon = self.bounds.min_lon + (grid_x + 0.5) * self.grid_lon_size
        
        return lat, lon
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters."""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat_rad = math.radians(lat2 - lat1)
        dlon_rad = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat_rad/2) * math.sin(dlat_rad/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon_rad/2) * math.sin(dlon_rad/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def get_statistics(self) -> Dict:
        """Get statistics about the spatial grid."""
        if not self.grid:
            return {"error": "Grid is empty"}
        
        # Calculate statistics
        total_nodes = sum(len(cell.nodes) for cell in self.grid.values())
        occupied_cells = len(self.grid)
        nodes_per_cell = [len(cell.nodes) for cell in self.grid.values()]
        
        stats = {
            "total_nodes": total_nodes,
            "occupied_cells": occupied_cells,
            "avg_nodes_per_cell": total_nodes / occupied_cells if occupied_cells > 0 else 0,
            "max_nodes_per_cell": max(nodes_per_cell) if nodes_per_cell else 0,
            "min_nodes_per_cell": min(nodes_per_cell) if nodes_per_cell else 0,
            "cell_size_meters": self.cell_size_meters,
            "grid_coverage": {
                "lat_range": f"{self.bounds.min_lat:.4f} to {self.bounds.max_lat:.4f}",
                "lon_range": f"{self.bounds.min_lon:.4f} to {self.bounds.max_lon:.4f}",
                "total_area_km2": self._calculate_coverage_area()
            }
        }
        
        return stats
    
    def _calculate_coverage_area(self) -> float:
        """Calculate approximate coverage area in km²."""
        if not self.bounds:
            return 0.0
        
        # Rough calculation using haversine for the bounds
        lat_distance = self._haversine_distance(
            self.bounds.min_lat, self.bounds.min_lon,
            self.bounds.max_lat, self.bounds.min_lon
        )
        
        lon_distance = self._haversine_distance(
            self.bounds.min_lat, self.bounds.min_lon,
            self.bounds.min_lat, self.bounds.max_lon
        )
        
        area_m2 = lat_distance * lon_distance
        return area_m2 / 1_000_000  # Convert to km²
    
    def clear(self):
        """Clear all data from the spatial grid."""
        self.grid.clear()
        self.bounds = None
        self.grid_lat_size = None
        self.grid_lon_size = None
        logger.info("Cleared spatial grid")