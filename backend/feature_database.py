"""
Discrete Feature Database for Perfect10k
Pre-computes and stores interpretable features for efficient candidate scoring.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx
from loguru import logger
from spatial_grid import GridCell, SpatialGrid


class FeatureType(Enum):
    """Supported discrete feature types."""
    CLOSE_TO_FOREST = "close_to_forest"
    CLOSE_TO_WATER = "close_to_water"
    CLOSE_TO_PARK = "close_to_park"
    PATH_QUALITY = "path_quality"
    INTERSECTION_DENSITY = "intersection_density"
    ELEVATION_VARIETY = "elevation_variety"  # Future: terrain interest


@dataclass
class CellFeatures:
    """Pre-computed features for a grid cell."""
    grid_x: int
    grid_y: int
    features: dict[FeatureType, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    computed_at: float = field(default_factory=time.time)

    def get_feature(self, feature_type: FeatureType, default: float = 0.0) -> float:
        """Get feature value with default fallback."""
        return self.features.get(feature_type, default)

    def set_feature(self, feature_type: FeatureType, value: float):
        """Set feature value (clamped to 0-1 range)."""
        self.features[feature_type] = max(0.0, min(1.0, value))


class FeatureDatabase:
    """
    Pre-computed discrete features per grid cell for efficient scoring.

    Computes interpretable features like proximity to forests, water, parks,
    path quality, and connectivity for each spatial grid cell.
    """

    def __init__(self, spatial_grid: SpatialGrid):
        """
        Initialize feature database.

        Args:
            spatial_grid: Spatial grid for organizing features
        """
        self.spatial_grid = spatial_grid
        self.cell_features: dict[tuple[int, int], CellFeatures] = {}

        # Feature computation parameters
        self.max_search_distance = 1000.0  # Maximum distance to search for features (meters)
        self.path_quality_weights = {
            'path': 1.0,
            'footway': 0.9,
            'track': 0.8,
            'cycleway': 0.7,
            'pedestrian': 0.9,
            'living_street': 0.6,
            'residential': 0.4,
            'tertiary': 0.3,
            'secondary': 0.2,
            'primary': 0.1,
            'default': 0.3
        }

        logger.info("Initialized feature database")

    def compute_features_for_area(self, graph: nx.MultiGraph,
                                semantic_overlay_manager=None) -> int:
        """
        Compute features for all cells in the spatial grid.

        Args:
            graph: NetworkX graph with node and edge data
            semantic_overlay_manager: Optional semantic overlay manager for nature features

        Returns:
            Number of cells processed
        """
        if not self.spatial_grid.grid:
            logger.warning("Spatial grid is empty, cannot compute features")
            return 0

        start_time = time.time()
        cells_processed = 0

        logger.info(f"Computing features for {len(self.spatial_grid.grid)} grid cells")

        # Pre-load semantic data for the entire area if semantic manager is available
        if semantic_overlay_manager:
            logger.info("Pre-loading semantic data for entire area...")
            self._preload_semantic_data(semantic_overlay_manager)

        for _grid_coords, cell in self.spatial_grid.grid.items():
            self._compute_cell_features(cell, graph, semantic_overlay_manager)
            cells_processed += 1

            # Progress logging every 100 cells
            if cells_processed % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Processed {cells_processed}/{len(self.spatial_grid.grid)} cells "
                           f"({elapsed:.1f}s elapsed)")

        elapsed = time.time() - start_time
        logger.info(f"Computed features for {cells_processed} cells in {elapsed:.2f}s "
                   f"({cells_processed/elapsed:.1f} cells/sec)")

        return cells_processed

    def get_cell_features(self, lat: float, lon: float) -> CellFeatures | None:
        """
        Get pre-computed features for a location.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate

        Returns:
            CellFeatures if available, None otherwise
        """
        cell = self.spatial_grid.get_cell_at_location(lat, lon)
        if not cell:
            return None

        return self.cell_features.get((cell.grid_x, cell.grid_y))

    def get_features_for_nodes(self, node_ids: list[int],
                             graph: nx.MultiGraph) -> dict[int, CellFeatures]:
        """
        Get features for a list of nodes.

        Args:
            node_ids: List of node IDs
            graph: NetworkX graph for coordinate lookup

        Returns:
            Dictionary mapping node_id to CellFeatures
        """
        node_features = {}

        for node_id in node_ids:
            if node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                lat, lon = node_data['y'], node_data['x']

                features = self.get_cell_features(lat, lon)
                if features:
                    node_features[node_id] = features

        return node_features

    def _compute_cell_features(self, cell: GridCell, graph: nx.MultiGraph,
                             semantic_overlay_manager=None):
        """Compute all features for a single grid cell."""
        grid_coords = (cell.grid_x, cell.grid_y)

        # Initialize cell features
        cell_features = CellFeatures(
            grid_x=cell.grid_x,
            grid_y=cell.grid_y
        )

        # Compute each feature type
        try:
            # Path quality based on connected edges
            path_quality = self._compute_path_quality(cell, graph)
            cell_features.set_feature(FeatureType.PATH_QUALITY, path_quality)

            # Intersection density based on node connectivity
            intersection_density = self._compute_intersection_density(cell, graph)
            cell_features.set_feature(FeatureType.INTERSECTION_DENSITY, intersection_density)

            # Nature features using semantic overlay manager if available
            if semantic_overlay_manager and hasattr(self, '_preloaded_features'):
                # Use preloaded feature data for fast computation
                forest_score = self._compute_proximity_from_preloaded(cell, 'forests')
                cell_features.set_feature(FeatureType.CLOSE_TO_FOREST, forest_score)

                rivers_score = self._compute_proximity_from_preloaded(cell, 'rivers')
                lakes_score = self._compute_proximity_from_preloaded(cell, 'lakes')
                water_score = max(rivers_score, lakes_score)  # Best of rivers/lakes
                cell_features.set_feature(FeatureType.CLOSE_TO_WATER, water_score)

                park_score = self._compute_proximity_from_preloaded(cell, 'parks')
                cell_features.set_feature(FeatureType.CLOSE_TO_PARK, park_score)

                # Compute elevation variety (simple approximation)
                elevation_variety = self._compute_elevation_variety(cell, graph)
                cell_features.set_feature(FeatureType.ELEVATION_VARIETY, elevation_variety)
            elif semantic_overlay_manager:
                # Fallback to individual scoring (slower)
                forest_score = self._compute_proximity_to_feature(
                    cell, semantic_overlay_manager, 'forests'
                )
                cell_features.set_feature(FeatureType.CLOSE_TO_FOREST, forest_score)

                water_score = self._compute_proximity_to_feature(
                    cell, semantic_overlay_manager, 'rivers'
                )
                # Also check lakes if available
                lake_score = self._compute_proximity_to_feature(
                    cell, semantic_overlay_manager, 'lakes'
                )
                water_score = max(water_score, lake_score)  # Best of rivers/lakes
                cell_features.set_feature(FeatureType.CLOSE_TO_WATER, water_score)

                park_score = self._compute_proximity_to_feature(
                    cell, semantic_overlay_manager, 'parks'
                )
                cell_features.set_feature(FeatureType.CLOSE_TO_PARK, park_score)
            else:
                # Fallback: compute from OSM tags in the graph
                forest_score = self._compute_nature_features_from_graph(cell, graph, ['forest', 'wood', 'tree'])
                cell_features.set_feature(FeatureType.CLOSE_TO_FOREST, forest_score)

                water_score = self._compute_nature_features_from_graph(cell, graph, ['water', 'river', 'lake'])
                cell_features.set_feature(FeatureType.CLOSE_TO_WATER, water_score)

                park_score = self._compute_nature_features_from_graph(cell, graph, ['park', 'garden'])
                cell_features.set_feature(FeatureType.CLOSE_TO_PARK, park_score)

            # Store metadata
            cell_features.metadata = {
                'nodes_in_cell': len(cell.nodes),
                'computation_method': 'semantic_overlay' if semantic_overlay_manager else 'graph_tags'
            }

        except Exception as e:
            logger.warning(f"Failed to compute features for cell {grid_coords}: {e}")
            # Set default values on failure
            for feature_type in FeatureType:
                if feature_type not in cell_features.features:
                    cell_features.set_feature(feature_type, 0.3)  # Neutral default

        # Store computed features
        self.cell_features[grid_coords] = cell_features

    def _compute_path_quality(self, cell: GridCell, graph: nx.MultiGraph) -> float:
        """Compute path quality score for a cell based on connected edges."""
        if not cell.nodes:
            return 0.3  # Neutral score for empty cells

        edge_scores = []

        # Analyze edges connected to nodes in this cell
        for node_id in cell.nodes:
            try:
                for neighbor in graph.neighbors(node_id):
                    edge_data = graph[node_id][neighbor]

                    # Handle multiple edges between nodes
                    for edge_attrs in edge_data.values():
                        highway_type = edge_attrs.get('highway', 'default')
                        path_score = self.path_quality_weights.get(highway_type, 0.3)
                        edge_scores.append(path_score)

                        # Bonus for named paths (often more interesting)
                        if edge_attrs.get('name'):
                            edge_scores.append(path_score + 0.1)
            except Exception:
                continue

        if not edge_scores:
            return 0.3  # Default if no edges found

        # Return weighted average with bias toward higher quality paths
        edge_scores.sort(reverse=True)

        # Weight higher scores more heavily
        if len(edge_scores) >= 3:
            # Use top 3 scores with decreasing weights
            weighted_score = (edge_scores[0] * 0.5 +
                            edge_scores[1] * 0.3 +
                            edge_scores[2] * 0.2)
        elif len(edge_scores) == 2:
            weighted_score = (edge_scores[0] * 0.7 + edge_scores[1] * 0.3)
        else:
            weighted_score = edge_scores[0]

        return min(1.0, weighted_score)

    def _compute_intersection_density(self, cell: GridCell, graph: nx.MultiGraph) -> float:
        """Compute intersection density (connectivity) score for a cell."""
        if not cell.nodes:
            return 0.0

        total_degree = 0
        node_count = 0

        for node_id in cell.nodes:
            try:
                degree = graph.degree(node_id)
                total_degree += degree
                node_count += 1
            except Exception:
                continue

        if node_count == 0:
            return 0.0

        avg_degree = total_degree / node_count

        # Normalize: degree 2 = low connectivity (0.2), degree 4+ = high connectivity (1.0)
        if avg_degree <= 2:
            return 0.2
        elif avg_degree >= 4:
            return 1.0
        else:
            # Linear interpolation between 2 and 4
            return 0.2 + (avg_degree - 2) * 0.4

    def _compute_proximity_to_feature(self, cell: GridCell, semantic_overlay_manager,
                                    feature_type: str) -> float:
        """Compute proximity score to a specific feature type using semantic overlays."""
        try:
            # Score the cell center location
            scores = semantic_overlay_manager.score_location_semantics(
                cell.center_lat, cell.center_lon, [feature_type]
            )

            return scores.get(feature_type, 0.0)

        except Exception as e:
            logger.debug(f"Failed to compute {feature_type} proximity for cell "
                        f"({cell.grid_x}, {cell.grid_y}): {e}")
            return 0.0

    def _compute_nature_features_from_graph(self, cell: GridCell, graph: nx.MultiGraph,
                                          keywords: list[str]) -> float:
        """Fallback: compute nature features from OSM tags in graph nodes."""
        if not cell.nodes:
            return 0.0

        feature_score = 0.0
        nodes_checked = 0

        # Check nodes in this cell and immediate neighbors
        all_nodes = set(cell.nodes)

        # Add neighboring nodes for broader context
        for node_id in cell.nodes:
            try:
                all_nodes.update(graph.neighbors(node_id))
            except Exception:
                continue

        # Score nodes based on tags
        for node_id in all_nodes:
            try:
                node_data = graph.nodes[node_id]
                nodes_checked += 1

                # Check all node attributes for keywords
                node_score = 0.0
                for _key, value in node_data.items():
                    if isinstance(value, str):
                        value_lower = value.lower()
                        for keyword in keywords:
                            if keyword in value_lower:
                                node_score = max(node_score, 0.8)
                                break

                feature_score = max(feature_score, node_score)

                # Don't check too many nodes for performance
                if nodes_checked > 50:
                    break

            except Exception:
                continue

        return feature_score

    def _compute_elevation_variety(self, cell: GridCell, graph: nx.MultiGraph) -> float:
        """Compute elevation variety score based on node elevation data."""
        if not cell.nodes:
            return 0.3  # Neutral score for empty cells

        elevations = []
        for node_id in cell.nodes:
            try:
                node_data = graph.nodes[node_id]
                # Try to get elevation from OSM data
                elevation = node_data.get('elevation', None)
                if elevation is not None:
                    try:
                        elevations.append(float(elevation))
                    except (ValueError, TypeError):
                        pass

                # Also check for 'ele' tag which is common in OSM
                ele = node_data.get('ele', None)
                if ele is not None:
                    try:
                        elevations.append(float(ele))
                    except (ValueError, TypeError):
                        pass

                # Don't check too many nodes for performance
                if len(elevations) > 20:
                    break

            except Exception:
                continue

        if len(elevations) < 2:
            # No elevation data available - use a simple heuristic based on node density variation
            # Areas with varying node density often indicate terrain changes
            if len(cell.nodes) > 0:
                # Simple heuristic: more nodes in a cell might indicate flatter terrain
                # Fewer nodes might indicate hills/obstacles
                len(cell.nodes) / max(1, len(cell.nodes))
                # Convert to terrain variety: lower density = more variety
                if len(cell.nodes) < 5:
                    return 0.7  # Sparse area, likely varied terrain
                elif len(cell.nodes) < 15:
                    return 0.5  # Medium density
                else:
                    return 0.3  # Dense area, likely flat
            return 0.0  # No data at all

        # Calculate elevation variety
        elevation_range = max(elevations) - min(elevations)
        elevation_std = 0.0
        if len(elevations) > 1:
            mean_elev = sum(elevations) / len(elevations)
            elevation_std = (sum((e - mean_elev) ** 2 for e in elevations) / len(elevations)) ** 0.5

        # Normalize to 0-1 range
        # 10m elevation change = 0.5, 20m+ = 1.0
        range_score = min(1.0, elevation_range / 20.0)
        # 5m std dev = 0.5, 10m+ std dev = 1.0
        std_score = min(1.0, elevation_std / 10.0)

        # Combine range and standard deviation
        variety_score = (range_score + std_score) / 2.0

        return max(0.0, min(1.0, variety_score))

    def _preload_semantic_data(self, semantic_overlay_manager):
        """Pre-load semantic data for the entire area to avoid redundant requests."""
        try:
            # Calculate overall bounding box for all grid cells
            min_lat = min(cell.center_lat for cell in self.spatial_grid.grid.values())
            max_lat = max(cell.center_lat for cell in self.spatial_grid.grid.values())
            min_lon = min(cell.center_lon for cell in self.spatial_grid.grid.values())
            max_lon = max(cell.center_lon for cell in self.spatial_grid.grid.values())

            # Add margin for feature proximity calculations
            margin = 0.01  # roughly 1km
            min_lat -= margin
            max_lat += margin
            min_lon -= margin
            max_lon += margin

            logger.info(f"Pre-loading semantic data for area: {min_lat:.4f} to {max_lat:.4f} lat, "
                       f"{min_lon:.4f} to {max_lon:.4f} lon")

            # Import BoundingBox here to avoid circular imports
            from semantic_overlays import BoundingBox
            bbox = BoundingBox(south=min_lat, west=min_lon, north=max_lat, east=max_lon)

            # Pre-load all feature types for the entire area and store for fast access
            self._preloaded_features = {}
            for feature_type in ['forests', 'rivers', 'lakes', 'parks']:
                try:
                    feature_data = semantic_overlay_manager.get_semantic_overlays(feature_type, bbox, use_cache=True)
                    self._preloaded_features[feature_type] = feature_data
                    logger.info(f"Pre-loaded {feature_type} data: {len(feature_data.get('features', []))} features")
                except Exception as e:
                    logger.warning(f"Failed to pre-load {feature_type}: {e}")
                    self._preloaded_features[feature_type] = {'features': []}

        except Exception as e:
            logger.error(f"Failed to pre-load semantic data: {e}")
            self._preloaded_features = {}

    def _compute_proximity_from_preloaded(self, cell: GridCell, feature_type: str) -> float:
        """Compute proximity score using preloaded feature data (fast)."""
        try:
            if not hasattr(self, '_preloaded_features') or feature_type not in self._preloaded_features:
                return 0.0

            feature_data = self._preloaded_features[feature_type]
            features = feature_data.get('features', [])

            if not features:
                return 0.0

            # Calculate minimum distance to any feature of this type
            min_distance = float('inf')
            cell_lat, cell_lon = cell.center_lat, cell.center_lon

            for feature in features:
                geometry = feature.get('geometry', {})
                coordinates = geometry.get('coordinates', [])

                if geometry.get('type') == 'Polygon' and coordinates:
                    # For polygons, check distance to perimeter
                    for point in coordinates[0]:  # Outer ring
                        if len(point) >= 2:
                            point_lon, point_lat = point[0], point[1]
                            distance = self._haversine_distance(cell_lat, cell_lon, point_lat, point_lon)
                            min_distance = min(min_distance, distance)
                elif geometry.get('type') == 'LineString' and coordinates:
                    # For lines (rivers), check distance to line segments
                    for point in coordinates:
                        if len(point) >= 2:
                            point_lon, point_lat = point[0], point[1]
                            distance = self._haversine_distance(cell_lat, cell_lon, point_lat, point_lon)
                            min_distance = min(min_distance, distance)

            if min_distance == float('inf'):
                return 0.0

            # Convert distance to score (closer = higher score)
            # 0m = 1.0, 500m = 0.5, 1000m = 0.0
            if min_distance <= 0:
                return 1.0
            elif min_distance >= 1000:
                return 0.0
            else:
                return 1.0 - (min_distance / 1000.0)

        except Exception as e:
            logger.debug(f"Failed to compute proximity for {feature_type}: {e}")
            return 0.0

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters between two points."""

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

    def get_statistics(self) -> dict:
        """Get statistics about the feature database."""
        if not self.cell_features:
            return {"error": "No features computed"}

        # Calculate feature statistics
        feature_stats = {}

        for feature_type in FeatureType:
            values = []
            for cell_features in self.cell_features.values():
                value = cell_features.get_feature(feature_type)
                values.append(value)

            if values:
                feature_stats[feature_type.value] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'non_zero_count': sum(1 for v in values if v > 0.1)
                }

        return {
            'cells_with_features': len(self.cell_features),
            'total_grid_cells': len(self.spatial_grid.grid),
            'coverage_percentage': (len(self.cell_features) / len(self.spatial_grid.grid) * 100)
                                  if self.spatial_grid.grid else 0,
            'feature_statistics': feature_stats,
            'computation_parameters': {
                'max_search_distance': self.max_search_distance,
                'path_quality_weights': self.path_quality_weights
            }
        }

    def clear(self):
        """Clear all computed features."""
        self.cell_features.clear()
        logger.info("Cleared feature database")
