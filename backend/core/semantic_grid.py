"""
Semantic Grid Cache for Perfect10k
Pre-computes semantic scores on a coarse grid for fast lookup during route planning.
"""

import math
import pickle
from dataclasses import dataclass

import networkx as nx
from core.poi_enrichment import POIEnricher
from loguru import logger


@dataclass
class GridCell:
    """A cell in the semantic grid with pre-computed scores."""

    lat: float
    lon: float
    feature_scores: dict[str, float]  # Feature type -> score
    dominant_features: list[str]  # Top features in this cell
    explanation_template: str  # Pre-built explanation template


class SemanticGrid:
    """
    Grid-based semantic cache for fast preference-based scoring.

    Instead of analyzing every node individually, we pre-compute semantic
    information on a coarse grid (e.g., 100m x 100m cells) and interpolate.
    """

    def __init__(self, cell_size_meters: float = 100):
        self.cell_size_meters = cell_size_meters
        self.grid: dict[tuple[int, int], GridCell] = {}
        self.bounds = None  # (min_lat, min_lon, max_lat, max_lon)
        self.grid_origin = None  # (lat, lon) of grid origin
        self.poi_enricher = None  # Will be set during build_grid

    def build_grid(self, graph: nx.MultiGraph, semantic_matcher) -> None:
        """
        Build semantic grid by analyzing graph in coarse cells.
        This is expensive but done once when graph is first loaded.
        """
        logger.info(f"Building semantic grid with {self.cell_size_meters}m cells")

        # Calculate graph bounds
        lats = [data["y"] for _, data in graph.nodes(data=True)]
        lons = [data["x"] for _, data in graph.nodes(data=True)]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        self.bounds = (min_lat, min_lon, max_lat, max_lon)
        self.grid_origin = (min_lat, min_lon)

        # Fetch POI data for the area
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        radius = max(self._haversine_distance(min_lat, min_lon, max_lat, max_lon) / 2, 1000)  # At least 1km

        self.poi_enricher = POIEnricher()
        self.poi_enricher.fetch_pois(center_lat, center_lon, radius)

        poi_stats = self.poi_enricher.get_stats()
        logger.info(f"Enriched with {poi_stats['total_pois']} POIs: {poi_stats['features']}")

        # Calculate grid dimensions
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        # Convert to approximate grid size
        lat_cells = max(1, int(lat_range / self._meters_to_lat_degrees(self.cell_size_meters)))
        lon_cells = max(
            1, int(lon_range / self._meters_to_lon_degrees(self.cell_size_meters, min_lat))
        )

        logger.info(
            f"Grid dimensions: {lat_cells} x {lon_cells} cells covering {lat_range:.4f}° x {lon_range:.4f}°"
        )

        # Group nodes by grid cell
        cell_nodes: dict[tuple[int, int], list[int]] = {}
        for node_id, data in graph.nodes(data=True):
            grid_x, grid_y = self._coords_to_grid(data["y"], data["x"])
            if (grid_x, grid_y) not in cell_nodes:
                cell_nodes[(grid_x, grid_y)] = []
            cell_nodes[(grid_x, grid_y)].append(node_id)

        # Analyze each cell
        total_cells = len(cell_nodes)
        for i, ((grid_x, grid_y), nodes) in enumerate(cell_nodes.items()):
            if i % 50 == 0:
                logger.info(f"Processing cell {i + 1}/{total_cells}")

            cell_lat, cell_lon = self._grid_to_coords(grid_x, grid_y)
            self.grid[(grid_x, grid_y)] = self._analyze_cell(graph, nodes, cell_lat, cell_lon)

        logger.info(f"Semantic grid built with {len(self.grid)} cells")

    def get_semantic_score(self, lat: float, lon: float, preference: str) -> tuple[float, str]:
        """
        Fast lookup of semantic score using grid interpolation.
        Returns (score, explanation) for the given location and preference.
        """
        # Find grid cell
        grid_x, grid_y = self._coords_to_grid(lat, lon)

        # Get cell data (with fallback to nearby cells if needed)
        cell = self._get_cell_with_fallback(grid_x, grid_y)
        if not cell:
            return 0.5, "Basic walkable area"

        # Calculate score based on preference and cell features
        score = self._calculate_preference_score(cell, preference)
        explanation = self._build_explanation(cell, preference)

        return score, explanation

    def _analyze_cell(
        self, graph: nx.MultiGraph, nodes: list[int], cell_lat: float, cell_lon: float
    ) -> GridCell:
        """Analyze all nodes in a cell to extract semantic features."""
        feature_counts = {}
        feature_weights = {
            # Water features (highest priority)
            "water": 1.0,
            "river": 1.0,
            "lake": 1.0,
            "pond": 0.8,
            "stream": 0.8,
            "fountain": 0.6,
            # Nature features
            "park": 0.9,
            "garden": 0.8,
            "forest": 0.9,
            "tree": 0.6,
            "green": 0.7,
            "nature": 0.8,
            # Scenic features
            "viewpoint": 0.9,
            "scenic": 0.8,
            "vista": 0.8,
            "overlook": 0.8,
            "monument": 0.7,
            # Path quality
            "footway": 0.8,
            "path": 0.7,
            "track": 0.6,
            "cycleway": 0.5,
            # Urban amenities
            "shop": 0.5,
            "cafe": 0.6,
            "restaurant": 0.6,
            "amenity": 0.4,
            # Historic
            "historic": 0.7,
            "heritage": 0.7,
            "church": 0.5,
            "castle": 0.8,
        }

        # Get POI features for this cell (much more reliable than OSM graph tags)
        if hasattr(self, 'poi_enricher') and self.poi_enricher:
            poi_features = self.poi_enricher.get_nearby_features(cell_lat, cell_lon, self.cell_size_meters / 2)
            for feature in poi_features:
                if feature in feature_weights:
                    # POI features get boosted weight since they're more reliable semantic data
                    # Scale with number of nodes so they compete with path features
                    boost = max(5, len(nodes) * 0.3)  # Ensure POI features are prominent
                    feature_counts[feature] = feature_counts.get(feature, 0) + feature_weights[feature] * boost

        # Analyze all nodes and edges in cell (mainly for path quality now)
        for node_id in nodes:
            node_data = graph.nodes[node_id]

            # Check node tags (less common but still useful)
            for tag_key, tag_value in node_data.items():
                if isinstance(tag_value, str):
                    tag_text = f"{tag_key} {tag_value}".lower()

                    # Count feature occurrences
                    for feature, weight in feature_weights.items():
                        if feature in tag_text:
                            feature_counts[feature] = feature_counts.get(feature, 0) + weight * 0.5  # Reduced weight

            # Check connected edges
            try:
                for neighbor in graph.neighbors(node_id):
                    edge_data = graph[node_id][neighbor]
                    for _, edge_attrs in edge_data.items():
                        highway_type = edge_attrs.get("highway", "")
                        if highway_type in feature_weights:
                            feature_counts[highway_type] = (
                                feature_counts.get(highway_type, 0)
                                + feature_weights[highway_type] * 0.5
                            )

                        # Check edge names for features
                        edge_name = edge_attrs.get("name", "")
                        if edge_name:
                            edge_name_lower = edge_name.lower()
                            for feature, weight in feature_weights.items():
                                if feature in edge_name_lower:
                                    feature_counts[feature] = (
                                        feature_counts.get(feature, 0) + weight * 0.3
                                    )
            except Exception:
                continue

        # Normalize feature scores by cell density
        total_nodes = max(len(nodes), 1)
        feature_scores = {
            feature: min(count / total_nodes, 1.0) for feature, count in feature_counts.items()
        }

        # Find dominant features (top 3)
        dominant_features = sorted(
            feature_scores.keys(), key=lambda f: feature_scores[f], reverse=True
        )[:3]
        dominant_features = [f for f in dominant_features if feature_scores[f] > 0.1]

        # Build explanation template
        explanation_template = self._build_explanation_template(dominant_features, feature_scores)

        return GridCell(
            lat=cell_lat,
            lon=cell_lon,
            feature_scores=feature_scores,
            dominant_features=dominant_features,
            explanation_template=explanation_template,
        )

    def _calculate_preference_score(self, cell: GridCell, preference: str) -> float:
        """Calculate score based on user preference and cell features."""
        preference_lower = preference.lower()
        base_score = 0.3

        # Define preference mappings
        preference_mappings = {
            "water": ["water", "river", "lake", "pond", "stream", "fountain"],
            "nature": ["park", "garden", "forest", "tree", "green", "nature"],
            "scenic": ["viewpoint", "scenic", "vista", "overlook", "monument"],
            "quiet": ["footway", "path", "park", "garden"],
            "urban": ["shop", "cafe", "restaurant", "amenity"],
            "historic": ["historic", "heritage", "church", "castle", "monument"],
        }

        # Calculate preference boost
        preference_boost = 0.0
        for _category, keywords in preference_mappings.items():
            if any(keyword in preference_lower for keyword in keywords):
                # Add boost based on matching features in cell
                for feature in keywords:
                    feature_score = cell.feature_scores.get(feature, 0)
                    preference_boost += feature_score * 0.4  # Boost factor

        # Add path quality bonus
        path_quality = (
            max(
                cell.feature_scores.get("footway", 0),
                cell.feature_scores.get("path", 0),
                cell.feature_scores.get("track", 0),
            )
            * 0.3
        )

        final_score = base_score + preference_boost + path_quality
        return max(0.0, min(1.0, final_score))

    def _build_explanation(self, cell: GridCell, preference: str) -> str:
        """Build explanation based on cell features and user preference."""
        explanations = []

        # Use pre-computed dominant features, but skip path quality indicators
        path_quality_features = {"footway", "path", "track", "cycleway"}

        for feature in cell.dominant_features:
            if cell.feature_scores[feature] > 0.15:  # Lower threshold to catch more POI features
                # Skip path quality features - users don't care about infrastructure details
                if feature in path_quality_features:
                    continue
                elif feature in ["water", "river", "lake", "pond", "stream"]:
                    explanations.append(f"near {feature}")
                elif feature in ["park", "garden", "forest"]:
                    explanations.append(f"in {feature} area")
                elif feature in ["viewpoint", "scenic", "vista"]:
                    explanations.append(f"scenic {feature}")
                elif feature in ["cafe", "restaurant"]:
                    explanations.append(f"near {feature}s")
                elif feature in ["shop"]:
                    explanations.append("near shops")
                elif feature in ["nature", "green"]:
                    explanations.append(f"{feature} area")
                elif feature in ["historic", "monument"]:
                    explanations.append("historic area")
                elif feature == "tree":
                    explanations.append("tree-lined")
                else:
                    explanations.append(feature)

        return ", ".join(explanations) if explanations else "walkable area"

    def _build_explanation_template(
        self, dominant_features: list[str], feature_scores: dict[str, float]
    ) -> str:
        """Pre-build explanation template for fast lookup."""
        explanations = []

        # Skip path quality features - users don't care about infrastructure details
        path_quality_features = {"footway", "path", "track", "cycleway"}

        for feature in dominant_features:
            if feature_scores[feature] > 0.15:  # Lower threshold to catch more POI features
                # Skip path quality features
                if feature in path_quality_features:
                    continue
                elif feature in ["water", "river", "lake", "pond", "stream"]:
                    explanations.append(f"near {feature}")
                elif feature in ["park", "garden", "forest"]:
                    explanations.append(f"in {feature} area")
                elif feature in ["viewpoint", "scenic", "vista"]:
                    explanations.append(f"scenic {feature}")
                elif feature in ["cafe", "restaurant"]:
                    explanations.append(f"near {feature}s")
                elif feature in ["shop"]:
                    explanations.append("near shops")
                elif feature in ["nature", "green"]:
                    explanations.append(f"{feature} area")
                elif feature in ["historic", "monument"]:
                    explanations.append("historic area")
                elif feature == "tree":
                    explanations.append("tree-lined")
                else:
                    explanations.append(feature)

        return ", ".join(explanations) if explanations else "walkable area"

    def _coords_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """Convert lat/lon coordinates to grid indices."""
        if not self.grid_origin:
            return (0, 0)

        origin_lat, origin_lon = self.grid_origin

        # Convert to approximate grid coordinates
        lat_offset = lat - origin_lat
        lon_offset = lon - origin_lon

        grid_x = int(lat_offset / self._meters_to_lat_degrees(self.cell_size_meters))
        grid_y = int(lon_offset / self._meters_to_lon_degrees(self.cell_size_meters, lat))

        return (grid_x, grid_y)

    def _grid_to_coords(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        """Convert grid indices to lat/lon coordinates (cell center)."""
        if not self.grid_origin:
            return (0.0, 0.0)

        origin_lat, origin_lon = self.grid_origin

        lat = origin_lat + (grid_x + 0.5) * self._meters_to_lat_degrees(self.cell_size_meters)
        lon = origin_lon + (grid_y + 0.5) * self._meters_to_lon_degrees(self.cell_size_meters, lat)

        return (lat, lon)

    def _get_cell_with_fallback(self, grid_x: int, grid_y: int) -> GridCell:
        """Get cell data with fallback to nearby cells if exact cell doesn't exist."""
        # Try exact cell first
        if (grid_x, grid_y) in self.grid:
            return self.grid[(grid_x, grid_y)]

        # Try nearby cells (3x3 neighborhood)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (grid_x + dx, grid_y + dy) in self.grid:
                    return self.grid[(grid_x + dx, grid_y + dy)]

        return None

    def _meters_to_lat_degrees(self, meters: float) -> float:
        """Convert meters to latitude degrees (approximate)."""
        return meters / 111320.0

    def _meters_to_lon_degrees(self, meters: float, lat: float) -> float:
        """Convert meters to longitude degrees at given latitude (approximate)."""
        return meters / (111320.0 * math.cos(math.radians(lat)))

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def save_to_disk(self, filepath: str) -> None:
        """Save semantic grid to disk."""
        try:
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "cell_size_meters": self.cell_size_meters,
                        "grid": self.grid,
                        "bounds": self.bounds,
                        "grid_origin": self.grid_origin,
                        "poi_enricher": self.poi_enricher,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"Semantic grid saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save semantic grid: {e}")
            raise

    def load_from_disk(self, filepath: str) -> bool:
        """Load semantic grid from disk."""
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.cell_size_meters = data["cell_size_meters"]
                self.grid = data["grid"]
                self.bounds = data["bounds"]
                self.grid_origin = data["grid_origin"]
                self.poi_enricher = data.get("poi_enricher", None)  # Backwards compatibility
            logger.info(f"Semantic grid loaded from {filepath}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load semantic grid: {e}")
            return False
