"""
Semantic Overlays Module - OSM Natural Feature Extraction
Fetches and caches forests, rivers, and lakes from OpenStreetMap data.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger

# Removed semantic_scoring dependency - now using interpretable_scorer


@dataclass
class BoundingBox:
    """Geographic bounding box for OSM queries."""
    south: float
    west: float
    north: float
    east: float

    def to_overpass_bbox(self) -> str:
        """Convert to Overpass API bbox format: south,west,north,east"""
        return f"{self.south},{self.west},{self.north},{self.east}"


class SemanticOverlayManager:
    """Manages fetching and caching of OSM natural features."""

    def __init__(self, cache_dir: str = "cache/overlays"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Overpass API endpoint
        self.overpass_url = "https://overpass-api.de/api/interpreter"

        # Lazy initialize semantic scorer for faster startup
        self._semantic_scorer = None

        # Initialize feature configurations
        self.__post_init__()

    @property
    def semantic_scorer(self):
        """Get semantic scorer instance (lazy initialization) - DEPRECATED."""
        # Semantic scoring is now handled by interpretable_scorer
        # This is kept for compatibility but returns None
        return None

    def __post_init__(self):
        """Initialize feature configurations."""
        # Feature type configurations
        self.feature_configs = {
            "forests": {
                "overpass_query": """
                    [out:json][timeout:30];
                    (
                        way["natural"="wood"](bbox);
                        way["landuse"="forest"](bbox);
                        relation["natural"="wood"](bbox);
                        relation["landuse"="forest"](bbox);
                    );
                    out geom;
                """,
                "style": {
                    "color": "#228B22",
                    "fillColor": "#32CD32",
                    "fillOpacity": 0.3,
                    "weight": 1
                }
            },
            "rivers": {
                "overpass_query": """
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
                "style": {
                    "color": "#0077BE",
                    "weight": 2,
                    "opacity": 0.8
                }
            },
            "lakes": {
                "overpass_query": """
                    [out:json][timeout:30];
                    (
                        way["natural"="water"](bbox);
                        way["leisure"="swimming_pool"](bbox);
                        relation["natural"="water"](bbox);
                        way["water"](bbox);
                    );
                    out geom;
                """,
                "style": {
                    "color": "#0077BE",
                    "fillColor": "#87CEEB",
                    "fillOpacity": 0.4,
                    "weight": 1
                }
            },
            "parks": {
                "overpass_query": """
                    [out:json][timeout:30];
                    (
                        way["leisure"="park"](bbox);
                        way["leisure"="garden"](bbox);
                        way["leisure"="nature_reserve"](bbox);
                        relation["leisure"="park"](bbox);
                        relation["leisure"="garden"](bbox);
                    );
                    out geom;
                """,
                "style": {
                    "color": "#228B22",
                    "fillColor": "#90EE90",
                    "fillOpacity": 0.3,
                    "weight": 1
                }
            }
        }

    def _generate_cache_key(self, feature_type: str, bbox: BoundingBox) -> str:
        """Generate cache key for a feature type and bounding box."""
        bbox_str = f"{bbox.south:.4f},{bbox.west:.4f},{bbox.north:.4f},{bbox.east:.4f}"
        key_string = f"{feature_type}_{bbox_str}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False

        cache_age = time.time() - cache_path.stat().st_mtime
        max_age_seconds = max_age_hours * 3600
        return cache_age < max_age_seconds

    def _fetch_osm_features(self, feature_type: str, bbox: BoundingBox) -> list[dict]:
        """Fetch features from OSM using Overpass API."""
        if feature_type not in self.feature_configs:
            raise ValueError(f"Unknown feature type: {feature_type}")

        config = self.feature_configs[feature_type]
        query = config["overpass_query"].replace("(bbox)", f"({bbox.to_overpass_bbox()})")

        logger.info(f"Fetching {feature_type} from OSM for bbox {bbox.to_overpass_bbox()}")

        try:
            response = requests.post(
                self.overpass_url,
                data=query,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            elements = data.get("elements", [])

            logger.info(f"Fetched {len(elements)} {feature_type} features from OSM")
            return elements

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {feature_type} from OSM: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OSM response for {feature_type}: {e}")
            raise

    def _convert_to_geojson(self, elements: list[dict], feature_type: str) -> dict:
        """Convert OSM elements to GeoJSON format."""
        features = []

        for element in elements:
            if element.get("type") == "way" and "geometry" in element:
                # Convert way to LineString or Polygon
                coordinates = [[node["lon"], node["lat"]] for node in element["geometry"]]

                # For areas (forests, lakes), close the polygon if not already closed
                if feature_type in ["forests", "lakes"] and len(coordinates) > 2:
                    if coordinates[0] != coordinates[-1]:
                        coordinates.append(coordinates[0])
                    geometry_type = "Polygon"
                    coordinates = [coordinates]  # Polygon requires array of rings
                else:
                    geometry_type = "LineString"

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": geometry_type,
                        "coordinates": coordinates
                    },
                    "properties": {
                        "id": element.get("id"),
                        "tags": element.get("tags", {}),
                        "feature_type": feature_type
                    }
                }
                features.append(feature)

            elif element.get("type") == "relation":
                # Handle multipolygon relations for complex areas
                if element.get("tags", {}).get("type") == "multipolygon":
                    # This is complex - for now, skip relations
                    # In a full implementation, you'd need to resolve member ways
                    continue

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "feature_type": feature_type,
                "generated_at": time.time(),
                "total_features": len(features)
            }
        }

    def get_semantic_overlays(self, feature_type: str, bbox: BoundingBox, use_cache: bool = True) -> dict:
        """Get semantic overlay data for a specific feature type and area."""
        cache_key = self._generate_cache_key(feature_type, bbox)
        cache_path = self._get_cache_path(cache_key)

        # Try to use cached data first
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Using cached {feature_type} data: {cache_key}")
            try:
                with open(cache_path) as f:
                    cached_data = json.load(f)
                    # Add style configuration
                    cached_data["style"] = self.feature_configs[feature_type]["style"]

                    # Semantic scoring is now handled by the new interpretable_scorer system
                    logger.info(f"Loaded cached {feature_type} features for visualization")

                    return cached_data
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load cached data {cache_path}: {e}")

        # Fetch fresh data from OSM
        try:
            elements = self._fetch_osm_features(feature_type, bbox)
            geojson_data = self._convert_to_geojson(elements, feature_type)

            # Add style configuration
            geojson_data["style"] = self.feature_configs[feature_type]["style"]

            # Features loaded for visualization (scoring handled by interpretable_scorer)
            logger.info(f"Loaded {feature_type} features for visualization")

            # Cache the data
            try:
                with open(cache_path, 'w') as f:
                    json.dump(geojson_data, f, indent=2)
                logger.info(f"Cached {feature_type} data: {cache_key}")
            except OSError as e:
                logger.warning(f"Failed to cache data: {e}")

            return geojson_data

        except Exception as e:
            logger.error(f"Failed to get {feature_type} overlay data: {e}")
            # Return empty geojson on error
            return {
                "type": "FeatureCollection",
                "features": [],
                "style": self.feature_configs[feature_type]["style"],
                "error": str(e)
            }

    def get_all_overlays(self, bbox: BoundingBox, use_cache: bool = True) -> dict[str, dict]:
        """Get all semantic overlay types for a given area."""
        overlays = {}

        for feature_type in self.feature_configs.keys():
            overlays[feature_type] = self.get_semantic_overlays(feature_type, bbox, use_cache)

        return overlays

    def calculate_bbox_from_center(self, lat: float, lon: float, radius_km: float = 2.0) -> BoundingBox:
        """Calculate bounding box from center point and radius."""
        # Rough conversion: 1 degree ≈ 111 km
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * abs(lat / 90.0))  # Adjust for latitude

        return BoundingBox(
            south=lat - lat_delta,
            west=lon - lon_delta,
            north=lat + lat_delta,
            east=lon + lon_delta
        )

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "total_cached_areas": len(cache_files),
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_directory": str(self.cache_dir)
        }

    def clear_cache(self, older_than_hours: int | None = None) -> int:
        """Clear cache files, optionally only those older than specified hours."""
        cleared_count = 0
        current_time = time.time()

        for cache_file in self.cache_dir.glob("*.json"):
            should_delete = True

            if older_than_hours is not None:
                file_age = current_time - cache_file.stat().st_mtime
                max_age_seconds = older_than_hours * 3600
                should_delete = file_age > max_age_seconds

            if should_delete:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count

    def score_location_semantics(self, lat: float, lon: float,
                                property_names: list[str] | None = None) -> dict[str, float]:
        """
        Score a location based on proximity to semantic features.

        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            property_names: List of properties to score (None = all available)

        Returns:
            Dict mapping property names to scores (0.0 to 1.0)
        """
        # Ensure features are loaded for requested properties
        if property_names is None:
            property_names = ['forests', 'rivers', 'lakes', 'parks']

        # Calculate bounding box for this location (2km radius should be enough)
        bbox = self.calculate_bbox_from_center(lat, lon, 2.0)

        # Load features for each property if not already loaded
        for prop_name in property_names:
            try:
                logger.info(f"Loading {prop_name} features for scoring")
                self.get_semantic_overlays(prop_name, bbox, use_cache=True)
            except Exception as e:
                logger.warning(f"Failed to load {prop_name} for scoring: {e}")

        # Semantic scoring is now handled by the interpretable_scorer system
        # Return neutral scores for API compatibility (scoring handled by interpretable_scorer)
        return {prop: 0.5 for prop in property_names}

    def score_multiple_locations(self, locations: list[tuple[float, float]],
                               property_names: list[str] | None = None) -> list[dict[str, float]]:
        """
        Score multiple locations efficiently.

        Args:
            locations: List of (lat, lon) tuples
            property_names: List of properties to score

        Returns:
            List of score dictionaries for each location
        """
        # Semantic scoring is now handled by the interpretable_scorer system
        # Return neutral scores for API compatibility
        if property_names is None:
            property_names = ['forests', 'rivers', 'lakes', 'parks']

        return [{prop: 0.5 for prop in property_names} for _ in locations]

    def get_semantic_property_info(self) -> dict[str, dict]:
        """Get information about all semantic properties"""
        # Return basic property info for API compatibility
        return {
            'forests': {'description': 'Proximity to forest areas', 'type': 'semantic'},
            'rivers': {'description': 'Proximity to rivers and streams', 'type': 'semantic'},
            'lakes': {'description': 'Proximity to lakes and water bodies', 'type': 'semantic'},
            'parks': {'description': 'Proximity to parks and gardens', 'type': 'semantic'}
        }

    def update_semantic_property(self, property_name: str, **kwargs):
        """Update configuration for a semantic property"""
        # Property updates are now handled by the interpretable_scorer system
        # This method is kept for API compatibility but does nothing
        logger.info(f"Property update for {property_name} ignored - using interpretable_scorer system")

    def ensure_features_loaded_for_area(self, lat: float, lon: float, radius_km: float = 2.0):
        """
        Ensure semantic features are loaded for a given area.

        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Radius around center to load features
        """
        bbox = self.calculate_bbox_from_center(lat, lon, radius_km)

        # Load features for all configured types
        for feature_type in self.feature_configs.keys():
            try:
                # This will load features into the semantic scorer
                self.get_semantic_overlays(feature_type, bbox, use_cache=True)
            except Exception as e:
                logger.warning(f"Failed to load {feature_type} features for area: {e}")

    def clear_semantic_cache(self, property_name: str | None = None):
        """Clear semantic scorer cache"""
        # Clear our overlay cache instead
        if property_name:
            logger.info(f"Clearing cache for {property_name} (overlay cache only)")
            self.clear_cache()
        else:
            logger.info("Clearing all overlay cache")
            self.clear_cache()
