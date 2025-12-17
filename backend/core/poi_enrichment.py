"""
POI Enrichment for Perfect10k
Fetches Points of Interest and semantic features from OSM to enhance route recommendations.
"""

import math
from typing import TypedDict

import osmnx as ox
from loguru import logger


class POI(TypedDict):
    """Type definition for Point of Interest."""

    lat: float
    lon: float
    features: list[str]
    name: str
    main_tag: str


class POIEnricher:
    """
    Enriches graph data with Points of Interest and semantic features from OSM.

    Since the walking network graph doesn't include POIs, we fetch them separately
    and spatially index them for fast lookup during semantic analysis.
    """

    def __init__(self):
        self.pois: list[POI] = []
        self.spatial_index: dict[tuple[int, int], list[int]] = {}  # Grid -> POI indices
        self.grid_size = 0.001  # ~100m grid for spatial indexing

    def fetch_pois(self, lat: float, lon: float, radius: float = 8000) -> None:
        """
        Fetch POIs around a location from OSM.
        This includes parks, water features, shops, restaurants, viewpoints, etc.
        """
        logger.info(f"Fetching POIs for ({lat:.6f}, {lon:.6f}) within {radius}m")

        # Define OSM tags for different semantic categories
        poi_tags = {
            # Water features
            "natural": [
                "water",
                "lake",
                "pond",
                "spring",
                "wood",
                "forest",
                "grassland",
                "heath",
                "scrub",
            ],
            "waterway": ["river", "stream", "canal"],
            "amenity": ["fountain", "restaurant", "cafe", "bar", "shop", "market"],
            # Nature and parks
            "leisure": ["park", "garden", "nature_reserve", "recreation_ground"],
            "landuse": ["forest", "grass", "recreation_ground"],
            # Scenic and historic
            "tourism": ["viewpoint", "attraction", "museum", "monument"],
            "historic": ["monument", "memorial", "castle", "ruins"],
            # Urban amenities
            "shop": True,  # All shop types
        }

        self.pois = []

        try:
            # Fetch POIs by category
            for main_tag, values in poi_tags.items():
                try:
                    # Fetch all values for this tag if True, else specific values
                    tags: dict[str, bool | str | list[str]] = (
                        {main_tag: True} if values is True else {main_tag: values}
                    )

                    gdf = ox.features_from_point((lat, lon), dist=radius, tags=tags)

                    if not gdf.empty:
                        # Convert to our POI format
                        for _idx, row in gdf.iterrows():
                            poi = self._extract_poi_info(row, main_tag)
                            if poi:
                                self.pois.append(poi)

                except Exception as e:
                    logger.debug(f"No POIs found for {main_tag}: {e}")
                    continue

            logger.info(f"Found {len(self.pois)} POIs")

            # Build spatial index for fast lookup
            self._build_spatial_index()

        except Exception as e:
            logger.warning(f"Failed to fetch POIs: {e}")
            self.pois = []

    def _extract_poi_info(self, row, main_tag: str) -> POI | None:
        """Extract POI information from OSM data."""
        try:
            # Get centroid for point/polygon geometries
            if hasattr(row.geometry, "centroid"):
                centroid = row.geometry.centroid
                lat, lon = centroid.y, centroid.x
            else:
                lat, lon = row.geometry.y, row.geometry.x

            # Extract semantic features
            poi: POI = {
                "lat": lat,
                "lon": lon,
                "features": [],
                "name": row.get("name", ""),
                "main_tag": main_tag,
            }

            # Categorize based on tags
            if main_tag == "natural":
                if row.get("natural") in ["water", "lake", "pond"]:
                    poi["features"].extend(
                        ["water", "lake" if row.get("natural") == "lake" else "water"]
                    )
                elif row.get("natural") in ["wood", "forest"]:
                    poi["features"].extend(["forest", "nature"])
                elif row.get("natural") in ["grassland", "heath", "scrub"]:
                    poi["features"].extend(["nature", "green"])

            elif main_tag == "waterway":
                water_type = row.get("waterway", "water")
                poi["features"].extend(["water", water_type])

            elif main_tag == "leisure":
                leisure_type = row.get("leisure", "park")
                if leisure_type in ["park", "garden"]:
                    poi["features"].extend(["park", "nature", "green"])
                elif leisure_type == "nature_reserve":
                    poi["features"].extend(["nature", "forest"])

            elif main_tag == "landuse":
                if row.get("landuse") == "forest":
                    poi["features"].extend(["forest", "nature"])
                elif row.get("landuse") in ["grass", "recreation_ground"]:
                    poi["features"].extend(["park", "green"])

            elif main_tag == "tourism":
                tourism_type = row.get("tourism", "attraction")
                if tourism_type == "viewpoint":
                    poi["features"].extend(["viewpoint", "scenic"])
                elif tourism_type in ["attraction", "monument"]:
                    poi["features"].extend(["scenic", "monument"])
                elif tourism_type == "museum":
                    poi["features"].extend(["historic", "cultural"])

            elif main_tag == "historic":
                poi["features"].extend(["historic", "monument"])

            elif main_tag == "amenity":
                amenity_type = row.get("amenity", "amenity")
                if amenity_type in ["restaurant", "cafe", "bar"]:
                    poi["features"].extend(["cafe", "restaurant", "urban"])
                elif amenity_type == "fountain":
                    poi["features"].extend(["water", "fountain"])
                elif amenity_type in ["shop", "market"]:
                    poi["features"].extend(["shop", "urban"])

            elif main_tag == "shop":
                poi["features"].extend(["shop", "urban"])

            # Add name-based features
            if poi["name"] and isinstance(poi["name"], str):
                name_lower = poi["name"].lower()
                if any(keyword in name_lower for keyword in ["park", "garden"]):
                    poi["features"].extend(["park", "nature"])
                if any(keyword in name_lower for keyword in ["lake", "pond", "river"]):
                    poi["features"].extend(["water"])
                if any(keyword in name_lower for keyword in ["view", "overlook", "scenic"]):
                    poi["features"].extend(["scenic", "viewpoint"])

            if poi["features"]:
                return poi
            return None

        except Exception as e:
            logger.debug(f"Failed to extract POI info: {e}")
            return None

    def _build_spatial_index(self):
        """Build spatial grid index for fast POI lookup."""
        self.spatial_index = {}

        for idx, poi in enumerate(self.pois):
            grid_x = int(poi["lat"] / self.grid_size)
            grid_y = int(poi["lon"] / self.grid_size)

            if (grid_x, grid_y) not in self.spatial_index:
                self.spatial_index[(grid_x, grid_y)] = []
            self.spatial_index[(grid_x, grid_y)].append(idx)

        logger.info(f"Built spatial index with {len(self.spatial_index)} grid cells")

    def get_nearby_features(self, lat: float, lon: float, radius_meters: float = 200) -> list[str]:
        """
        Get semantic features for POIs near a location.
        Returns list of features like ['water', 'park', 'scenic'].
        """
        if not self.pois:
            return []

        features = []
        radius_degrees = radius_meters / 111320.0  # Approximate conversion

        # Get nearby grid cells
        center_grid_x = int(lat / self.grid_size)
        center_grid_y = int(lon / self.grid_size)

        # Check surrounding grid cells
        grid_radius = max(1, int(radius_degrees / self.grid_size) + 1)

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_key = (center_grid_x + dx, center_grid_y + dy)

                if grid_key in self.spatial_index:
                    for poi_idx in self.spatial_index[grid_key]:
                        poi = self.pois[poi_idx]

                        # Check actual distance
                        distance = self._haversine_distance(lat, lon, poi["lat"], poi["lon"])
                        if distance <= radius_meters:
                            features.extend(poi["features"])

        # Remove duplicates while preserving order
        unique_features = []
        for feature in features:
            if feature not in unique_features:
                unique_features.append(feature)

        return unique_features

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters."""
        earth_radius_m = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return earth_radius_m * c

    def get_stats(self) -> dict:
        """Get statistics about loaded POIs."""
        if not self.pois:
            return {"total_pois": 0, "features": {}}

        feature_counts = {}
        for poi in self.pois:
            for feature in poi["features"]:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        return {
            "total_pois": len(self.pois),
            "features": feature_counts,
            "spatial_index_cells": len(self.spatial_index),
        }
