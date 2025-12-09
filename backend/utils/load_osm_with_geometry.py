#!/usr/bin/env python3
"""
OSM Graph Loading Utility with Geometry Preservation
===================================================

This utility provides functions to load OSM data with proper geometry preservation
for accurate route visualization. The geometry data is essential for displaying
routes that follow the actual curves and turns of streets rather than straight lines.

Usage:
    python load_osm_with_geometry.py --lat <lat> --lon <lon> --radius <radius>

Example:
    python load_osm_with_geometry.py --lat 40.7128 --lon -74.0060 --radius 5000
"""

import argparse
import sys
from pathlib import Path

import networkx as nx
import osmnx as ox
from loguru import logger

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from core.spatial_tile_storage import SpatialTileStorage  # noqa: E402


def load_osm_area_with_geometry(lat: float, lon: float, radius: int = 8000) -> nx.MultiGraph:
    """
    Load OSM data for an area with full geometry preservation.

    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius: Radius in meters (default: 8000m for ~10k steps)

    Returns:
        NetworkX MultiGraph with geometry data preserved
    """
    logger.info(f"Loading OSM data for area: lat={lat}, lon={lon}, radius={radius}m")

    try:
        # Load graph with parameters to preserve geometry for curved visualization
        graph = ox.graph_from_point(
            (lat, lon),
            dist=radius,
            network_type="walk",
            retain_all=True,  # Keep all nodes and edges
            truncate_by_edge=True,  # More accurate boundary handling
            simplify=False,  # Keep all intermediate nodes for better geometry
        )

        logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

        # Verify geometry data is present
        geometry_count = 0
        total_edges = 0

        for _u, _v, data in graph.edges(data=True):
            total_edges += 1
            if "geometry" in data and data["geometry"] is not None:
                geometry_count += 1

        logger.info(
            f"Geometry data: {geometry_count}/{total_edges} edges ({geometry_count / total_edges * 100:.1f}%)"
        )

        if geometry_count == 0:
            logger.warning(
                "No geometry data found in loaded graph! This will result in straight-line route visualization."
            )

        return graph

    except Exception as e:
        logger.error(f"Failed to load OSM data: {e}")
        raise


def load_osm_bbox_with_geometry(
    north: float, south: float, east: float, west: float
) -> nx.MultiGraph:
    """
    Load OSM data for a bounding box with full geometry preservation.

    Args:
        north: Northern boundary latitude
        south: Southern boundary latitude
        east: Eastern boundary longitude
        west: Western boundary longitude

    Returns:
        NetworkX MultiGraph with geometry data preserved
    """
    logger.info(f"Loading OSM data for bbox: N={north}, S={south}, E={east}, W={west}")

    try:
        # Load graph with parameters to preserve geometry
        graph = ox.graph_from_bbox(  # pyright: ignore[reportCallIssue]
            north,
            south,
            east,
            west,
            network_type="walk",
            retain_all=True,
            truncate_by_edge=True,
            simplify=False,
        )

        logger.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph

    except Exception as e:
        logger.error(f"Failed to load OSM data: {e}")
        raise


def update_tile_cache_with_geometry(lat: float, lon: float, radius: int = 8000):
    """
    Load OSM data with geometry and update the spatial tile cache.
    This will replace existing tiles with geometry-enhanced versions.

    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius: Radius in meters
    """
    logger.info("Updating tile cache with geometry-enhanced data")

    # Load graph with geometry
    graph = load_osm_area_with_geometry(lat, lon, radius)

    # Initialize spatial tile storage
    tile_storage = SpatialTileStorage()

    # Store the graph (this will create tiles with geometry)
    stored_tiles = tile_storage.store_graph(graph, set())  # type: ignore[attr-defined]

    logger.info(f"Updated {len(stored_tiles)} tiles with geometry data")
    return stored_tiles


def verify_tile_geometry(geohash: str) -> bool:
    """
    Verify that a specific tile contains geometry data.

    Args:
        geohash: The tile geohash to check

    Returns:
        True if geometry data is present, False otherwise
    """
    tile_storage = SpatialTileStorage()

    try:
        # Load the tile
        tile_path = tile_storage.cache_dir / f"tile_{geohash}.pickle"  # type: ignore[attr-defined]
        if not tile_path.exists():
            logger.error(f"Tile {geohash} does not exist")
            return False

        # Load and check the graph
        import pickle

        with open(tile_path, "rb") as f:
            tile_data = pickle.load(f)

        graph = tile_data["graph"]

        # Check for geometry in edges
        geometry_count = 0
        total_edges = 0

        for _u, _v, data in graph.edges(data=True):
            total_edges += 1
            if "geometry" in data and data["geometry"] is not None:
                geometry_count += 1

        logger.info(
            f"Tile {geohash}: {geometry_count}/{total_edges} edges have geometry ({geometry_count / total_edges * 100:.1f}%)"
        )
        return geometry_count > 0

    except Exception as e:
        logger.error(f"Failed to verify tile {geohash}: {e}")
        return False


def main():
    """Command line interface for OSM loading with geometry."""
    parser = argparse.ArgumentParser(description="Load OSM data with geometry preservation")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--radius", type=int, default=8000, help="Radius in meters (default: 8000)")
    parser.add_argument(
        "--update-cache", action="store_true", help="Update tile cache with loaded data"
    )
    parser.add_argument(
        "--verify-tile", type=str, help="Verify geometry in specific tile (geohash)"
    )

    args = parser.parse_args()

    if args.verify_tile:
        has_geometry = verify_tile_geometry(args.verify_tile)
        if has_geometry:
            logger.info(f"✅ Tile {args.verify_tile} has geometry data")
        else:
            logger.warning(f"❌ Tile {args.verify_tile} lacks geometry data")
        return

    if args.update_cache:
        stored_tiles = update_tile_cache_with_geometry(args.lat, args.lon, args.radius)
        logger.info(f"✅ Updated {len(stored_tiles)} tiles with geometry")
    else:
        graph = load_osm_area_with_geometry(args.lat, args.lon, args.radius)
        logger.info(f"✅ Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")


if __name__ == "__main__":
    main()
