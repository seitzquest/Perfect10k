#!/usr/bin/env python3
"""
Quick script to check if cached tiles have geometry data
"""

import pickle
from pathlib import Path


def check_tile_geometry(tile_path):
    """Check if a tile has geometry data"""
    try:
        with open(tile_path, "rb") as f:
            tile_data = pickle.load(f)

        graph = tile_data["graph"]

        geometry_count = 0
        total_edges = 0

        for _u, _v, data in graph.edges(data=True):
            total_edges += 1
            if "geometry" in data and data["geometry"] is not None:
                geometry_count += 1

        return geometry_count, total_edges
    except Exception as e:
        print(f"Error checking {tile_path}: {e}")
        return 0, 0


def main():
    cache_dir = Path("cache/graphs")
    if not cache_dir.exists():
        print("Cache directory not found!")
        return

    total_geometry = 0
    total_edges = 0

    # Check first 5 tiles as sample
    tile_files = list(cache_dir.glob("tile_*.pickle"))[:5]

    print(f"Checking {len(tile_files)} sample tiles for geometry data...")

    for _tiles_checked, tile_file in enumerate(tile_files, start=1):
        geom_count, edge_count = check_tile_geometry(tile_file)
        total_geometry += geom_count
        total_edges += edge_count

        percentage = (geom_count / edge_count * 100) if edge_count > 0 else 0
        print(
            f"{tile_file.name}: {geom_count}/{edge_count} edges have geometry ({percentage:.1f}%)"
        )

    if total_edges > 0:
        overall_percentage = total_geometry / total_edges * 100
        print(
            f"\nOverall: {total_geometry}/{total_edges} edges have geometry ({overall_percentage:.1f}%)"
        )

        if overall_percentage < 10:
            print("❌ Cache lacks geometry data - routes will show as straight lines")
            print(
                "✅ Solution: Regenerate cache with geometry using the load_osm_with_geometry.py script"
            )
        else:
            print("✅ Cache has sufficient geometry data")
    else:
        print("❌ No edge data found")


if __name__ == "__main__":
    main()
