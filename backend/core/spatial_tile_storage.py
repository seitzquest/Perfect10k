"""
Spatial Tile Storage System for Perfect10k
Permanent storage with spatial locality using geohash-based tiling.
"""

import json
import math
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import geohash2
import networkx as nx
from loguru import logger


@dataclass
class TileInfo:
    """Information about a cached tile."""

    geohash: str
    bbox: tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
    node_count: int
    edge_count: int
    created_at: float
    last_accessed: float
    file_size_bytes: int
    semantic_features: set[str]  # Which semantic features are cached


class SpatialTileStorage:
    """
    Permanent spatial storage using geohash-based tiles.

    Key advantages:
    1. Spatial locality - nearby areas share tile boundaries
    2. Permanent storage - no temporal eviction
    3. Efficient queries - geohash prefix matching
    4. Scalable - supports precomputed tiles for major cities
    5. Hierarchical - different zoom levels for different needs
    """

    def __init__(self, storage_dir: str = "storage/tiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Geohash precision levels
        # Precision 6: ~1.2km x 0.6km tiles (good for 8km radius requests)
        # Precision 7: ~153m x 153m tiles (good for detailed areas)
        self.default_precision = 6
        self.detail_precision = 7

        # SQLite index for fast spatial queries
        self.db_path = self.storage_dir / "tile_index.db"
        self._init_database()

        logger.info(f"Initialized spatial tile storage at {self.storage_dir}")
        self._log_stats()

    def _init_database(self):
        """Initialize SQLite database for tile indexing."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tiles (
                geohash TEXT PRIMARY KEY,
                precision INTEGER,
                min_lat REAL,
                min_lon REAL,
                max_lat REAL,
                max_lon REAL,
                node_count INTEGER,
                edge_count INTEGER,
                created_at REAL,
                last_accessed REAL,
                file_size_bytes INTEGER,
                semantic_features TEXT  -- JSON array of feature types
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_spatial ON tiles
            (min_lat, min_lon, max_lat, max_lon)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_geohash_prefix ON tiles
            (geohash)
        """)
        self.conn.commit()

    def get_covering_tiles(self, lat: float, lon: float, radius_m: float = 8000) -> list[str]:
        """
        Get geohashes of tiles that cover the requested area.

        Returns:
            List of geohash strings for tiles covering the area
        """
        # Calculate bounding box
        lat_delta = radius_m / 111000  # Rough degree conversion
        lon_delta = radius_m / (111000 * math.cos(math.radians(lat)))

        min_lat = lat - lat_delta
        max_lat = lat + lat_delta
        min_lon = lon - lon_delta
        max_lon = lon + lon_delta

        # Find all tiles that intersect this bounding box
        cursor = self.conn.execute(
            """
            SELECT geohash FROM tiles
            WHERE max_lat >= ? AND min_lat <= ?
            AND max_lon >= ? AND min_lon <= ?
            ORDER BY last_accessed DESC
        """,
            (min_lat, max_lat, min_lon, max_lon),
        )

        covering_tiles = [row[0] for row in cursor.fetchall()]

        # If no tiles found, generate the geohashes we'd need
        if not covering_tiles:
            covering_tiles = self._generate_needed_geohashes(lat, lon, radius_m)

        return covering_tiles

    def _generate_needed_geohashes(self, lat: float, lon: float, radius_m: float) -> list[str]:
        """Generate the geohashes needed to cover an area."""
        # Get the center geohash
        center_hash = geohash2.encode(lat, lon, precision=self.default_precision)

        # Get all neighbor geohashes to ensure coverage
        needed_hashes = set()
        needed_hashes.add(center_hash)

        # Add neighbors for coverage (geohash neighbors share boundaries)
        try:
            # geohash2.neighbors returns a dict, get its values
            if hasattr(geohash2, "neighbors"):
                neighbors = geohash2.neighbors(center_hash)  # type: ignore[attr-defined]
                needed_hashes.update(neighbors.values())
        except Exception:
            # Fallback to just center if neighbors fail
            pass

        return list(needed_hashes)

    def _make_edge_data_hashable(self, edge_data: dict) -> tuple:
        """Convert edge data to a hashable tuple."""
        hashable_items = []
        for key, value in edge_data.items():
            if isinstance(value, list):
                # Convert lists to tuples
                hashable_value = tuple(value)
            elif isinstance(value, dict):
                # Convert nested dicts to sorted tuples of items
                hashable_value = tuple(sorted(value.items()))
            elif isinstance(value, set):
                # Convert sets to sorted tuples
                hashable_value = tuple(sorted(value))
            else:
                # Keep other types as-is (should be hashable)
                hashable_value = value
            hashable_items.append((key, hashable_value))
        return tuple(hashable_items)

    def load_graph_for_area(
        self, lat: float, lon: float, radius_m: float = 8000
    ) -> nx.MultiGraph | None:
        """
        Load a graph covering the requested area from tiles.
        If no tiles exist, automatically downloads OSM data with geometry.

        Returns:
            Combined graph from all covering tiles, or None if not available
        """
        covering_tiles = self.get_covering_tiles(lat, lon, radius_m)

        # Check if any of the covering tiles actually exist
        existing_tiles = []
        for geohash in covering_tiles:
            if self._tile_exists(geohash):
                existing_tiles.append(geohash)

        if not existing_tiles:
            logger.info(
                f"No existing tiles found for area ({lat:.6f}, {lon:.6f}), loading from OSM with geometry"
            )
            return self._load_osm_and_create_tiles(lat, lon, radius_m)

        # Use only existing tiles for loading
        covering_tiles = existing_tiles

        # Load and combine graphs from all covering tiles
        combined_graph = None
        tiles_loaded = 0

        for geohash in covering_tiles:
            tile_graph = self._load_tile_graph(geohash)
            if tile_graph:
                if combined_graph is None:
                    combined_graph = tile_graph.copy()
                else:
                    # Merge graphs (NetworkX handles node/edge deduplication)
                    combined_graph = nx.compose(combined_graph, tile_graph)  # type: ignore[arg-type]
                tiles_loaded += 1

                # Update access time
                self._update_tile_access(geohash)

        if tiles_loaded > 0:
            # Ensure the combined graph has proper OSMnx CRS metadata
            if combined_graph is not None:
                # Set CRS and other OSMnx metadata that might be missing
                combined_graph.graph["crs"] = "epsg:4326"
                if "created_with" not in combined_graph.graph:
                    combined_graph.graph["created_with"] = "osmnx"
                if "simplified" not in combined_graph.graph:
                    combined_graph.graph["simplified"] = True

                # Ensure connectivity by using largest connected component
                if not nx.is_connected(combined_graph.to_undirected()):
                    logger.warning(
                        f"Combined graph has {nx.number_connected_components(combined_graph.to_undirected())} components, using largest"
                    )
                    largest_cc = max(
                        nx.connected_components(combined_graph.to_undirected()), key=len
                    )
                    combined_graph = combined_graph.subgraph(largest_cc).copy()
                    logger.info(
                        f"Using largest connected component: {len(combined_graph.nodes)} nodes"
                    )

            logger.info(f"Loaded graph from {tiles_loaded} tiles covering ({lat:.6f}, {lon:.6f})")
            return combined_graph  # type: ignore[return-value]

        return None

    def _load_osm_and_create_tiles(
        self, lat: float, lon: float, radius_m: float = 8000
    ) -> nx.MultiGraph | None:
        """
        Load OSM data with geometry and create tiles automatically.
        This ensures new areas always have proper geometry data.
        """
        try:
            import osmnx as ox

            logger.info(
                f"Loading OSM data with geometry for ({lat:.6f}, {lon:.6f}), radius={radius_m}m"
            )

            # Load with proper parameters to ensure route visualization with geometry
            try:
                graph = ox.graph_from_point(
                    (lat, lon),
                    dist=radius_m,
                    network_type="walk",
                    retain_all=True,
                    truncate_by_edge=True,
                    simplify=False,  # Keep intermediate nodes for better geometry
                )
            except Exception as osm_error:
                logger.warning(
                    f"Initial OSM load failed: {osm_error}, trying with simplified parameters..."
                )
                # Fallback to simpler parameters
                try:
                    graph = ox.graph_from_point(
                        (lat, lon),
                        dist=radius_m,
                        network_type="walk",
                        simplify=True,  # Allow simplification as fallback
                    )
                    logger.info("✅ OSM load succeeded with simplified parameters")
                except Exception as fallback_error:
                    logger.error(f"OSM loading failed completely: {fallback_error}")
                    return None

            # Add geometry data to edges (required for curved route visualization)
            logger.info("Adding geometry data to edges...")
            try:
                # Convert to GeoDataFrames to get geometries
                _gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

                # Add geometry from GeoDataFrame back to graph edges
                for idx, edge_data in gdf_edges.iterrows():
                    # idx is a tuple of (u, v, key) for MultiGraph edges
                    if isinstance(idx, tuple) and len(idx) >= 3:
                        u, v, key = idx[0], idx[1], idx[2]
                        if hasattr(edge_data, "geometry") and edge_data.geometry is not None:
                            graph[u][v][key]["geometry"] = edge_data.geometry

            except Exception as e:
                logger.warning(f"Could not add detailed geometry: {e}")
                # Fallback: create simple line geometries between nodes
                try:
                    from shapely.geometry import LineString

                    for u, v, _key, edge_data in graph.edges(keys=True, data=True):  # type: ignore[call-overload]
                        if "geometry" not in edge_data:
                            u_coord = (graph.nodes[u]["x"], graph.nodes[u]["y"])
                            v_coord = (graph.nodes[v]["x"], graph.nodes[v]["y"])
                            edge_data["geometry"] = LineString([u_coord, v_coord])
                except Exception as e2:
                    logger.warning(f"Could not create fallback geometry: {e2}")

            logger.info(
                f"Loaded OSM graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
            )

            # Verify geometry data
            geometry_count = 0
            total_edges = 0
            for _u, _v, data in graph.edges(data=True):
                total_edges += 1
                if "geometry" in data and data["geometry"] is not None:
                    geometry_count += 1

            geometry_percentage = (geometry_count / total_edges * 100) if total_edges > 0 else 0
            logger.info(
                f"Geometry coverage: {geometry_count}/{total_edges} edges ({geometry_percentage:.1f}%)"
            )

            # Store as tiles for future use
            stored_tiles = self.store_graph_tiles(graph, lat, lon, radius_m, set())
            logger.info(f"Stored graph as {len(stored_tiles)} tiles with geometry data")

            return graph

        except ImportError:
            logger.error("osmnx not available - cannot load OSM data")
            return None
        except Exception as e:
            logger.error(f"Failed to load OSM data: {e}")
            return None

    def store_graph_tiles(
        self,
        graph: nx.MultiGraph,
        center_lat: float,
        center_lon: float,
        radius_m: float = 8000,
        semantic_features: set[str] | None = None,
    ) -> list[str]:
        """
        Store a graph by splitting it into spatial tiles.

        Returns:
            List of geohash strings for stored tiles
        """
        semantic_features = semantic_features or set()
        stored_tiles = []

        # Group nodes by geohash
        geohash_groups = {}

        for node_id, data in graph.nodes(data=True):
            lat, lon = data.get("y"), data.get("x")
            if lat is None or lon is None:
                continue

            geohash = geohash2.encode(lat, lon, precision=self.default_precision)

            if geohash not in geohash_groups:
                geohash_groups[geohash] = {"nodes": set(), "edges": set()}

            geohash_groups[geohash]["nodes"].add(node_id)

        # Add edges to appropriate tiles and ensure boundary nodes are included
        for edge in graph.edges(data=True):
            node1, node2, edge_data = edge

            # Get geohashes for both nodes
            node1_data = graph.nodes.get(node1, {})
            node2_data = graph.nodes.get(node2, {})

            lat1, lon1 = node1_data.get("y"), node1_data.get("x")
            lat2, lon2 = node2_data.get("y"), node2_data.get("x")

            if all([lat1, lon1, lat2, lon2]):
                hash1 = geohash2.encode(lat1, lon1, precision=self.default_precision)
                hash2 = geohash2.encode(lat2, lon2, precision=self.default_precision)

                # Convert edge data to hashable form
                hashable_edge_data = self._make_edge_data_hashable(edge_data)

                # Add edge to both tiles (ensures connectivity across boundaries)
                for geohash in [hash1, hash2]:
                    if geohash in geohash_groups:
                        geohash_groups[geohash]["edges"].add((node1, node2, hashable_edge_data))

                        # Ensure both nodes are included in tiles that have this edge
                        geohash_groups[geohash]["nodes"].add(node1)
                        geohash_groups[geohash]["nodes"].add(node2)

        # Store each tile
        for geohash, tile_data in geohash_groups.items():
            if len(tile_data["nodes"]) > 0:  # Only store non-empty tiles
                tile_graph = self._create_tile_subgraph(
                    graph, tile_data["nodes"], tile_data["edges"]
                )
                tile_path = self._store_tile_graph(geohash, tile_graph, semantic_features)

                if tile_path:
                    stored_tiles.append(geohash)

        logger.info(f"Stored graph as {len(stored_tiles)} tiles")
        return stored_tiles

    def _create_tile_subgraph(
        self, full_graph: nx.MultiGraph, nodes: set, edges: set
    ) -> nx.MultiGraph:
        """Create a subgraph for a tile with geometry preservation."""
        tile_graph = nx.MultiGraph()

        # Add nodes with their data
        for node_id in nodes:
            if node_id in full_graph.nodes:
                tile_graph.add_node(node_id, **full_graph.nodes[node_id])

        # Add edges with their data, ensuring geometry is preserved
        geometry_preserved = 0
        total_edges = 0

        for node1, node2, edge_data_tuple in edges:
            total_edges += 1
            edge_data = dict(edge_data_tuple)

            # Explicitly check for and preserve geometry data
            if "geometry" in edge_data and edge_data["geometry"] is not None:
                geometry_preserved += 1

            if node1 in tile_graph.nodes and node2 in tile_graph.nodes:
                tile_graph.add_edge(node1, node2, **edge_data)

        if total_edges > 0:
            logger.debug(
                f"Tile subgraph: {geometry_preserved}/{total_edges} edges have geometry ({geometry_preserved / total_edges * 100:.1f}%)"
            )

        return tile_graph

    def _store_tile_graph(
        self, geohash: str, graph: nx.MultiGraph, semantic_features: set[str]
    ) -> Path | None:
        """Store a single tile graph to disk."""
        try:
            # Calculate bounding box for this geohash
            lat, lon, lat_err, lon_err = geohash2.decode_exactly(geohash)
            min_lat = lat - lat_err
            max_lat = lat + lat_err
            min_lon = lon - lon_err
            max_lon = lon + lon_err
            bbox = (min_lat, min_lon, max_lat, max_lon)

            # Create file path
            tile_file = self.storage_dir / f"tile_{geohash}.pickle"

            # Store graph data
            tile_data = {
                "graph": graph,
                "geohash": geohash,
                "bbox": bbox,
                "semantic_features": list(semantic_features),
                "created_at": time.time(),
            }

            with open(tile_file, "wb") as f:
                pickle.dump(tile_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size = tile_file.stat().st_size

            # Update database index
            self.conn.execute(
                """
                INSERT OR REPLACE INTO tiles
                (geohash, precision, min_lat, min_lon, max_lat, max_lon,
                 node_count, edge_count, created_at, last_accessed,
                 file_size_bytes, semantic_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    geohash,
                    self.default_precision,
                    min_lat,
                    min_lon,
                    max_lat,
                    max_lon,
                    len(graph.nodes),
                    len(graph.edges),
                    time.time(),
                    time.time(),
                    file_size,
                    json.dumps(list(semantic_features)),
                ),
            )
            self.conn.commit()

            logger.debug(
                f"Stored tile {geohash}: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
            )
            return tile_file

        except Exception as e:
            logger.error(f"Failed to store tile {geohash}: {e}")
            return None

    def _tile_exists(self, geohash: str) -> bool:
        """Check if a tile with the given geohash exists."""
        tile_file = self.storage_dir / f"tile_{geohash}.pickle"
        return tile_file.exists()

    def _load_tile_graph(self, geohash: str) -> nx.MultiGraph | None:
        """Load a single tile graph from disk."""
        tile_file = self.storage_dir / f"tile_{geohash}.pickle"

        if not tile_file.exists():
            return None

        try:
            with open(tile_file, "rb") as f:
                tile_data = pickle.load(f)

            return tile_data["graph"]

        except Exception as e:
            logger.error(f"Failed to load tile {geohash}: {e}")
            return None

    def _update_tile_access(self, geohash: str):
        """Update last accessed time for a tile."""
        self.conn.execute(
            """
            UPDATE tiles SET last_accessed = ? WHERE geohash = ?
        """,
            (time.time(), geohash),
        )
        self.conn.commit()

    def _log_stats(self):
        """Log storage statistics."""
        cursor = self.conn.execute("SELECT COUNT(*), SUM(file_size_bytes) FROM tiles")
        count, total_size = cursor.fetchone()

        if count and total_size:
            logger.info(f"Tile storage: {count} tiles, {total_size / 1024 / 1024:.1f} MB total")
        else:
            logger.info("Tile storage: empty")

    def precompute_city_tiles(
        self, city_coords: list[tuple[float, float, str]], radius_m: float = 15000
    ):
        """
        Precompute tiles for major cities.

        Args:
            city_coords: List of (lat, lon, city_name) tuples
            radius_m: Radius to precompute around each city
        """
        logger.info(f"Starting precomputation for {len(city_coords)} cities")

        for lat, lon, city_name in city_coords:
            logger.info(f"Precomputing tiles for {city_name} ({lat:.4f}, {lon:.4f})")

            # Check if tiles already exist
            existing_tiles = self.get_covering_tiles(lat, lon, radius_m)
            if existing_tiles:
                logger.info(f"  {city_name}: {len(existing_tiles)} tiles already exist")
                continue

            # This would integrate with the existing graph loading system
            logger.info(f"  {city_name}: needs {self._estimate_tiles_needed(radius_m)} tiles")

    def _estimate_tiles_needed(self, radius_m: float) -> int:
        """Estimate number of tiles needed for a radius."""
        # Rough calculation based on geohash tile size
        tile_size_m = 1200  # Approximate size of precision-6 geohash tile
        area_tiles = (radius_m * 2) ** 2 / (tile_size_m**2)
        return int(area_tiles * 1.5)  # Add buffer for coverage

    def cleanup_old_tiles(self, max_age_days: int = 90):
        """Clean up tiles that haven't been accessed recently."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        cursor = self.conn.execute(
            """
            SELECT geohash FROM tiles WHERE last_accessed < ?
        """,
            (cutoff_time,),
        )

        old_tiles = [row[0] for row in cursor.fetchall()]

        for geohash in old_tiles:
            tile_file = self.storage_dir / f"tile_{geohash}.pickle"
            if tile_file.exists():
                tile_file.unlink()

            self.conn.execute("DELETE FROM tiles WHERE geohash = ?", (geohash,))

        self.conn.commit()
        logger.info(f"Cleaned up {len(old_tiles)} old tiles")

    def get_storage_stats(self) -> dict:
        """Get comprehensive storage statistics."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as tile_count,
                SUM(file_size_bytes) as total_size,
                SUM(node_count) as total_nodes,
                SUM(edge_count) as total_edges,
                MIN(created_at) as oldest_tile,
                MAX(last_accessed) as most_recent_access
            FROM tiles
        """)

        stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone(), strict=False))

        # Add precision breakdown
        cursor = self.conn.execute("""
            SELECT precision, COUNT(*) FROM tiles GROUP BY precision
        """)
        stats["precision_breakdown"] = dict(cursor.fetchall())

        return stats


# Integration with existing system
def create_spatial_tile_storage() -> SpatialTileStorage:
    """Create spatial tile storage instance."""
    return SpatialTileStorage()


# Major cities for precomputation
MAJOR_CITIES = [
    (37.7749, -122.4194, "San Francisco"),
    (40.7128, -74.0060, "New York"),
    (34.0522, -118.2437, "Los Angeles"),
    (41.8781, -87.6298, "Chicago"),
    (29.7604, -95.3698, "Houston"),
    (33.4484, -112.0740, "Phoenix"),
    (39.9526, -75.1652, "Philadelphia"),
    (29.4241, -98.4936, "San Antonio"),
    (32.7767, -96.7970, "Dallas"),
    (37.2712, -121.8058, "San Jose"),
    # International cities
    (51.5074, -0.1278, "London"),
    (48.8566, 2.3522, "Paris"),
    (52.5200, 13.4050, "Berlin"),
    (35.6762, 139.6503, "Tokyo"),
    (1.3521, 103.8198, "Singapore"),
]
