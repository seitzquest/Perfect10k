"""
Enhanced Graph Cache with Spatial Tile Storage
Integrates the new spatial tile system with existing graph loading.
"""

import time

import networkx as nx
from core.spatial_tile_storage import MAJOR_CITIES, SpatialTileStorage
from loguru import logger


class EnhancedGraphCache:
    """
    Enhanced graph cache that uses spatial tile storage for permanent,
    locality-aware graph storage with instant loading for popular areas.
    """

    def __init__(self, storage_dir: str = "storage/tiles"):
        self.tile_storage = SpatialTileStorage(storage_dir)

        # Memory cache for recently used combined graphs
        self.memory_cache = {}
        self.memory_cache_max_size = 3  # Keep 3 combined graphs in memory

        logger.info("Enhanced graph cache initialized with spatial tile storage")

    def get_graph(self, lat: float, lon: float, radius: float = 8000) -> nx.MultiGraph | None:
        """
        Get a graph covering the requested area.

        Uses spatial tile storage for permanent, fast access.
        Falls back to OSM loading only if tiles don't exist.
        """
        # Create cache key for this request
        cache_key = f"{lat:.4f}_{lon:.4f}_{int(radius)}"

        # Check memory cache first (fastest)
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            entry['last_accessed'] = time.time()
            logger.info(f"Using memory cached graph for ({lat:.6f}, {lon:.6f})")
            return entry['graph']

        # Try to load from spatial tiles (very fast - permanent storage)
        start_time = time.time()
        graph = self.tile_storage.load_graph_for_area(lat, lon, radius)

        if graph:
            load_time = time.time() - start_time
            logger.info(f"Loaded graph from spatial tiles in {load_time:.2f}s ({len(graph.nodes)} nodes)")

            # Cache in memory for immediate reuse
            self._add_to_memory_cache(cache_key, graph)
            return graph

        # No tiles available - need to load from OSM
        logger.info(f"No spatial tiles found for ({lat:.6f}, {lon:.6f}) - OSM loading needed")
        return None

    def store_graph(self, graph: nx.MultiGraph, center_lat: float, center_lon: float,
                   radius: float = 8000, semantic_features: set[str] = None) -> bool:
        """
        Store a graph in the spatial tile system for permanent caching.

        This is called after loading a new graph from OSM to ensure
        it's available for future requests.
        """
        try:
            start_time = time.time()

            # Ensure semantic_features is a set
            if semantic_features is None:
                semantic_features = set()
            elif not isinstance(semantic_features, set):
                # Convert from list or other iterable to set
                try:
                    semantic_features = set(semantic_features) if semantic_features else set()
                except TypeError as e:
                    logger.error(f"Cannot convert semantic_features to set: {e}, type: {type(semantic_features)}, value: {semantic_features}")
                    semantic_features = set()

            logger.debug(f"Storing graph with semantic_features: {semantic_features} (type: {type(semantic_features)})")

            # Store as spatial tiles
            stored_tiles = self.tile_storage.store_graph_tiles(
                graph, center_lat, center_lon, radius, semantic_features
            )

            if stored_tiles:
                store_time = time.time() - start_time
                logger.info(f"Stored graph as {len(stored_tiles)} spatial tiles in {store_time:.2f}s")

                # Also add to memory cache
                cache_key = f"{center_lat:.4f}_{center_lon:.4f}_{int(radius)}"
                self._add_to_memory_cache(cache_key, graph)

                return True
            else:
                logger.warning("Failed to store graph in spatial tiles")
                return False

        except Exception as e:
            logger.error(f"Error storing graph in spatial tiles: {e}")
            return False

    def _add_to_memory_cache(self, cache_key: str, graph: nx.MultiGraph):
        """Add graph to memory cache with LRU eviction."""
        # Remove oldest entry if cache is full
        if len(self.memory_cache) >= self.memory_cache_max_size:
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]['last_accessed']
            )
            del self.memory_cache[oldest_key]

        # Add new entry
        self.memory_cache[cache_key] = {
            'graph': graph,
            'cached_at': time.time(),
            'last_accessed': time.time()
        }

    def precompute_popular_cities(self, max_cities: int = 10):
        """
        Precompute spatial tiles for major cities to enable instant loading.

        This should be run as a background task or during deployment.
        """
        logger.info(f"Starting precomputation for {max_cities} major cities")

        cities_to_process = MAJOR_CITIES[:max_cities]
        successful_precomputes = 0

        for lat, lon, city_name in cities_to_process:
            logger.info(f"Checking precomputation for {city_name}")

            # Check if tiles already exist for this city
            covering_tiles = self.tile_storage.get_covering_tiles(lat, lon, 15000)  # 15km radius
            existing_tiles = [tile for tile in covering_tiles if self.tile_storage._load_tile_graph(tile)]

            if len(existing_tiles) > 0:
                logger.info(f"  {city_name}: {len(existing_tiles)} tiles already exist")
                successful_precomputes += 1
            else:
                logger.info(f"  {city_name}: needs precomputation (would load from OSM)")
                # Note: Actual OSM loading would happen here in a full implementation
                # This would integrate with the existing _load_graph method

        logger.info(f"Precomputation status: {successful_precomputes}/{len(cities_to_process)} cities ready")
        return successful_precomputes

    def get_cache_statistics(self) -> dict:
        """Get comprehensive cache statistics."""
        tile_stats = self.tile_storage.get_storage_stats()

        memory_stats = {
            'memory_cached_graphs': len(self.memory_cache),
            'memory_cache_keys': list(self.memory_cache.keys())
        }

        return {
            'spatial_tile_storage': tile_stats,
            'memory_cache': memory_stats,
            'total_coverage_estimate': self._estimate_total_coverage()
        }

    def _estimate_total_coverage(self) -> dict:
        """Estimate what geographic areas are covered by cached tiles."""
        stats = self.tile_storage.get_storage_stats()
        tile_count = stats.get('tile_count', 0)

        # Each precision-6 geohash covers roughly 1.2km x 0.6km
        estimated_area_km2 = tile_count * 1.2 * 0.6

        return {
            'cached_tiles': tile_count,
            'estimated_area_km2': estimated_area_km2,
            'estimated_cities_covered': estimated_area_km2 / 200  # Rough city size estimate
        }

    def cleanup_cache(self, max_age_days: int = 90):
        """Clean up old cached data."""
        # Clean up spatial tiles
        self.tile_storage.cleanup_old_tiles(max_age_days)

        # Clean up memory cache (keep only recent)
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # 24 hours

        keys_to_remove = [
            key for key, entry in self.memory_cache.items()
            if entry['last_accessed'] < cutoff_time
        ]

        for key in keys_to_remove:
            del self.memory_cache[key]

        logger.info(f"Cleaned up {len(keys_to_remove)} memory cache entries")

    def is_area_cached(self, lat: float, lon: float, radius: float = 8000) -> bool:
        """Check if an area is already cached in spatial tiles."""
        covering_tiles = self.tile_storage.get_covering_tiles(lat, lon, radius)

        # Check if at least some tiles exist
        existing_tiles = 0
        for tile_hash in covering_tiles:
            if self.tile_storage._load_tile_graph(tile_hash):
                existing_tiles += 1

        # Consider area cached if we have at least 50% of needed tiles
        coverage_ratio = existing_tiles / max(1, len(covering_tiles))
        return coverage_ratio >= 0.5


def create_enhanced_graph_cache() -> EnhancedGraphCache:
    """Create enhanced graph cache instance."""
    return EnhancedGraphCache()
