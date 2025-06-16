#!/usr/bin/env python3
"""
Migration script to convert existing cache to spatial tile storage
and precompute tiles for major cities.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Set

import networkx as nx
from loguru import logger

from core.enhanced_graph_cache import EnhancedGraphCache, MAJOR_CITIES
from core.graph_cache import PersistentGraphCache


def migrate_existing_cache():
    """Migrate existing persistent cache to spatial tile storage."""
    logger.info("Starting migration from old cache to spatial tile storage")
    
    # Initialize both systems
    old_cache = PersistentGraphCache()
    new_cache = EnhancedGraphCache()
    
    migrated_count = 0
    error_count = 0
    
    # Get all existing cache entries
    if hasattr(old_cache, 'cache_index') and old_cache.cache_index:
        logger.info(f"Found {len(old_cache.cache_index)} entries in old cache")
        
        for cache_key, cache_info in old_cache.cache_index.items():
            try:
                # Load the old graph
                graph = old_cache._load_cached_graph(cache_key)
                
                if graph and len(graph.nodes) > 0:
                    # Extract metadata
                    center_lat = cache_info.get('center_lat')
                    center_lon = cache_info.get('center_lon') 
                    radius = cache_info.get('radius', 8000)
                    
                    if center_lat and center_lon:
                        # Store in new spatial tile system
                        success = new_cache.store_graph(
                            graph, center_lat, center_lon, radius
                        )
                        
                        if success:
                            migrated_count += 1
                            logger.info(f"Migrated cache entry {cache_key}: "
                                      f"({center_lat:.6f}, {center_lon:.6f}), "
                                      f"{len(graph.nodes)} nodes")
                        else:
                            error_count += 1
                            logger.warning(f"Failed to migrate cache entry {cache_key}")
                    else:
                        logger.warning(f"Missing coordinates for cache entry {cache_key}")
                        error_count += 1
                else:
                    logger.warning(f"Empty or invalid graph for cache entry {cache_key}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error migrating cache entry {cache_key}: {e}")
                error_count += 1
    
    logger.info(f"Migration complete: {migrated_count} success, {error_count} errors")
    
    # Show new cache statistics
    stats = new_cache.get_cache_statistics()
    logger.info(f"New spatial tile storage: {stats}")
    
    return migrated_count, error_count


def precompute_city_tiles(max_cities: int = 10):
    """Precompute tiles for major cities by triggering graph loading."""
    logger.info(f"Precomputing tiles for {max_cities} major cities")
    
    cache = EnhancedGraphCache()
    
    # Import the graph loading function
    try:
        from interactive_router import InteractiveRouteBuilder
        router = InteractiveRouteBuilder()
    except Exception as e:
        logger.error(f"Cannot import InteractiveRouteBuilder: {e}")
        return 0
    
    successful_precomputes = 0
    
    for i, (lat, lon, city_name) in enumerate(MAJOR_CITIES[:max_cities]):
        logger.info(f"Precomputing {city_name} ({i+1}/{max_cities})")
        
        try:
            # Check if already cached
            if cache.is_area_cached(lat, lon, 10000):  # 10km radius
                logger.info(f"  {city_name}: already cached")
                successful_precomputes += 1
                continue
            
            # Load graph (this will automatically cache it in spatial tiles)
            start_time = time.time()
            
            # Use the router's graph loading method
            graph = router._get_cached_graph(lat, lon, 10000)
            
            if graph and len(graph.nodes) > 0:
                load_time = time.time() - start_time
                logger.info(f"  {city_name}: loaded and cached {len(graph.nodes)} nodes "
                          f"in {load_time:.1f}s")
                successful_precomputes += 1
            else:
                logger.warning(f"  {city_name}: failed to load graph")
                
        except Exception as e:
            logger.error(f"  {city_name}: error during precomputation: {e}")
    
    logger.info(f"Precomputation complete: {successful_precomputes}/{max_cities} cities")
    return successful_precomputes


def update_router_to_use_spatial_tiles():
    """Update InteractiveRouteBuilder to use the new spatial tile system."""
    logger.info("Note: To complete migration, update InteractiveRouteBuilder")
    logger.info("Replace the PersistentGraphCache with EnhancedGraphCache in:")
    logger.info("  - interactive_router.py line 72")
    logger.info("  - Change: self.persistent_cache = PersistentGraphCache(cache_dir)")
    logger.info("  - To: self.persistent_cache = EnhancedGraphCache(cache_dir)")


def show_performance_comparison():
    """Show expected performance improvements."""
    cache = EnhancedGraphCache()
    stats = cache.get_cache_statistics()
    
    logger.info("Performance improvements with spatial tile storage:")
    logger.info("  ✅ Permanent storage - no cache clearing")
    logger.info("  ✅ Spatial locality - nearby areas share tiles") 
    logger.info("  ✅ Instant loading for cached areas")
    logger.info("  ✅ Scalable to global coverage")
    logger.info("  ✅ Efficient for overlapping requests")
    
    tile_count = stats['spatial_tile_storage'].get('tile_count', 0)
    if tile_count > 0:
        coverage = stats['total_coverage_estimate']
        logger.info(f"Current coverage: {tile_count} tiles, "
                   f"~{coverage['estimated_area_km2']:.0f} km², "
                   f"~{coverage['estimated_cities_covered']:.1f} cities")


if __name__ == "__main__":
    logger.info("Perfect10k Spatial Tile Migration")
    logger.info("=" * 50)
    
    # Step 1: Migrate existing cache
    logger.info("Step 1: Migrating existing cache...")
    migrated, errors = migrate_existing_cache()
    
    # Step 2: Precompute major cities
    logger.info("\nStep 2: Precomputing major cities...")
    precomputed = precompute_city_tiles(5)  # Start with 5 cities
    
    # Step 3: Show results and next steps
    logger.info("\nStep 3: Migration summary")
    show_performance_comparison()
    
    logger.info("\nNext steps:")
    logger.info("1. Update router to use EnhancedGraphCache")
    logger.info("2. Run precomputation for more cities")
    logger.info("3. Monitor performance improvements")
    
    logger.info(f"\nMigration complete! Migrated: {migrated}, Precomputed: {precomputed}")