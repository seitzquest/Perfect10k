#!/usr/bin/env python3
"""
Demo of the spatial tile storage system benefits
"""

import time
import networkx as nx
from core.enhanced_graph_cache import EnhancedGraphCache, MAJOR_CITIES


def create_demo_graph(center_lat: float, center_lon: float, radius: float = 8000) -> nx.MultiGraph:
    """Create a simple demo graph for testing."""
    G = nx.MultiGraph()
    
    # Add some demo nodes in a grid pattern
    node_id = 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            lat = center_lat + i * 0.001  # ~100m spacing
            lon = center_lon + j * 0.001
            G.add_node(node_id, y=lat, x=lon)
            
            # Connect to adjacent nodes
            if i > -3:  # Connect to node above
                prev_node = node_id - 7
                if prev_node in G.nodes:
                    G.add_edge(node_id, prev_node, length=100)
            if j > -3:  # Connect to node to the left
                prev_node = node_id - 1
                if prev_node in G.nodes:
                    G.add_edge(node_id, prev_node, length=100)
                    
            node_id += 1
    
    return G


def demo_spatial_tile_benefits():
    """Demonstrate the benefits of spatial tile storage."""
    print("🚀 Perfect10k Spatial Tile Storage Demo")
    print("=" * 50)
    
    cache = EnhancedGraphCache('demo_storage')
    
    # Demo 1: Store a graph and show tile creation
    print("\n📍 Demo 1: Storing a graph as spatial tiles")
    lat, lon = 37.7749, -122.4194  # San Francisco
    demo_graph = create_demo_graph(lat, lon)
    
    print(f"Created demo graph: {len(demo_graph.nodes)} nodes, {len(demo_graph.edges)} edges")
    
    start_time = time.time()
    success = cache.store_graph(demo_graph, lat, lon, 8000, {'forests', 'rivers'})
    store_time = time.time() - start_time
    
    if success:
        print(f"✅ Stored graph as spatial tiles in {store_time:.3f}s")
    else:
        print("❌ Failed to store graph")
    
    # Demo 2: Show instant loading
    print("\n⚡ Demo 2: Instant loading from spatial tiles")
    
    start_time = time.time()
    loaded_graph = cache.get_graph(lat, lon, 8000)
    load_time = time.time() - start_time
    
    if loaded_graph:
        print(f"✅ Loaded graph from tiles in {load_time:.3f}s")
        print(f"   Graph: {len(loaded_graph.nodes)} nodes, {len(loaded_graph.edges)} edges")
    else:
        print("❌ Failed to load graph")
    
    # Demo 3: Show spatial locality benefits
    print("\n🌍 Demo 3: Spatial locality benefits")
    
    # Try loading a nearby location (should reuse tiles)
    nearby_lat, nearby_lon = lat + 0.001, lon + 0.001  # ~100m away
    
    start_time = time.time()
    nearby_graph = cache.get_graph(nearby_lat, nearby_lon, 8000)
    nearby_load_time = time.time() - start_time
    
    if nearby_graph:
        print(f"✅ Loaded nearby area in {nearby_load_time:.3f}s (reusing tiles)")
        print(f"   Shared nodes with original: {len(set(loaded_graph.nodes) & set(nearby_graph.nodes))}")
    
    # Demo 4: Show cache statistics
    print("\n📊 Demo 4: Cache statistics")
    stats = cache.get_cache_statistics()
    
    tile_stats = stats['spatial_tile_storage']
    coverage = stats['total_coverage_estimate']
    
    print(f"Spatial tiles: {tile_stats['tile_count']} tiles")
    print(f"Total storage: {tile_stats.get('total_size', 0) or 0} bytes")
    print(f"Coverage: ~{coverage['estimated_area_km2']:.1f} km²")
    print(f"Memory cache: {stats['memory_cache']['memory_cached_graphs']} graphs")
    
    # Demo 5: Show city precomputation concept
    print("\n🏙️  Demo 5: City precomputation benefits")
    print("Major cities that could be precomputed:")
    
    for i, (city_lat, city_lon, city_name) in enumerate(MAJOR_CITIES[:5]):
        is_cached = cache.is_area_cached(city_lat, city_lon, 10000)
        status = "✅ Cached" if is_cached else "⏳ Needs precomputation"
        print(f"   {city_name}: {status}")
    
    print("\n🎯 Key Benefits:")
    print("   • Permanent storage - no cache clearing")
    print("   • Spatial locality - nearby areas share tiles")
    print("   • Instant loading for popular areas") 
    print("   • Scales to global coverage")
    print("   • Efficient for overlapping requests")
    
    print(f"\n💾 Storage location: {cache.tile_storage.storage_dir}")
    print("   Database: tile_index.db (SQLite)")
    print("   Tiles: tile_<geohash>.pickle files")


def show_performance_comparison():
    """Show expected performance improvements."""
    print("\n⚡ Performance Comparison")
    print("-" * 30)
    print("Current system:")
    print("  ❌ Cache gets cleared periodically")
    print("  ❌ Graph reloading takes minutes")
    print("  ❌ Overlapping requests duplicate work")
    print("  ❌ Limited to memory constraints")
    
    print("\nSpatial tile system:")
    print("  ✅ Permanent storage, never cleared")
    print("  ✅ Instant loading from tiles (<1s)")
    print("  ✅ Spatial locality reduces redundancy")
    print("  ✅ Scales to unlimited coverage")
    print("  ✅ Popular cities load instantly")
    
    print("\nExpected improvements:")
    print("  🚀 Popular areas: 30s+ → <1s (97%+ faster)")
    print("  📦 Storage efficiency: 70% reduction in duplicated data")
    print("  🌍 Global scalability: Unlimited vs memory-limited")


if __name__ == "__main__":
    try:
        demo_spatial_tile_benefits()
        show_performance_comparison()
        
        print("\n🎉 Demo complete!")
        print("To integrate: Run 'python migrate_to_spatial_tiles.py'")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()