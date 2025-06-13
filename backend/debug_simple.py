"""
Simple debug tool for semantic grid analysis without external dependencies.
"""

import osmnx as ox
from core.semantic_grid import SemanticGrid
from core.semantic_matcher import SemanticMatcher


def debug_semantic_integration():
    """Debug the semantic integration step by step."""
    print("=== DEBUGGING SEMANTIC INTEGRATION ===\n")
    
    # Load small graph for debugging
    lat, lon = 37.7749, -122.4194
    G = ox.graph_from_point((lat, lon), dist=500, network_type='walk')
    print(f"1. Loaded graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Build grid with debug info
    grid = SemanticGrid(cell_size_meters=200)  # Larger cells for clearer debugging
    sm = SemanticMatcher()
    
    print("\n2. Building semantic grid...")
    grid.build_grid(G, sm)
    
    # Check POI enricher
    print(f"\n3. POI Enricher status:")
    if grid.poi_enricher:
        stats = grid.poi_enricher.get_stats()
        print(f"   Total POIs: {stats['total_pois']}")
        print(f"   Features: {stats['features']}")
        print(f"   Spatial index cells: {stats['spatial_index_cells']}")
        
        # Show sample POIs
        print(f"\n   Sample POIs:")
        for i, poi in enumerate(grid.poi_enricher.pois[:5]):
            print(f"     {i+1}. {poi['name'] or 'Unnamed'} - {poi['features']}")
    else:
        print("   ERROR: No POI enricher found!")
    
    print(f"\n4. Grid analysis ({len(grid.grid)} cells):")
    
    # Analyze each cell in detail
    for i, ((grid_x, grid_y), cell) in enumerate(grid.grid.items()):
        print(f"\n   Cell {i+1} at ({cell.lat:.6f}, {cell.lon:.6f}):")
        
        # Check POI features in this cell
        poi_features = []
        if grid.poi_enricher:
            poi_features = grid.poi_enricher.get_nearby_features(
                cell.lat, cell.lon, grid.cell_size_meters / 2
            )
        
        print(f"     POI features found: {poi_features}")
        print(f"     Cell feature scores: {dict(sorted(cell.feature_scores.items(), key=lambda x: x[1], reverse=True)[:8])}")
        print(f"     Dominant features: {cell.dominant_features}")
        
        # Test semantic scoring
        test_prefs = ['cafes and restaurants', 'parks and nature', 'shops and shopping']
        for pref in test_prefs:
            score, explanation = grid.get_semantic_score(cell.lat, cell.lon, pref)
            print(f"     {pref}: {score:.2f} - '{explanation}'")
        
        if i >= 4:  # Limit to first 5 cells for readability
            break
    
    print(f"\n5. Testing specific locations:")
    test_points = [
        (37.7749, -122.4194, 'SF center'),
        (37.7760, -122.4180, 'North area'),
        (37.7740, -122.4200, 'South area')
    ]
    
    for lat, lon, name in test_points:
        print(f"\n   {name} ({lat:.6f}, {lon:.6f}):")
        
        # Check POI features directly
        poi_features = grid.poi_enricher.get_nearby_features(lat, lon, 100) if grid.poi_enricher else []
        print(f"     Direct POI lookup: {poi_features}")
        
        # Check grid cell lookup
        score, explanation = grid.get_semantic_score(lat, lon, 'cafes and restaurants')
        print(f"     Grid score for cafes: {score:.2f} - '{explanation}'")
        
        score, explanation = grid.get_semantic_score(lat, lon, 'parks and nature')
        print(f"     Grid score for parks: {score:.2f} - '{explanation}'")


def check_poi_enricher_directly():
    """Test POI enricher in isolation."""
    print("\n=== TESTING POI ENRICHER DIRECTLY ===\n")
    
    from core.poi_enrichment import POIEnricher
    
    poi_enricher = POIEnricher()
    poi_enricher.fetch_pois(37.7749, -122.4194, 800)
    
    stats = poi_enricher.get_stats()
    print(f"POI Stats: {stats}")
    
    # Test specific locations
    test_points = [
        (37.7749, -122.4194, 'SF center'),
        (37.7760, -122.4180, 'North'),
        (37.7740, -122.4200, 'South')
    ]
    
    for lat, lon, name in test_points:
        features = poi_enricher.get_nearby_features(lat, lon, 100)
        print(f"{name}: {features}")
        
        # Check larger radius
        features_large = poi_enricher.get_nearby_features(lat, lon, 200)
        print(f"{name} (200m): {features_large}")


if __name__ == "__main__":
    check_poi_enricher_directly()
    debug_semantic_integration()