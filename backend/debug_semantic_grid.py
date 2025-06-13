"""
Debug tool for visualizing semantic grid and POI data.
Creates an HTML visualization showing grid cells, POIs, and semantic scores.
"""

from pathlib import Path

import folium
import osmnx as ox
from core.semantic_grid import SemanticGrid
from core.semantic_matcher import SemanticMatcher


def create_debug_visualization(lat: float, lon: float, radius: float = 800, output_file: str = "semantic_debug.html"):
    """
    Create an interactive map showing:
    1. Grid cells with their semantic scores and features
    2. Individual POIs
    3. Graph nodes and edges
    4. Color-coded by semantic strength
    """
    print(f"Creating debug visualization for ({lat:.6f}, {lon:.6f})...")

    # Load graph and build semantic grid
    G = ox.graph_from_point((lat, lon), dist=radius, network_type='walk')
    print(f"Loaded graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    grid = SemanticGrid(cell_size_meters=150)
    sm = SemanticMatcher()
    grid.build_grid(G, sm)

    print(f"Built semantic grid: {len(grid.grid)} cells")

    # Create folium map
    m = folium.Map(location=[lat, lon], zoom_start=15)

    # Add POIs to map
    if grid.poi_enricher and grid.poi_enricher.pois:
        print(f"Visualizing {len(grid.poi_enricher.pois)} POIs...")

        poi_colors = {
            'water': 'blue',
            'park': 'green',
            'nature': 'darkgreen',
            'cafe': 'orange',
            'restaurant': 'red',
            'shop': 'purple',
            'historic': 'brown',
            'scenic': 'pink'
        }

        for poi in grid.poi_enricher.pois:
            # Choose color based on primary feature
            color = 'gray'
            for feature in poi['features']:
                if feature in poi_colors:
                    color = poi_colors[feature]
                    break

            folium.CircleMarker(
                location=[poi['lat'], poi['lon']],
                radius=5,
                popup=f"<b>{poi['name'] or 'Unnamed'}</b><br>Features: {', '.join(poi['features'])}",
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)

    # Add grid cells to map
    print("Visualizing grid cells...")
    test_preferences = ['parks and nature', 'cafes and restaurants', 'shops and shopping']

    for i, ((_grid_x, _grid_y), cell) in enumerate(grid.grid.items()):
        # Calculate cell bounds
        cell_size_degrees_lat = grid._meters_to_lat_degrees(grid.cell_size_meters)
        cell_size_degrees_lon = grid._meters_to_lon_degrees(grid.cell_size_meters, cell.lat)

        min_lat = cell.lat - cell_size_degrees_lat / 2
        max_lat = cell.lat + cell_size_degrees_lat / 2
        min_lon = cell.lon - cell_size_degrees_lon / 2
        max_lon = cell.lon + cell_size_degrees_lon / 2

        # Test semantic scores for different preferences
        scores = {}
        explanations = {}
        for pref in test_preferences:
            score, explanation = grid.get_semantic_score(cell.lat, cell.lon, pref)
            scores[pref] = score
            explanations[pref] = explanation

        # Get POI features for this cell
        poi_features = []
        if grid.poi_enricher:
            poi_features = grid.poi_enricher.get_nearby_features(cell.lat, cell.lon, grid.cell_size_meters / 2)

        # Choose color based on best score
        max_score = max(scores.values()) if scores else 0.5
        if max_score > 0.8:
            color = 'green'
            opacity = 0.8
        elif max_score > 0.6:
            color = 'yellow'
            opacity = 0.6
        elif max_score > 0.4:
            color = 'orange'
            opacity = 0.4
        else:
            color = 'red'
            opacity = 0.3

        # Create detailed popup
        popup_html = f"""
        <div style="width: 300px;">
            <h4>Grid Cell {i+1}</h4>
            <p><b>Location:</b> ({cell.lat:.6f}, {cell.lon:.6f})</p>
            <h5>Cell Features:</h5>
            <ul>
            {"".join(f"<li>{feat}: {score:.2f}</li>" for feat, score in sorted(cell.feature_scores.items(), key=lambda x: x[1], reverse=True)[:8])}
            </ul>
            <h5>POI Features Nearby:</h5>
            <p>{", ".join(poi_features) if poi_features else "None"}</p>
            <h5>Semantic Scores:</h5>
            <ul>
            {"".join(f"<li><b>{pref}:</b> {scores[pref]:.2f} - {explanations[pref]}</li>" for pref in test_preferences)}
            </ul>
            <h5>Dominant Features:</h5>
            <p>{", ".join(cell.dominant_features)}</p>
        </div>
        """

        # Add cell rectangle
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            popup=folium.Popup(popup_html, max_width=400),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=opacity
        ).add_to(m)

        # Add cell center marker with score
        folium.Marker(
            location=[cell.lat, cell.lon],
            popup=popup_html,
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10px; color: black; font-weight: bold;">{max_score:.2f}</div>',
                icon_size=(30, 20),
                icon_anchor=(15, 10)
            )
        ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                top: 10px; right: 10px; width: 200px; height: 200px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <h4>Semantic Grid Debug</h4>
    <p><b>Grid Cells:</b></p>
    <p>🟢 Green: High semantic score (>0.8)</p>
    <p>🟡 Yellow: Medium score (0.6-0.8)</p>
    <p>🟠 Orange: Low score (0.4-0.6)</p>
    <p>🔴 Red: Very low score (<0.4)</p>
    <p><b>POI Markers:</b></p>
    <p>🔵 Water 🟢 Parks 🟠 Cafes</p>
    <p>🔴 Restaurants 🟣 Shops 🟤 Historic</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    output_path = Path(output_file)
    m.save(str(output_path))
    print(f"Debug visualization saved to: {output_path.absolute()}")

    # Print summary statistics
    print("\n=== DEBUG SUMMARY ===")
    print(f"Total POIs: {len(grid.poi_enricher.pois) if grid.poi_enricher else 0}")
    if grid.poi_enricher:
        stats = grid.poi_enricher.get_stats()
        print(f"POI features: {stats['features']}")

    print(f"Grid cells: {len(grid.grid)}")
    print("Sample cell analysis:")
    for i, ((_grid_x, _grid_y), cell) in enumerate(list(grid.grid.items())[:3]):
        print(f"\nCell {i+1}:")
        print(f"  Features: {dict(sorted(cell.feature_scores.items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"  Dominant: {cell.dominant_features}")

        # Test if POI features are being detected
        poi_features = grid.poi_enricher.get_nearby_features(cell.lat, cell.lon, grid.cell_size_meters / 2) if grid.poi_enricher else []
        print(f"  POI features: {poi_features}")

        # Test semantic scoring
        score, explanation = grid.get_semantic_score(cell.lat, cell.lon, "cafes and restaurants")
        print(f"  Cafe score: {score:.2f} - {explanation}")

    return str(output_path.absolute())


if __name__ == "__main__":
    # Test with San Francisco
    output_file = create_debug_visualization(37.7749, -122.4194, 600)
    print(f"\nOpen this file in your browser: {output_file}")
