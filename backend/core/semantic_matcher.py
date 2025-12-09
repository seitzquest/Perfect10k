"""
Semantic matching for route preferences.
Enhanced semantic matching that translates natural language preferences into route values.
"""

from typing import ClassVar


class SemanticMatcher:
    """Enhanced semantic matching for route preferences."""

    PREFERENCE_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "nature": ["park", "forest", "green", "tree", "garden", "nature", "woods", "trail"],
        "water": ["river", "lake", "pond", "stream", "water", "creek", "canal", "waterfront"],
        "quiet": ["residential", "quiet", "peaceful", "path", "footway", "calm", "serene"],
        "scenic": ["scenic", "view", "beautiful", "hill", "overlook", "vista", "landscape"],
        "urban": ["street", "avenue", "downtown", "city", "commercial", "shopping"],
        "historic": ["historic", "old", "heritage", "monument", "church", "castle"],
        "exercise": ["fitness", "running", "jogging", "workout", "steep", "hill", "climb"],
    }

    @staticmethod
    def create_value_function(preference: str) -> dict[str, float]:
        """Create sophisticated value function based on preferences."""
        preference_lower = preference.lower()

        # Enhanced base values for different path types
        values = {
            "path": 0.8,
            "footway": 0.9,
            "track": 0.7,
            "steps": 0.5,
            "cycleway": 0.6,
            "bridleway": 0.7,
            "pedestrian": 0.8,
            "living_street": 0.6,
            "residential": 0.4,
            "tertiary": 0.3,
            "secondary": 0.2,
            "primary": 0.1,
            "default": 0.3,
        }

        # Apply preference-based boosts
        for category, keywords in SemanticMatcher.PREFERENCE_KEYWORDS.items():
            boost_factor = sum(1 for keyword in keywords if keyword in preference_lower)
            boost_factor = min(boost_factor, 3) * 0.1  # Cap boost at 0.3

            if category == "nature" and boost_factor > 0:
                values["path"] += boost_factor * 2
                values["footway"] += boost_factor * 1.5
                values["track"] += boost_factor * 2.5
                values["bridleway"] += boost_factor * 1.5
            elif category == "water" and boost_factor > 0:
                values["path"] += boost_factor * 1.5
                values["footway"] += boost_factor * 1.2
            elif category == "quiet" and boost_factor > 0:
                values["footway"] += boost_factor * 2
                values["path"] += boost_factor * 1.8
                values["residential"] += boost_factor * 1.2
            elif category == "scenic" and boost_factor > 0:
                values["track"] += boost_factor * 2
                values["path"] += boost_factor * 1.5
                values["steps"] += boost_factor * 1.5
            elif category == "urban" and boost_factor > 0:
                values["pedestrian"] += boost_factor * 1.5
                values["cycleway"] += boost_factor * 1.2
                values["living_street"] += boost_factor * 1.3
            elif category == "exercise" and boost_factor > 0:
                values["steps"] += boost_factor * 2
                values["track"] += boost_factor * 1.2

        # Normalize to [0, 1] range
        max_val = max(values.values())
        if max_val > 1:
            values = {k: min(v / max_val, 1.0) for k, v in values.items()}

        return values

    def calculate_node_value(
        self, node_id: int, value_function: dict[str, float], graph=None
    ) -> tuple[float, str]:
        """
        Calculate the value score for a node based on OSM data and preferences.
        Returns (score, explanation) tuple.
        """
        if graph is None:
            return 0.5, "No map data available"

        try:
            node_data = graph.nodes[node_id]
            explanations = []

            # Start with base score
            base_score = 0.3

            # Check node's own tags for relevant features
            node_score, node_reasons = self._score_node_tags_with_explanation(
                node_data, value_function
            )
            explanations.extend(node_reasons)

            # Check connected edges for path types and features
            edge_score, edge_reasons = self._score_connected_edges_with_explanation(
                graph, node_id, value_function
            )
            explanations.extend(edge_reasons)

            # Check for nearby points of interest
            poi_score, poi_reasons = self._score_nearby_features_with_explanation(
                graph, node_id, value_function
            )
            explanations.extend(poi_reasons)

            # Combine scores with weights
            final_score = (
                base_score * 0.2  # Base walkability
                + node_score * 0.3  # Node features
                + edge_score * 0.4  # Path quality
                + poi_score * 0.1  # Nearby POIs
            )

            # Ensure score is in [0, 1] range
            final_score = max(0.0, min(1.0, final_score))

            # Create explanation (remove duplicates while preserving order)
            if explanations:
                unique_explanations = []
                for exp in explanations:
                    if exp not in unique_explanations:
                        unique_explanations.append(exp)
                explanation = ", ".join(unique_explanations)
            else:
                explanation = "Basic walkable area"

            return final_score, explanation

        except Exception:
            return 0.5, "Unable to analyze location"

    def _score_node_tags(self, node_data: dict, value_function: dict[str, float]) -> float:
        """Score a node based on its OSM tags."""
        score = 0.0

        # Check if node has interesting tags
        for tag_key, tag_value in node_data.items():
            if isinstance(tag_value, str):
                tag_text = f"{tag_key} {tag_value}".lower()

                # Look for water features
                if any(
                    keyword in tag_text
                    for keyword in ["water", "river", "lake", "pond", "stream", "fountain"]
                ):
                    score += 0.8

                # Look for nature features
                if any(
                    keyword in tag_text
                    for keyword in ["park", "garden", "tree", "forest", "green", "nature"]
                ):
                    score += 0.7

                # Look for scenic features
                if any(
                    keyword in tag_text
                    for keyword in ["viewpoint", "scenic", "vista", "overlook", "monument"]
                ):
                    score += 0.6

                # Look for amenities
                if any(
                    keyword in tag_text for keyword in ["shop", "cafe", "restaurant", "amenity"]
                ):
                    score += 0.4

        return min(score, 1.0)

    def _score_node_tags_with_explanation(
        self, node_data: dict, value_function: dict[str, float]
    ) -> tuple[float, list[str]]:
        """Score a node based on its OSM tags with explanations."""
        score = 0.0
        reasons = []

        # Check if node has interesting tags
        for tag_key, tag_value in node_data.items():
            if isinstance(tag_value, str):
                tag_text = f"{tag_key} {tag_value}".lower()

                # Look for water features
                water_keywords = ["water", "river", "lake", "pond", "stream", "fountain"]
                found_water = [kw for kw in water_keywords if kw in tag_text]
                if found_water:
                    score += 0.8
                    reasons.append(f"near {found_water[0]}")

                # Look for nature features
                nature_keywords = ["park", "garden", "tree", "forest", "green", "nature"]
                found_nature = [kw for kw in nature_keywords if kw in tag_text]
                if found_nature:
                    score += 0.7
                    reasons.append(f"in {found_nature[0]} area")

                # Look for scenic features
                scenic_keywords = ["viewpoint", "scenic", "vista", "overlook", "monument"]
                found_scenic = [kw for kw in scenic_keywords if kw in tag_text]
                if found_scenic:
                    score += 0.6
                    reasons.append(f"scenic {found_scenic[0]}")

                # Look for amenities
                amenity_keywords = ["shop", "cafe", "restaurant", "amenity"]
                found_amenity = [kw for kw in amenity_keywords if kw in tag_text]
                if found_amenity:
                    score += 0.4
                    reasons.append(f"near {found_amenity[0]}")

        return min(score, 1.0), reasons

    def _score_connected_edges(
        self, graph, node_id: int, value_function: dict[str, float]
    ) -> float:
        """Score based on the types of paths connected to this node."""
        edge_scores = []

        try:
            # Get all edges connected to this node
            for neighbor in graph.neighbors(node_id):
                edge_data = graph[node_id][neighbor]

                # Handle multiple edges between same nodes
                for _edge_key, edge_attrs in edge_data.items():
                    highway_type = edge_attrs.get("highway", "default")

                    # Get score from value function
                    path_score = value_function.get(
                        highway_type, value_function.get("default", 0.3)
                    )
                    edge_scores.append(path_score)

                    # Bonus for named paths (often more interesting)
                    if edge_attrs.get("name"):
                        edge_scores.append(path_score + 0.1)
        except Exception:
            pass

        # Return average score of connected edges
        return sum(edge_scores) / len(edge_scores) if edge_scores else 0.3

    def _score_connected_edges_with_explanation(
        self, graph, node_id: int, value_function: dict[str, float]
    ) -> tuple[float, list[str]]:
        """Score based on the types of paths connected to this node with explanations."""
        edge_scores = []
        reasons = []
        path_types = []

        try:
            # Get all edges connected to this node
            for neighbor in graph.neighbors(node_id):
                edge_data = graph[node_id][neighbor]

                # Handle multiple edges between same nodes
                for _edge_key, edge_attrs in edge_data.items():
                    highway_type = edge_attrs.get("highway", "default")
                    path_types.append(highway_type)

                    # Get score from value function
                    path_score = value_function.get(
                        highway_type, value_function.get("default", 0.3)
                    )
                    edge_scores.append(path_score)

                    # Bonus for named paths (often more interesting)
                    if edge_attrs.get("name"):
                        edge_scores.append(path_score + 0.1)
                        reasons.append(f"named {highway_type}")
        except Exception:
            pass

        # Add reasons for high-quality path types
        if path_types:
            best_paths = [
                p
                for p in path_types
                if p in ["footway", "path", "track"] and value_function.get(p, 0) > 0.6
            ]
            if best_paths:
                path_type = best_paths[0].replace("way", " path")
                reasons.append(f"good {path_type}")

        # Return average score of connected edges
        avg_score = sum(edge_scores) / len(edge_scores) if edge_scores else 0.3
        return avg_score, reasons

    def _score_nearby_features(
        self, graph, node_id: int, value_function: dict[str, float]
    ) -> float:
        """Score based on nearby interesting features within walking distance."""
        try:
            node_data = graph.nodes[node_id]
            node_lat, node_lon = node_data["y"], node_data["x"]

            nearby_score = 0.0
            checked_nodes = 0

            # Check nodes within ~200m radius (rough estimate)
            for other_node, other_data in graph.nodes(data=True):
                if other_node == node_id:
                    continue

                other_lat, other_lon = other_data["y"], other_data["x"]

                # Simple distance check (rough approximation)
                lat_diff = abs(node_lat - other_lat)
                lon_diff = abs(node_lon - other_lon)

                # ~200m in degrees (very rough)
                if lat_diff < 0.002 and lon_diff < 0.002:
                    feature_score = self._score_node_tags(other_data, value_function)
                    nearby_score += feature_score * 0.1  # Reduced weight for nearby features
                    checked_nodes += 1

                    # Don't check too many nodes for performance
                    if checked_nodes > 20:
                        break

            return min(nearby_score, 0.5)  # Cap nearby influence

        except Exception:
            return 0.0

    def _score_nearby_features_with_explanation(
        self, graph, node_id: int, value_function: dict[str, float]
    ) -> tuple[float, list[str]]:
        """Score based on nearby interesting features with explanations."""
        try:
            node_data = graph.nodes[node_id]
            node_lat, node_lon = node_data["y"], node_data["x"]

            nearby_score = 0.0
            checked_nodes = 0
            reasons = []

            # Check nodes within ~200m radius (rough estimate)
            for other_node, other_data in graph.nodes(data=True):
                if other_node == node_id:
                    continue

                other_lat, other_lon = other_data["y"], other_data["x"]

                # Simple distance check (rough approximation)
                lat_diff = abs(node_lat - other_lat)
                lon_diff = abs(node_lon - other_lon)

                # ~200m in degrees (very rough)
                if lat_diff < 0.002 and lon_diff < 0.002:
                    feature_score, feature_reasons = self._score_node_tags_with_explanation(
                        other_data, value_function
                    )
                    if feature_score > 0.1 and feature_reasons:
                        nearby_score += feature_score * 0.1  # Reduced weight for nearby features
                        reasons.extend(
                            [f"nearby {reason}" for reason in feature_reasons[:1]]
                        )  # Limit to avoid spam
                    checked_nodes += 1

                    # Don't check too many nodes for performance
                    if checked_nodes > 20:
                        break

            return min(nearby_score, 0.5), reasons  # Cap nearby influence

        except Exception:
            return 0.0, []
