"""
Simple Candidate Selection Algorithm for Raspberry Pi
====================================================

A lightweight, intuitive routing algorithm that prioritizes simplicity and reliability
over complex optimization. Designed specifically for efficient execution on Raspberry Pi.

Key Principles:
- Denavit-Hartenberg-like directional constraints (no backtracking)
- Simple feature-based scoring without heavy computations
- Diversity selection to reduce decision fatigue
- Convex route maintenance for intuitive exploration
"""

import math
from dataclasses import dataclass


@dataclass
class SimpleCandidate:
    """A routing candidate with basic properties"""
    node_id: str
    lat: float
    lon: float
    distance_from_current: float
    feature_score: float
    direction_angle: float  # Angle from current position

class SimpleCandidateGenerator:
    """
    Ultra-simple candidate generator optimized for Raspberry Pi performance.

    This algorithm replaces the complex multi-layered optimization system with
    a straightforward approach that focuses on:
    1. Fast execution (< 0.1s on Raspberry Pi)
    2. Low memory usage
    3. Intuitive route generation
    4. Reliable performance across different hardware
    """

    def __init__(self):
        # Simple feature weights (no complex text embedding)
        self.feature_weights = {
            'forests': 1.0,
            'rivers': 1.0,
            'lakes': 1.0
        }

        # Performance parameters optimized for Pi
        self.max_radius_m = 2000  # Keep search radius reasonable
        self.max_candidates = 100  # Limit processing load
        self.output_candidates = 3  # Simple choice, not overwhelming

        # Directional constraints (Denavit-Hartenberg-like)
        self.min_direction_change = 30  # degrees - avoid tiny direction changes
        self.max_backtrack_angle = 135  # degrees - prevent sharp reversals

        # Feature scoring thresholds (simple distance-based)
        self.feature_radius = {
            'forests': 300,  # meters
            'rivers': 150,
            'lakes': 200
        }

    def set_preferences(self, preference_text: str) -> None:
        """
        Simple preference parsing - no complex NLP or embeddings.
        Just basic keyword matching for Pi efficiency.
        """
        pref_lower = preference_text.lower()

        # Reset weights
        self.feature_weights = {'forests': 1.0, 'rivers': 1.0, 'lakes': 1.0}

        # Simple keyword boost
        if any(word in pref_lower for word in ['forest', 'tree', 'park', 'green', 'nature']):
            self.feature_weights['forests'] = 2.0

        if any(word in pref_lower for word in ['river', 'stream', 'water', 'creek']):
            self.feature_weights['rivers'] = 2.0

        if any(word in pref_lower for word in ['lake', 'pond', 'water']):
            self.feature_weights['lakes'] = 2.0

    def generate_candidates(
        self,
        graph,
        current_lat: float,
        current_lon: float,
        previous_lat: float | None,
        previous_lon: float | None,
        target_distance: int = 8000,
        features_data: dict | None = None
    ) -> list[SimpleCandidate]:
        """
        Generate routing candidates using simple, reliable algorithm.

        Args:
            graph: OSM graph (simplified interface)
            current_lat, current_lon: Current position
            previous_lat, previous_lon: Previous waypoint (for directional constraint)
            target_distance: Desired total route distance
            features_data: Simple dict of geographical features

        Returns:
            List of SimpleCandidate objects
        """

        # Step 1: Get nearby nodes (simple radius search)
        nearby_nodes = self._get_nearby_nodes(graph, current_lat, current_lon, self.max_radius_m)

        if len(nearby_nodes) == 0:
            return []

        # Step 2: Apply directional constraint (Denavit-Hartenberg-like)
        if previous_lat is not None and previous_lon is not None:
            filtered_nodes = self._apply_directional_filter(
                nearby_nodes, current_lat, current_lon, previous_lat, previous_lon
            )
        else:
            filtered_nodes = nearby_nodes

        # Step 3: Score nodes based on features (simple distance-based)
        scored_candidates = []
        for node in filtered_nodes:
            feature_score = self._calculate_feature_score(
                node['lat'], node['lon'], features_data
            )

            # Calculate distance from current position
            distance = self._haversine_simple(
                current_lat, current_lon, node['lat'], node['lon']
            )

            # Calculate direction angle for diversity selection
            direction_angle = self._calculate_angle(
                current_lat, current_lon, node['lat'], node['lon']
            )

            candidate = SimpleCandidate(
                node_id=node['id'],
                lat=node['lat'],
                lon=node['lon'],
                distance_from_current=distance,
                feature_score=feature_score,
                direction_angle=direction_angle
            )
            scored_candidates.append(candidate)

        # Step 4: Select diverse candidates (spread out spatially)
        diverse_candidates = self._select_diverse_candidates(scored_candidates)

        return diverse_candidates

    def _get_nearby_nodes(self, graph, lat: float, lon: float, radius: float) -> list[dict]:
        """
        Simple radius-based node search. No complex spatial indexing.
        """
        nearby_nodes = []

        # Simple iteration through graph nodes (good enough for Pi performance)
        for node_id, node_data in graph.nodes(data=True):
            if 'y' in node_data and 'x' in node_data:
                node_lat, node_lon = node_data['y'], node_data['x']
                distance = self._haversine_simple(lat, lon, node_lat, node_lon)

                if distance <= radius:
                    nearby_nodes.append({
                        'id': node_id,
                        'lat': node_lat,
                        'lon': node_lon,
                        'distance': distance
                    })

                    # Limit candidates to prevent Pi overload
                    if len(nearby_nodes) >= self.max_candidates:
                        break

        return nearby_nodes

    def _apply_directional_filter(
        self,
        nodes: list[dict],
        current_lat: float,
        current_lon: float,
        previous_lat: float,
        previous_lon: float
    ) -> list[dict]:
        """
        Apply Denavit-Hartenberg-like constraint: exclude nodes that would cause
        sharp reversals or minimal direction changes.
        """
        # Calculate the angle of the current direction vector
        current_direction = self._calculate_angle(previous_lat, previous_lon, current_lat, current_lon)

        filtered_nodes = []

        for node in nodes:
            # Calculate angle from current position to candidate node
            candidate_angle = self._calculate_angle(current_lat, current_lon, node['lat'], node['lon'])

            # Calculate angle difference
            angle_diff = abs(candidate_angle - current_direction)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # Filter out nodes that would cause problematic direction changes
            if (angle_diff >= self.min_direction_change and
                angle_diff <= self.max_backtrack_angle):
                filtered_nodes.append(node)

        return filtered_nodes

    def _calculate_feature_score(self, lat: float, lon: float, features_data: dict | None) -> float:
        """
        Simple feature scoring based on proximity to geographical features.
        No complex vectorization or spatial indexing.
        """
        if not features_data:
            return 0.0

        total_score = 0.0

        for feature_type, weight in self.feature_weights.items():
            if feature_type in features_data:
                feature_locations = features_data[feature_type]

                # Find closest feature of this type
                min_distance = float('inf')
                for feature_loc in feature_locations:
                    distance = self._haversine_simple(
                        lat, lon, feature_loc['lat'], feature_loc['lon']
                    )
                    min_distance = min(min_distance, distance)

                # Simple exponential decay scoring
                if min_distance < self.feature_radius[feature_type]:
                    feature_score = math.exp(-min_distance / self.feature_radius[feature_type])
                    total_score += feature_score * weight

        return min(1.0, total_score)  # Normalize to [0, 1]

    def _select_diverse_candidates(self, candidates: list[SimpleCandidate]) -> list[SimpleCandidate]:
        """
        Select diverse candidates to reduce decision fatigue and provide
        intuitive exploration options.
        """
        if len(candidates) <= self.output_candidates:
            return sorted(candidates, key=lambda c: c.feature_score, reverse=True)

        # Sort by feature score first
        candidates.sort(key=lambda c: c.feature_score, reverse=True)

        # Select the top candidate
        selected = [candidates[0]]
        remaining = candidates[1:]

        # Select additional candidates that are spatially diverse
        while len(selected) < self.output_candidates and remaining:
            best_candidate = None
            best_diversity_score = -1

            for candidate in remaining:
                # Calculate minimum angular separation from selected candidates
                min_angle_diff = float('inf')
                for selected_candidate in selected:
                    angle_diff = abs(candidate.direction_angle - selected_candidate.direction_angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    min_angle_diff = min(min_angle_diff, angle_diff)

                # Combine diversity with feature score
                diversity_score = min_angle_diff * 0.7 + candidate.feature_score * 0.3

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _haversine_simple(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Simple haversine distance calculation without numpy vectorization.
        More efficient on Pi's ARM architecture.
        """
        R = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_angle(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate bearing angle between two points in degrees.
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)

        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

        angle = math.atan2(y, x)
        return (math.degrees(angle) + 360) % 360  # Normalize to [0, 360)


# Simple integration interface
def create_simple_generator() -> SimpleCandidateGenerator:
    """Factory function for easy integration"""
    return SimpleCandidateGenerator()
