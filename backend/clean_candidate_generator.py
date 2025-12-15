"""
Clean Candidate Generator for Perfect10k
Unified, fast candidate generator using spatial indexing and interpretable scoring.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

from backend.feature_database import FeatureDatabase
from backend.interpretable_scorer import InterpretableScorer, ScoredCandidate
from backend.smart_cache_manager import cache_manager
from backend.spatial_grid import SpatialGrid


@dataclass
class CandidateGenerationResult:
    """Result from candidate generation."""

    candidates: list[ScoredCandidate]
    generation_time_ms: float
    search_stats: dict[str, Any]
    preference_analysis: dict[str, Any]


class CleanCandidateGenerator:
    """
    Clean, fast candidate generator for interactive route building.

    Combines spatial indexing, discrete features, and interpretable scoring
    for sub-50ms candidate retrieval with clear explanations.
    """

    def __init__(self, graph: nx.MultiGraph, semantic_overlay_manager=None, area_key=None):
        """
        Initialize the clean candidate generator.

        Args:
            graph: NetworkX graph with node/edge data
            semantic_overlay_manager: Optional semantic overlay manager for nature features
            area_key: Optional area key for cache persistence
        """
        self.graph = graph
        self.semantic_overlay_manager = semantic_overlay_manager
        self.area_key = area_key or "default"

        # Core components - will be loaded from cache if available
        self.spatial_grid = None
        self.feature_database = None
        self.scorer = InterpretableScorer()

        # Performance parameters
        self.target_generation_time_ms = 50.0
        self.max_candidates_to_score = 50  # Limit for performance
        self.min_distance_between_candidates = 100.0  # meters

        # Probabilistic selection parameters for variety
        self.enable_probabilistic_selection = True
        self.variety_factor = 0.3  # 30% randomness, 70% score-based
        self.min_score_threshold = 0.1  # Only consider candidates above this score

        # Score visualization tracking
        self.scored_nodes = {}  # node_id -> ScoredCandidate for visualization
        self.max_scored_nodes = 1000  # Limit to prevent memory issues

        # Initialization status
        self.is_initialized = False

        logger.info(f"Initialized clean candidate generator for area {self.area_key}")

    def initialize(self) -> bool:
        """
        Initialize spatial grid and compute features with cache support.

        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            logger.info("Clean candidate generator already initialized")
            return True

        start_time = time.time()
        logger.info(f"Initializing clean candidate generator for area {self.area_key}...")

        try:
            # Try to load from cache first
            cached_grid = cache_manager.get_spatial_grid(self.area_key)
            cached_cell_features = cache_manager.get_cell_features(self.area_key)

            if cached_grid and cached_cell_features:
                logger.info("Loading spatial grid and features from cache")
                self.spatial_grid = cached_grid
                self.feature_database = FeatureDatabase(self.spatial_grid)
                self.feature_database.cell_features = cached_cell_features

                cache_time = time.time() - start_time
                logger.info(f"Loaded from cache in {cache_time:.3f}s")

                self.is_initialized = True
                return True

            # Cache miss - build from scratch
            logger.info("Cache miss - building spatial grid and features from scratch")

            # Build spatial grid from graph
            self.spatial_grid = SpatialGrid(cell_size_meters=200.0)
            self.feature_database = FeatureDatabase(self.spatial_grid)

            self.spatial_grid.add_nodes_from_graph(self.graph)
            grid_time = time.time() - start_time

            # Compute features for all cells
            feature_start = time.time()
            cells_processed = self.feature_database.compute_features_for_area(
                self.graph, self.semantic_overlay_manager
            )
            feature_time = time.time() - feature_start

            # Store in cache for next time
            cache_start = time.time()
            cache_manager.store_spatial_grid(self.area_key, self.spatial_grid)
            cache_manager.store_cell_features(self.area_key, self.feature_database.cell_features)
            cache_time = time.time() - cache_start

            total_time = time.time() - start_time

            logger.info(f"Initialization complete in {total_time:.2f}s:")
            logger.info(f"  - Spatial grid: {grid_time:.2f}s")
            logger.info(f"  - Features: {feature_time:.2f}s ({cells_processed} cells)")
            logger.info(f"  - Cache storage: {cache_time:.3f}s")

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize clean candidate generator: {e}")
            return False

    def generate_candidates(
        self,
        from_lat: float,
        from_lon: float,
        target_distance: float,
        preference: str,
        exclude_nodes: list[int] | None = None,
        existing_route_nodes: list[int] | None = None,
    ) -> CandidateGenerationResult:
        """
        Generate candidate waypoints for route building.

        Args:
            from_lat: Starting latitude
            from_lon: Starting longitude
            target_distance: Target distance to candidates in meters
            preference: User preference text
            exclude_nodes: List of node IDs to exclude
            existing_route_nodes: List of nodes in current route (for proximity filtering)

        Returns:
            CandidateGenerationResult with scored candidates and metadata
        """
        start_time = time.time()

        if not self.is_initialized and not self.initialize():
            return CandidateGenerationResult(
                candidates=[],
                generation_time_ms=0.0,
                search_stats={"error": "Initialization failed"},
                preference_analysis={},
            )

        exclude_nodes_set = set(exclude_nodes or [])
        existing_route = existing_route_nodes or []

        logger.info(
            f"Generating candidates from ({from_lat:.6f}, {from_lon:.6f}) "
            f"with target distance {target_distance}m, excluding {len(exclude_nodes_set)} nodes"
        )

        # Step 1: Find candidate nodes in target distance range
        candidate_nodes = self._find_candidate_nodes(
            from_lat, from_lon, target_distance, exclude_nodes_set, existing_route
        )

        logger.info(f"Found {len(candidate_nodes)} candidate nodes in target area")

        if not candidate_nodes:
            logger.warning(
                f"No candidate nodes found! Target distance: {target_distance}m, "
                f"Exclude nodes: {len(exclude_nodes_set)}, Route nodes: {len(existing_route)}"
            )
            return CandidateGenerationResult(
                candidates=[],
                generation_time_ms=(time.time() - start_time) * 1000,
                search_stats={"error": "No candidate nodes found in target area"},
                preference_analysis={},
            )

        # Step 2: Get features for candidate nodes
        candidates_with_features = self._get_candidate_features(candidate_nodes)

        if not candidates_with_features:
            return CandidateGenerationResult(
                candidates=[],
                generation_time_ms=(time.time() - start_time) * 1000,
                search_stats={"error": "No features available for candidate nodes"},
                preference_analysis={},
            )

        # Step 3: Score candidates using interpretable scorer
        scored_candidates = self._score_candidates(candidates_with_features, preference)

        # Step 4: Apply probabilistic selection and geometric diversity
        if self.enable_probabilistic_selection:
            diverse_candidates = self._apply_probabilistic_selection(
                scored_candidates, from_lat, from_lon, max_candidates=3
            )
        else:
            diverse_candidates = self._apply_geometric_diversity(
                scored_candidates, from_lat, from_lon, max_candidates=3
            )

        generation_time_ms = (time.time() - start_time) * 1000

        # Analyze preferences for debugging/explanation
        preference_analysis = self.scorer.explain_preferences(preference)

        # Compile search statistics
        search_stats = {
            "initial_candidate_nodes": len(candidate_nodes),
            "nodes_with_features": len(candidates_with_features),
            "scored_candidates": len(scored_candidates),
            "final_candidates": len(diverse_candidates),
            "target_distance": target_distance,
            "search_area_km2": self._calculate_search_area(target_distance),
            "performance_target_ms": self.target_generation_time_ms,
            "actual_time_ms": generation_time_ms,
            "performance_ok": generation_time_ms <= self.target_generation_time_ms,
        }

        logger.info(f"Generated {len(diverse_candidates)} candidates in {generation_time_ms:.1f}ms")

        return CandidateGenerationResult(
            candidates=diverse_candidates,
            generation_time_ms=generation_time_ms,
            search_stats=search_stats,
            preference_analysis=preference_analysis,
        )

    def _find_candidate_nodes(
        self,
        from_lat: float,
        from_lon: float,
        target_distance: float,
        exclude_nodes: set[int],
        existing_route_nodes: list[int],
    ) -> list[tuple[int, float, float]]:
        """Find nodes within target distance range."""
        # Search in an annulus: target_distance ± 50%
        min_distance = target_distance * 0.5
        max_distance = target_distance * 1.5

        # Get nodes in search area using spatial grid
        if self.spatial_grid is None:
            return []
        nearby_nodes = self.spatial_grid.get_nearby_nodes(from_lat, from_lon, max_distance)

        logger.info(
            f"Spatial grid returned {len(nearby_nodes)} nearby nodes within {max_distance}m"
        )
        logger.info(f"Distance filter: {min_distance}m to {max_distance}m")

        candidate_nodes = []

        # Prepare route nodes for proximity checking
        route_node_coords = []
        if existing_route_nodes:
            for route_node_id in existing_route_nodes:
                if route_node_id in self.graph.nodes:
                    route_data = self.graph.nodes[route_node_id]
                    route_node_coords.append((route_data["y"], route_data["x"]))

        # Adaptive minimum route distance based on search area
        # When search area is small (close to target), use smaller minimum distance
        min_route_distance = max(50.0, max_distance * 0.3) if max_distance <= 500 else 300.0

        logger.info(
            f"Using min_route_distance: {min_route_distance:.0f}m for search area: {max_distance:.0f}m"
        )

        excluded_count = 0
        missing_count = 0
        distance_filtered = 0
        route_proximity_filtered = 0

        for node_id in nearby_nodes:
            if node_id in exclude_nodes:
                excluded_count += 1
                continue

            if node_id not in self.graph.nodes:
                missing_count += 1
                continue

            # Get node coordinates
            node_data = self.graph.nodes[node_id]
            node_lat, node_lon = node_data["y"], node_data["x"]

            # Check distance from current position
            distance = self._haversine_distance(from_lat, from_lon, node_lat, node_lon)

            if not (min_distance <= distance <= max_distance):
                distance_filtered += 1
                continue

            # Check proximity to existing route nodes
            too_close_to_route = False
            for route_lat, route_lon in route_node_coords:
                route_distance = self._haversine_distance(node_lat, node_lon, route_lat, route_lon)
                if route_distance < min_route_distance:
                    too_close_to_route = True
                    break

            if too_close_to_route:
                route_proximity_filtered += 1
            else:
                candidate_nodes.append((node_id, node_lat, node_lon))

        # Log filtering summary
        logger.info("Candidate filtering summary:")
        logger.info(f"  Total nearby nodes: {len(nearby_nodes)}")
        logger.info(f"  Excluded nodes: {excluded_count}")
        logger.info(f"  Missing from graph: {missing_count}")
        logger.info(f"  Distance filtered: {distance_filtered}")
        logger.info(f"  Route proximity filtered: {route_proximity_filtered}")
        logger.info(f"  Final candidates: {len(candidate_nodes)}")

        # If no candidates found due to distance filtering, try without distance filter
        if len(candidate_nodes) == 0 and distance_filtered > 0:
            logger.warning(
                f"No candidates with distance filter ({min_distance:.0f}m-{max_distance:.0f}m), retrying without distance filter"
            )

            candidate_nodes = []
            distance_filtered = 0

            for node_id in nearby_nodes:
                if node_id in exclude_nodes:
                    continue

                if node_id not in self.graph.nodes:
                    continue

                # Get node coordinates
                node_data = self.graph.nodes[node_id]
                node_lat, node_lon = node_data["y"], node_data["x"]

                # Skip distance check - accept any node in search radius

                # Check proximity to existing route nodes
                too_close_to_route = False
                for route_lat, route_lon in route_node_coords:
                    route_distance = self._haversine_distance(
                        node_lat, node_lon, route_lat, route_lon
                    )
                    if route_distance < min_route_distance:
                        too_close_to_route = True
                        break

                if not too_close_to_route:
                    candidate_nodes.append((node_id, node_lat, node_lon))

            logger.info(f"Without distance filter: found {len(candidate_nodes)} candidates")

        # If we filtered out too many candidates, relax the route proximity constraint
        if len(candidate_nodes) < 5 and existing_route_nodes and len(route_node_coords) > 0:
            logger.warning(
                f"Only {len(candidate_nodes)} candidates found, relaxing route proximity filter"
            )
            relaxed_min_distance = min_route_distance * 0.5  # Reduce minimum distance by half

            # Re-scan with relaxed constraints
            for node_id in nearby_nodes:
                if node_id in exclude_nodes or node_id not in self.graph.nodes:
                    continue

                node_data = self.graph.nodes[node_id]
                node_lat, node_lon = node_data["y"], node_data["x"]
                distance = self._haversine_distance(from_lat, from_lon, node_lat, node_lon)

                if not (min_distance <= distance <= max_distance):
                    continue

                # Check with relaxed proximity
                too_close_to_route = False
                for route_lat, route_lon in route_node_coords:
                    route_distance = self._haversine_distance(
                        node_lat, node_lon, route_lat, route_lon
                    )
                    if route_distance < relaxed_min_distance:
                        too_close_to_route = True
                        break

                if not too_close_to_route and (node_id, node_lat, node_lon) not in candidate_nodes:
                    candidate_nodes.append((node_id, node_lat, node_lon))

        # Limit candidates for performance
        if len(candidate_nodes) > self.max_candidates_to_score:
            # Take a representative sample
            step = len(candidate_nodes) // self.max_candidates_to_score
            candidate_nodes = candidate_nodes[::step][: self.max_candidates_to_score]

        logger.debug(
            f"Found {len(candidate_nodes)} candidate nodes in distance range "
            f"{min_distance:.0f}-{max_distance:.0f}m"
        )

        return candidate_nodes

    def _get_candidate_features(
        self, candidate_nodes: list[tuple[int, float, float]]
    ) -> list[tuple[tuple[int, float, float], Any]]:
        """Get pre-computed features for candidate nodes."""
        if self.feature_database is None:
            return []

        candidates_with_features = []

        for candidate in candidate_nodes:
            _node_id, lat, lon = candidate
            features = self.feature_database.get_cell_features(lat, lon)

            if features:
                candidates_with_features.append((candidate, features))

        logger.debug(f"Retrieved features for {len(candidates_with_features)} candidates")
        return candidates_with_features

    def _score_candidates(
        self, candidates_with_features: list[tuple[tuple[int, float, float], Any]], preference: str
    ) -> list[ScoredCandidate]:
        """Score candidates using interpretable scorer."""
        candidates = [(node_id, lat, lon) for (node_id, lat, lon), _ in candidates_with_features]
        features_list = [features for _, features in candidates_with_features]

        scored_candidates = self.scorer.score_multiple_candidates(
            candidates, features_list, preference
        )

        # Track scored nodes for visualization (limit to prevent memory issues)
        for candidate in scored_candidates:
            self.scored_nodes[candidate.node_id] = candidate

            # Keep only the most recent scored nodes
            if len(self.scored_nodes) > self.max_scored_nodes:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.scored_nodes.keys())[: -self.max_scored_nodes]
                for old_key in oldest_keys:
                    del self.scored_nodes[old_key]

        logger.debug(
            f"Scored {len(scored_candidates)} candidates, tracking {len(self.scored_nodes)} total"
        )
        return scored_candidates

    def _apply_probabilistic_selection(
        self,
        scored_candidates: list[ScoredCandidate],
        from_lat: float,
        from_lon: float,
        max_candidates: int = 3,
    ) -> list[ScoredCandidate]:
        """Apply probabilistic selection for variety while maintaining performance."""
        if len(scored_candidates) <= max_candidates:
            return scored_candidates

        # Step 1: Filter candidates by score threshold
        scores = [candidate.overall_score for candidate in scored_candidates]
        if scores:
            score_threshold = max(self.min_score_threshold, min(scores))
            qualified_candidates = [
                candidate
                for candidate in scored_candidates
                if candidate.overall_score >= score_threshold
            ]
        else:
            qualified_candidates = scored_candidates

        # Step 2: Create probability weights (higher score = higher probability)
        if not qualified_candidates:
            return scored_candidates[:max_candidates]

        scores = [candidate.overall_score for candidate in qualified_candidates]
        min_score = min(scores)
        max_score = max(scores)

        # Normalize scores to 0-1 range and apply variety factor
        if max_score > min_score:
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        else:
            normalized_scores = [1.0] * len(scores)

        # Mix deterministic (score-based) and random probabilities
        random_weights = [random.random() for _ in range(len(qualified_candidates))]
        final_weights = [
            (1 - self.variety_factor) * score + self.variety_factor * rand
            for score, rand in zip(normalized_scores, random_weights, strict=False)
        ]

        # Step 3: Sample candidates based on weights with geometric diversity
        selected = []
        available_candidates = list(enumerate(qualified_candidates))
        available_weights = final_weights.copy()

        min_angle_separation = 45  # degrees

        while len(selected) < max_candidates and available_candidates:
            # Weighted random selection
            if sum(available_weights) > 0:
                idx = np.random.choice(
                    len(available_candidates),
                    p=np.array(available_weights) / sum(available_weights),
                )
            else:
                idx = random.randint(0, len(available_candidates) - 1)

            _candidate_idx, candidate = available_candidates[idx]

            # Check angular separation from already selected candidates
            if not selected:
                selected.append(candidate)
            else:
                candidate_angle = self._calculate_bearing(
                    from_lat, from_lon, candidate.lat, candidate.lon
                )

                too_close = False
                for selected_candidate in selected:
                    selected_angle = self._calculate_bearing(
                        from_lat, from_lon, selected_candidate.lat, selected_candidate.lon
                    )
                    angle_diff = abs(candidate_angle - selected_angle)
                    angle_diff = min(angle_diff, 360 - angle_diff)

                    if angle_diff < min_angle_separation:
                        too_close = True
                        break

                if not too_close:
                    selected.append(candidate)

            # Remove selected candidate from available pool
            available_candidates.pop(idx)
            available_weights.pop(idx)

            # If we can't find enough diverse candidates, relax the angle constraint
            if (
                len(available_candidates) > 0
                and len(selected) < max_candidates
                and len(available_candidates) < max_candidates - len(selected)
            ):
                min_angle_separation = max(20, min_angle_separation - 10)

        logger.debug(
            f"Probabilistic selection: {len(selected)} candidates from {len(qualified_candidates)} qualified"
        )
        return selected

    def _apply_geometric_diversity(
        self,
        scored_candidates: list[ScoredCandidate],
        from_lat: float,
        from_lon: float,
        max_candidates: int = 3,
    ) -> list[ScoredCandidate]:
        """Apply geometric diversity to avoid clustering candidates."""
        if len(scored_candidates) <= max_candidates:
            return scored_candidates

        selected = []
        min_angle_separation = 60  # degrees - ensure good directional spread

        # Always include the highest scoring candidate
        if scored_candidates:
            selected.append(scored_candidates[0])

        # Select additional candidates with good angular separation
        for candidate in scored_candidates[1:]:
            if len(selected) >= max_candidates:
                break

            candidate_angle = self._calculate_bearing(
                from_lat, from_lon, candidate.lat, candidate.lon
            )

            # Check angular separation from existing candidates
            too_close = False
            for selected_candidate in selected:
                selected_angle = self._calculate_bearing(
                    from_lat, from_lon, selected_candidate.lat, selected_candidate.lon
                )

                angle_diff = abs(candidate_angle - selected_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wraparound

                if angle_diff < min_angle_separation:
                    too_close = True
                    break

            if not too_close:
                selected.append(candidate)

        logger.debug(f"Applied geometric diversity: {len(selected)} final candidates")
        return selected

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing in degrees from point 1 to point 2."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
            lat2_rad
        ) * math.cos(dlon_rad)

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)

        return (bearing_deg + 360) % 360

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters."""
        earth_radius_m = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat_rad = math.radians(lat2 - lat1)
        dlon_rad = math.radians(lon2 - lon1)

        a = math.sin(dlat_rad / 2) * math.sin(dlat_rad / 2) + math.cos(lat1_rad) * math.cos(
            lat2_rad
        ) * math.sin(dlon_rad / 2) * math.sin(dlon_rad / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return earth_radius_m * c

    def get_scored_nodes_for_visualization(self, score_type: str = "overall") -> dict[str, Any]:
        """
        Export scored nodes for semantic overlay visualization.

        Args:
            score_type: Type of score to visualize ("overall", "nature", "water", etc.)

        Returns:
            Dictionary with nodes and their scores for map visualization
        """
        if not self.scored_nodes:
            return {
                "nodes": [],
                "score_range": {"min": 0, "max": 1},
                "total_nodes": 0,
                "score_type": score_type,
            }

        visualization_nodes = []
        scores = []

        for node_id, scored_candidate in self.scored_nodes.items():
            # Extract the specific score type
            if score_type == "overall":
                score = scored_candidate.overall_score
            else:
                # Look for feature-specific scores
                feature_score = None
                for feature_type, feature_score_val in scored_candidate.feature_scores.items():
                    if score_type.lower() in feature_type.value.lower():
                        feature_score = feature_score_val
                        break

                if feature_score is None:
                    continue  # Skip nodes without this feature score
                score = feature_score

            visualization_nodes.append(
                {
                    "node_id": node_id,
                    "lat": scored_candidate.lat,
                    "lon": scored_candidate.lon,
                    "score": score,
                    "overall_score": scored_candidate.overall_score,
                    "explanation": scored_candidate.explanation,
                    "feature_scores": {
                        ft.value: fs for ft, fs in scored_candidate.feature_scores.items()
                    },
                }
            )
            scores.append(score)

        # Calculate score range for color mapping
        score_range = {"min": min(scores), "max": max(scores)} if scores else {"min": 0, "max": 1}

        return {
            "nodes": visualization_nodes,
            "score_range": score_range,
            "total_nodes": len(visualization_nodes),
            "score_type": score_type,
            "available_score_types": self._get_available_score_types(),
        }

    def _get_available_score_types(self) -> list[str]:
        """Get list of available score types for visualization."""
        available_types = ["overall"]

        if self.scored_nodes:
            # Get feature types from first scored node
            first_node = next(iter(self.scored_nodes.values()))
            for feature_type in first_node.feature_scores:
                available_types.append(feature_type.value)

        return available_types

    def _calculate_search_area(self, target_distance: float) -> float:
        """Calculate search area in km²."""
        max_radius = target_distance * 1.5
        area_m2 = math.pi * (max_radius**2)
        return area_m2 / 1_000_000  # Convert to km²

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the generator."""
        if not self.is_initialized:
            return {"error": "Generator not initialized"}

        stats: dict[str, Any] = {
            "initialization": {
                "is_initialized": self.is_initialized,
                "graph_nodes": len(self.graph.nodes),
                "graph_edges": len(self.graph.edges),
            },
            "performance_targets": {
                "target_generation_time_ms": self.target_generation_time_ms,
                "max_candidates_to_score": self.max_candidates_to_score,
                "min_distance_between_candidates": self.min_distance_between_candidates,
            },
        }

        if self.spatial_grid:
            stats["spatial_grid"] = self.spatial_grid.get_statistics()
        if self.feature_database:
            stats["feature_database"] = self.feature_database.get_statistics()

        return stats

    def clear_cache(self):
        """Clear all cached data."""
        if self.spatial_grid:
            self.spatial_grid.clear()
        if self.feature_database:
            self.feature_database.clear()
        self.is_initialized = False
        logger.info("Cleared clean candidate generator cache")
