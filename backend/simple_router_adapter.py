"""
Simple Router Adapter for Raspberry Pi Integration
=================================================

Adapter class that bridges the SimpleCandidateGenerator with the existing
InteractiveRouter interface, allowing seamless replacement of the complex
multi-layered optimization system with a simple, Pi-optimized approach.
"""

import time
from typing import Any

from loguru import logger
from simple_candidate_generator import SimpleCandidate, SimpleCandidateGenerator


class RouteCandidate:
    """Adapter class matching the existing RouteCandidate interface"""
    def __init__(self, simple_candidate: SimpleCandidate, explanation: str = ""):
        self.node_id = simple_candidate.node_id  # Already an integer from the graph
        self.lat = simple_candidate.lat
        self.lon = simple_candidate.lon
        self.value_score = simple_candidate.feature_score
        self.distance_from_current = simple_candidate.distance_from_current
        self.estimated_route_completion = 0.0  # Not used in simple algorithm
        self.explanation = explanation or f"Feature score: {simple_candidate.feature_score:.2f}"
        self.semantic_scores = {"simple_score": simple_candidate.feature_score}
        self.semantic_details = explanation


class SimpleCandidateGeneratorAdapter:
    """
    Adapter that makes SimpleCandidateGenerator compatible with the existing
    InteractiveRouter interface while maintaining Pi optimization.
    """

    def __init__(self, semantic_overlay_manager=None):
        """
        Initialize the adapter.
        
        Args:
            semantic_overlay_manager: Optional semantic overlay manager (not used in simple algorithm)
        """
        self.generator = SimpleCandidateGenerator()
        self.semantic_overlay_manager = semantic_overlay_manager
        self._features_cache = {}
        self._cache_timestamp = 0
        self._stats = {
            'generation_count': 0,
            'total_generation_time': 0.0,
            'avg_generation_time': 0.0
        }

        logger.info("Initialized SimpleCandidateGeneratorAdapter for Raspberry Pi optimization")

    def precompute_area_scores_fast(self, graph, center_lat, center_lon, radius_m, preference, **kwargs):
        """
        Precomputation adapter - simplified for Pi performance.
        
        The simple algorithm doesn't require heavy precomputation, so this
        just sets up basic preference parsing and caches geographical features.
        """
        start_time = time.time()

        # Set preferences in the simple generator
        self.generator.set_preferences(preference)

        # Optionally cache some basic geographical features if semantic overlay manager exists
        if self.semantic_overlay_manager:
            try:
                # Get geographical features in a simple format
                features = self._get_simple_features(center_lat, center_lon, radius_m)
                self._features_cache = features
                self._cache_timestamp = time.time()
            except Exception as e:
                logger.warning(f"Failed to cache geographical features: {e}")
                self._features_cache = {}

        precompute_time = time.time() - start_time
        logger.debug(f"Simple precomputation completed in {precompute_time:.3f}s")

        return True  # Success indicator

    def precompute_area_scores(self, graph, center_lat, center_lon, radius_m, preference, **kwargs):
        """Alias for precompute_area_scores_fast (simple algorithm doesn't differentiate)"""
        return self.precompute_area_scores_fast(graph, center_lat, center_lon, radius_m, preference, **kwargs)

    def generate_candidates_ultra_fast(
        self,
        graph,
        current_node_id,
        current_lat,
        current_lon,
        previous_lat=None,
        previous_lon=None,
        target_distance=8000,
        **kwargs
    ) -> list[RouteCandidate]:
        """
        Generate candidates using the simple algorithm.
        
        This is the main integration point that replaces the complex
        multi-layered optimization with simple, reliable candidate selection.
        """
        start_time = time.time()

        try:
            # Generate candidates using the simple algorithm
            simple_candidates = self.generator.generate_candidates(
                graph=graph,
                current_lat=current_lat,
                current_lon=current_lon,
                previous_lat=previous_lat,
                previous_lon=previous_lon,
                target_distance=target_distance,
                features_data=self._features_cache
            )

            # Convert to RouteCandidate format expected by InteractiveRouter
            route_candidates = []
            for simple_candidate in simple_candidates:
                explanation = self._generate_explanation(simple_candidate)
                route_candidate = RouteCandidate(simple_candidate, explanation)
                route_candidates.append(route_candidate)

            generation_time = time.time() - start_time

            # Update statistics
            self._stats['generation_count'] += 1
            self._stats['total_generation_time'] += generation_time
            self._stats['avg_generation_time'] = (
                self._stats['total_generation_time'] / self._stats['generation_count']
            )

            logger.debug(
                f"Generated {len(route_candidates)} candidates in {generation_time:.3f}s "
                f"(avg: {self._stats['avg_generation_time']:.3f}s)"
            )

            return route_candidates

        except Exception as e:
            logger.error(f"Simple candidate generation failed: {e}")
            return []

    def generate_candidates_fast(self, *args, **kwargs) -> list[RouteCandidate]:
        """Alias for generate_candidates_ultra_fast (simple algorithm doesn't differentiate)"""
        return self.generate_candidates_ultra_fast(*args, **kwargs)

    def clear_cache(self):
        """Clear any cached data"""
        self._features_cache = {}
        self._cache_timestamp = 0
        logger.debug("Simple generator cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics"""
        return {
            **self._stats,
            'cache_size': len(self._features_cache),
            'cache_age': time.time() - self._cache_timestamp if self._cache_timestamp > 0 else 0
        }

    def _get_simple_features(self, center_lat: float, center_lon: float, radius_m: float) -> dict:
        """
        Extract geographical features in a simple format optimized for Pi performance.
        
        Returns a dict like:
        {
            'forests': [{'lat': lat1, 'lon': lon1}, ...],
            'rivers': [{'lat': lat2, 'lon': lon2}, ...],
            'lakes': [{'lat': lat3, 'lon': lon3}, ...]
        }
        """
        if not self.semantic_overlay_manager:
            return {}

        try:
            features = {}

            # Get semantic overlays in a simple format
            for feature_type in ['forests', 'rivers', 'lakes']:
                try:
                    # Try to get feature locations from the semantic overlay manager
                    # This is a simplified extraction - adapt based on your semantic overlay format
                    overlay_data = getattr(self.semantic_overlay_manager, f'get_{feature_type}', None)
                    if overlay_data:
                        # Convert to simple lat/lon list
                        feature_locations = []
                        # Add extraction logic based on your semantic overlay format
                        # For now, return empty list to prevent errors
                        features[feature_type] = feature_locations
                    else:
                        features[feature_type] = []

                except Exception as e:
                    logger.debug(f"Could not extract {feature_type}: {e}")
                    features[feature_type] = []

            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {}

    def _generate_explanation(self, candidate: SimpleCandidate) -> str:
        """Generate human-readable explanation for the candidate"""
        score = candidate.feature_score
        distance = candidate.distance_from_current

        if score > 0.7:
            quality = "Excellent"
        elif score > 0.4:
            quality = "Good"
        elif score > 0.2:
            quality = "Fair"
        else:
            quality = "Basic"

        return f"{quality} location ({distance:.0f}m away, score: {score:.2f})"


# Factory function for easy integration
def create_simple_adapter(semantic_overlay_manager=None) -> SimpleCandidateGeneratorAdapter:
    """Create a SimpleCandidateGeneratorAdapter instance"""
    return SimpleCandidateGeneratorAdapter(semantic_overlay_manager)
