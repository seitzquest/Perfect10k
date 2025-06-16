"""
Fast Semantic Candidate Generator
High-performance implementation with intelligent sampling, spatial indexing, and vectorized scoring.
"""

import math
import time
import random
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import networkx as nx
from functools import lru_cache
from loguru import logger

from optimized_semantic_scoring import (
    OptimizedSemanticScorer, 
    IntelligentNodeSampler,
    create_optimized_semantic_scorer,
    haversine_cached
)


@dataclass
class FastSemanticCandidate:
    """Lightweight candidate with minimal data for fast processing"""
    node_id: int
    lat: float
    lon: float
    semantic_scores: Dict[str, float]
    overall_score: float
    explanation: str
    semantic_details: str


class FastCandidateGenerator:
    """
    Ultra-fast candidate generation using advanced optimization techniques:
    - Intelligent node sampling (reduce from 15k to 1k nodes)
    - Spatial indexing with R-trees
    - Vectorized distance calculations
    - Early termination and smart caching
    """
    
    def __init__(self, semantic_overlay_manager):
        self.semantic_overlay_manager = semantic_overlay_manager
        self.optimized_scorer = create_optimized_semantic_scorer()
        self.node_sampler = IntelligentNodeSampler()
        
        # Probabilistic generation parameters
        self.probabilistic_mode = True  # Enable probabilistic selection
        self.exploration_factor = 0.3  # How much randomness to inject (0.0 = deterministic, 1.0 = random)
        self.diversity_enforcement = True  # Enforce spatial diversity in selection
        
        # Angular diversity parameters
        self.min_angular_separation = 90  # Minimum degrees between candidates (90° = good spread for 3)
        self.distance_threshold = 500  # Minimum meters between candidates
        
        # Caching
        self.precomputed_cache: Dict[str, Dict] = {}
        self.node_selection_cache: Dict[str, Tuple] = {}
        
        # Performance tracking
        self.perf_stats = {
            'total_generations': 0,
            'avg_generation_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'avg_nodes_processed': 0.0,
            'spatial_index_enabled': True
        }
    
    def set_probabilistic_mode(self, enabled: bool, exploration_factor: float = 0.3, diversity_enforcement: bool = True):
        """
        Configure probabilistic generation parameters.
        
        Args:
            enabled: Whether to enable probabilistic selection
            exploration_factor: How much randomness to inject (0.0 = deterministic, 1.0 = random)
            diversity_enforcement: Whether to enforce spatial diversity
        """
        self.probabilistic_mode = enabled
        self.exploration_factor = max(0.0, min(1.0, exploration_factor))  # Clamp to 0-1
        self.diversity_enforcement = diversity_enforcement
        
        logger.info(f"FastGenerator probabilistic mode: {enabled}, exploration: {self.exploration_factor:.2f}, diversity: {diversity_enforcement}")
    
    def set_diversity_parameters(self, min_angular_separation: float = 90, distance_threshold: float = 500):
        """
        Configure spatial and angular diversity parameters.
        
        Args:
            min_angular_separation: Minimum degrees between candidates (90° good for 3 candidates)
            distance_threshold: Minimum meters between candidates
        """
        self.min_angular_separation = max(30, min(180, min_angular_separation))  # Clamp 30-180 degrees
        self.distance_threshold = max(100, distance_threshold)  # At least 100m
        
        logger.info(f"FastGenerator diversity: {self.min_angular_separation:.0f}° angular, {self.distance_threshold:.0f}m distance")
    
    def clear_cache(self):
        """Clear all cached precomputed nodes to ensure fresh probabilistic results."""
        cache_size = len(self.precomputed_cache)
        self.precomputed_cache.clear()
        self.node_selection_cache.clear()
        
        if cache_size > 0:
            logger.info(f"Cleared {cache_size} cached fast precomputed areas for fresh probabilistic generation")
    
    def generate_candidates_ultra_fast_fresh(self, graph: nx.MultiGraph, center_lat: float, center_lon: float,
                                           from_lat: float, from_lon: float, target_radius: float,
                                           preference: str = "scenic nature", exclude_nodes: Set[int] = None,
                                           min_score: float = 0.15, max_candidates: int = 5) -> List[FastSemanticCandidate]:
        """
        Generate fresh candidates without using cache (for maximum probabilistic variety).
        """
        # Temporarily disable cache and force fresh generation
        original_cache = self.precomputed_cache.copy()
        original_node_cache = self.node_selection_cache.copy()
        
        try:
            # Clear caches to force fresh generation
            self.precomputed_cache.clear()
            self.node_selection_cache.clear()
            
            # Generate fresh candidates
            cache_key = self.precompute_area_scores_fast(
                graph, center_lat, center_lon, 
                radius_m=8000, preference=preference, max_nodes=1000
            )
            
            candidates = self.generate_candidates_ultra_fast(
                cache_key, from_lat, from_lon, target_radius, 
                exclude_nodes, min_score, max_candidates
            )
            
            return candidates
            
        finally:
            # Restore original caches (but keep the fresh generation for this session)
            pass  # Leave fresh cache in place
    
    def get_probabilistic_status(self) -> Dict:
        """Get current probabilistic configuration for debugging."""
        return {
            "probabilistic_mode": self.probabilistic_mode,
            "exploration_factor": self.exploration_factor,
            "diversity_enforcement": self.diversity_enforcement,
            "min_angular_separation_degrees": self.min_angular_separation,
            "distance_threshold_meters": self.distance_threshold,
            "cache_size": len(self.precomputed_cache),
            "node_cache_size": len(self.node_selection_cache)
        }
    
    def precompute_area_scores_fast(self, graph: nx.MultiGraph, center_lat: float, center_lon: float, 
                                   radius_m: float = 8000, preference: str = "scenic nature",
                                   max_nodes: int = 1000) -> str:
        """
        Fast precomputation using intelligent sampling and optimized scoring.
        
        Target: Complete in 0.5-2 seconds instead of 10-30 seconds.
        """
        # Add randomization to cache key if probabilistic mode is enabled
        cache_suffix = ""
        if self.probabilistic_mode:
            # Use a smaller random component to allow some cache reuse but still provide variety
            cache_suffix = f"_{random.randint(1, 5)}"  # 5 different probabilistic variants
        
        cache_key = f"fast_{center_lat:.4f}_{center_lon:.4f}_{radius_m}_{hash(preference) % 10000}{cache_suffix}"
        
        if cache_key in self.precomputed_cache:
            logger.info(f"Using cached fast precomputation: {cache_key}")
            return cache_key
        
        start_time = time.time()
        logger.info(f"Fast precomputing semantic scores for area (target: {max_nodes} nodes)")
        
        # Phase 1: Intelligent node sampling (0.1-0.5s)
        sampling_start = time.time()
        node_locations, node_ids = self._intelligent_node_selection(
            graph, center_lat, center_lon, radius_m, max_nodes
        )
        sampling_time = time.time() - sampling_start
        
        logger.info(f"Sampled {len(node_locations)} strategic nodes in {sampling_time:.3f}s")
        
        # Phase 2: Load semantic features if needed (0.1-1s for cached, 10-30s for new areas)
        features_start = time.time()
        self._ensure_semantic_features_loaded(center_lat, center_lon, radius_m)
        features_time = time.time() - features_start
        
        # Phase 3: Fast batch scoring using optimization (0.1-0.5s)
        scoring_start = time.time()
        semantic_scores_list = self._batch_score_optimized(node_locations)
        scoring_time = time.time() - scoring_start
        
        logger.info(f"Batch scored {len(node_locations)} nodes in {scoring_time:.3f}s")
        
        # Phase 4: Convert and cache results (0.1s)
        conversion_start = time.time()
        precomputed_candidates = self._convert_to_candidates(
            node_ids, node_locations, semantic_scores_list, preference
        )
        conversion_time = time.time() - conversion_start
        
        # Cache results
        cache_data = {
            'candidates': precomputed_candidates,
            'bounds': self._calculate_bounds(node_locations),
            'created_at': time.time(),
            'node_count': len(precomputed_candidates)
        }
        
        self.precomputed_cache[cache_key] = cache_data
        
        logger.info(f"Cached {len(precomputed_candidates)} precomputed candidates with scores")
        if precomputed_candidates:
            scores = [c.overall_score for c in precomputed_candidates]
            logger.info(f"Precomputed score range: {min(scores):.3f} - {max(scores):.3f}, mean: {sum(scores)/len(scores):.3f}")
        
        total_time = time.time() - start_time
        
        # Update performance stats
        self.perf_stats['total_generations'] += 1
        self.perf_stats['avg_generation_time_ms'] = (
            (self.perf_stats['avg_generation_time_ms'] * (self.perf_stats['total_generations'] - 1) + 
             total_time * 1000) / self.perf_stats['total_generations']
        )
        self.perf_stats['avg_nodes_processed'] = (
            (self.perf_stats['avg_nodes_processed'] * (self.perf_stats['total_generations'] - 1) + 
             len(node_locations)) / self.perf_stats['total_generations']
        )
        
        logger.info(f"Fast precomputation completed in {total_time:.3f}s "
                   f"(sampling: {sampling_time:.3f}s, features: {features_time:.3f}s, "
                   f"scoring: {scoring_time:.3f}s, conversion: {conversion_time:.3f}s)")
        
        return cache_key
    
    def generate_candidates_ultra_fast(self, cache_key: str, from_lat: float, from_lon: float,
                                     target_radius: float, exclude_nodes: Set[int] = None,
                                     min_score: float = 0.15, max_candidates: int = 5) -> List[FastSemanticCandidate]:
        """
        Ultra-fast candidate generation with aggressive optimizations.
        
        Target: Complete in 0.05-0.2 seconds.
        """
        start_time = time.time()
        
        if cache_key not in self.precomputed_cache:
            logger.warning(f"Cache key {cache_key} not found")
            return []
        
        exclude_nodes = exclude_nodes or set()
        cached_data = self.precomputed_cache[cache_key]
        all_candidates = cached_data['candidates']
        
        # Phase 1: Fast filtering using vectorized operations
        candidates = self._filter_candidates_vectorized(
            all_candidates, from_lat, from_lon, target_radius, exclude_nodes, min_score
        )
        
        # Phase 2: Smart selection with diversity
        if len(candidates) > max_candidates * 3:
            # Pre-filter to reasonable size using score-distance combination
            candidates = self._pre_filter_candidates(candidates, from_lat, from_lon, max_candidates * 3)
        
        # Phase 3: Apply directional diversity (limit expensive operations)
        diverse_candidates = self._apply_fast_directional_diversity(
            candidates, from_lat, from_lon, max_candidates
        )
        
        # Phase 4: Guarantee minimum candidates with fallback system
        final_candidates = self._ensure_minimum_candidates(
            diverse_candidates, all_candidates, from_lat, from_lon, 
            target_radius, exclude_nodes, min_score, max_candidates
        )
        
        elapsed = time.time() - start_time
        logger.debug(f"Ultra-fast generation: {len(final_candidates)} candidates in {elapsed*1000:.1f}ms")
        
        return final_candidates[:max_candidates]
    
    def _intelligent_node_selection(self, graph: nx.MultiGraph, center_lat: float, center_lon: float,
                                   radius_m: float, max_nodes: int) -> Tuple[List[Tuple[float, float]], List[int]]:
        """Intelligent node selection using multiple sampling strategies"""
        # Add randomization to cache key if probabilistic mode is enabled
        cache_suffix = ""
        if self.probabilistic_mode:
            cache_suffix = f"_{random.randint(1, 3)}"  # 3 different node selection variants
        
        cache_key = f"nodes_{center_lat:.4f}_{center_lon:.4f}_{radius_m}_{max_nodes}{cache_suffix}"
        
        if cache_key in self.node_selection_cache:
            logger.debug(f"Using cached node selection: {cache_key}")
            return self.node_selection_cache[cache_key]
        
        # Use intelligent sampler with probabilistic strategy selection
        if self.probabilistic_mode:
            # Randomly vary the sampling strategy for diversity
            strategies = ['hybrid', 'intersection_priority', 'spatial_stratified']
            strategy = random.choice(strategies)
        else:
            strategy = 'hybrid'
        
        locations, node_ids = self.node_sampler.sample_nodes_intelligently(
            graph, center_lat, center_lon, radius_m, max_nodes, strategy=strategy
        )
        
        # Cache result
        result = (locations, node_ids)
        self.node_selection_cache[cache_key] = result
        
        return result
    
    def _ensure_semantic_features_loaded(self, center_lat: float, center_lon: float, radius_m: float):
        """Ensure semantic features are loaded and indexed"""
        bbox = self.semantic_overlay_manager.calculate_bbox_from_center(
            center_lat, center_lon, radius_m / 1000.0
        )
        
        for feature_type in ['forests', 'rivers', 'lakes']:
            try:
                # Load features from semantic overlay manager
                overlay_data = self.semantic_overlay_manager.get_semantic_overlays(
                    feature_type, bbox, use_cache=True
                )
                
                # Extract geometries and load into optimized scorer
                if 'features' in overlay_data:
                    features = overlay_data['features']
                    geometries = []
                    
                    for feature in features:
                        geom_data = feature.get('geometry')
                        if geom_data:
                            try:
                                shapely_geom = self._geojson_to_shapely_simple(geom_data)
                                if shapely_geom and not shapely_geom.is_empty:
                                    geometries.append(shapely_geom)
                            except Exception as e:
                                logger.debug(f"Failed to convert geometry: {e}")
                                continue
                    
                    # Load into optimized scorer with spatial indexing
                    self.optimized_scorer.load_property_features(feature_type, geometries)
                    
            except Exception as e:
                logger.warning(f"Failed to load {feature_type}: {e}")
    
    def _geojson_to_shapely_simple(self, geom_data: Dict) -> Optional:
        """Simple GeoJSON to Shapely conversion with error handling"""
        try:
            from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
            
            geom_type = geom_data.get('type')
            coordinates = geom_data.get('coordinates', [])
            
            if not coordinates:
                return None
            
            if geom_type == 'Point':
                return Point(coordinates)
            elif geom_type == 'LineString':
                return LineString(coordinates)
            elif geom_type == 'Polygon' and len(coordinates) > 0:
                exterior = coordinates[0]
                holes = coordinates[1:] if len(coordinates) > 1 else []
                return Polygon(exterior, holes)
            elif geom_type == 'MultiPolygon':
                polygons = []
                for poly_coords in coordinates:
                    if len(poly_coords) > 0:
                        exterior = poly_coords[0]
                        holes = poly_coords[1:] if len(poly_coords) > 1 else []
                        polygons.append(Polygon(exterior, holes))
                return MultiPolygon(polygons) if polygons else None
            elif geom_type == 'MultiLineString':
                return MultiLineString([LineString(line) for line in coordinates])
                
        except ImportError:
            logger.warning("Shapely not available for geometry conversion")
            return None
        except Exception as e:
            logger.debug(f"Error converting {geom_data.get('type', 'unknown')} geometry: {e}")
            return None
        
        return None
    
    def _batch_score_optimized(self, node_locations: List[Tuple[float, float]]) -> List[Dict[str, float]]:
        """Optimized batch scoring using vectorization"""
        if not node_locations:
            return []
        
        try:
            # Use optimized scorer for batch scoring
            scores = self.optimized_scorer.score_multiple_locations_batch(
                node_locations, ['forests', 'rivers', 'lakes']
            )
            logger.info(f"Optimized scorer returned {len(scores)} score sets for {len(node_locations)} locations")
            return scores
        except Exception as e:
            logger.error(f"Optimized scoring failed: {e}")
            # Fallback to basic scoring
            return self._batch_score_fallback(node_locations)
    
    def _batch_score_fallback(self, node_locations: List[Tuple[float, float]]) -> List[Dict[str, float]]:
        """Fallback scoring if optimized version fails"""
        logger.warning("Using fallback scoring method")
        # Return basic scores for all locations
        return [{'forests': 0.3, 'rivers': 0.2, 'lakes': 0.2} for _ in node_locations]
    
    def _convert_to_candidates(self, node_ids: List[int], node_locations: List[Tuple[float, float]], 
                             semantic_scores_list: List[Dict], preference: str) -> List[FastSemanticCandidate]:
        """Convert scoring results to candidate objects"""
        candidates = []
        
        for i, (node_id, (lat, lon)) in enumerate(zip(node_ids, node_locations)):
            semantic_scores = semantic_scores_list[i] if i < len(semantic_scores_list) else {}
            
            overall_score, explanation, details = self._calculate_overall_score_fast(
                semantic_scores, preference
            )
            
            candidate = FastSemanticCandidate(
                node_id=node_id,
                lat=lat,
                lon=lon,
                semantic_scores=semantic_scores,
                overall_score=overall_score,
                explanation=explanation,
                semantic_details=details
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _filter_candidates_vectorized(self, all_candidates: List[FastSemanticCandidate],
                                    from_lat: float, from_lon: float, target_radius: float,
                                    exclude_nodes: Set[int], min_score: float) -> List[FastSemanticCandidate]:
        """Vectorized candidate filtering for speed"""
        if not all_candidates:
            logger.warning("No candidates to filter - empty candidate list")
            return []
        
        logger.info(f"Filtering {len(all_candidates)} candidates with radius {target_radius:.0f}m, min_score {min_score}")
        
        # Convert to numpy arrays for vectorized operations
        candidate_lats = np.array([c.lat for c in all_candidates])
        candidate_lons = np.array([c.lon for c in all_candidates])
        candidate_scores = np.array([c.overall_score for c in all_candidates])
        candidate_ids = np.array([c.node_id for c in all_candidates])
        
        # Debug: Check score distribution
        logger.info(f"Score distribution: min={candidate_scores.min():.3f}, max={candidate_scores.max():.3f}, mean={candidate_scores.mean():.3f}")
        
        # Vectorized distance calculation
        distances = haversine_vectorized(from_lat, from_lon, candidate_lats, candidate_lons)
        
        # Debug: Check distance distribution
        logger.info(f"Distance distribution: min={distances.min():.0f}m, max={distances.max():.0f}m, mean={distances.mean():.0f}m")
        
        # Vectorized filtering - be more lenient initially
        distance_mask = (distances >= target_radius * 0.3) & (distances <= target_radius * 2.0)  # More lenient
        score_mask = candidate_scores >= max(0.05, min_score * 0.5)  # Lower score threshold
        exclude_mask = ~np.isin(candidate_ids, list(exclude_nodes))
        
        # Debug: Check how many pass each filter
        logger.info(f"Filter results: distance_ok={distance_mask.sum()}, score_ok={score_mask.sum()}, not_excluded={exclude_mask.sum()}")
        logger.info(f"Distance range: {target_radius * 0.3:.0f}m - {target_radius * 2.0:.0f}m, min_score: {max(0.05, min_score * 0.5):.3f}")
        
        # Combine masks
        valid_mask = distance_mask & score_mask & exclude_mask
        valid_indices = np.where(valid_mask)[0]
        
        logger.info(f"Combined filter: {valid_mask.sum()} candidates passed all filters")
        
        # Return filtered candidates
        filtered_candidates = [all_candidates[i] for i in valid_indices]
        logger.info(f"Returning {len(filtered_candidates)} filtered candidates")
        
        return filtered_candidates
    
    def _pre_filter_candidates(self, candidates: List[FastSemanticCandidate], 
                             from_lat: float, from_lon: float, target_count: int) -> List[FastSemanticCandidate]:
        """Pre-filter candidates using combined score-distance ranking"""
        if len(candidates) <= target_count:
            return candidates
        
        # Calculate combined scores (semantic + distance preference)
        scored_candidates = []
        target_distance = 1000  # Preferred distance in meters
        
        for candidate in candidates:
            distance = haversine_cached(from_lat, from_lon, candidate.lat, candidate.lon)
            
            # Distance score (prefer moderate distances)
            distance_score = 1.0 - abs(distance - target_distance) / target_distance
            distance_score = max(0.0, distance_score)
            
            # Combined score
            combined_score = candidate.overall_score * 0.7 + distance_score * 0.3
            
            scored_candidates.append((candidate, combined_score))
        
        # Sort by combined score and take top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in scored_candidates[:target_count]]
    
    def _apply_fast_directional_diversity(self, candidates: List[FastSemanticCandidate],
                                        from_lat: float, from_lon: float, max_candidates: int) -> List[FastSemanticCandidate]:
        """Fast directional diversity with probabilistic selection"""
        if len(candidates) <= max_candidates:
            # Add randomization even for small candidate sets
            if self.probabilistic_mode:
                candidates_copy = candidates.copy()
                random.shuffle(candidates_copy)
                return candidates_copy
            return candidates
        
        if self.probabilistic_mode:
            return self._probabilistic_directional_selection(candidates, from_lat, from_lon, max_candidates)
        else:
            return self._deterministic_directional_selection(candidates, from_lat, from_lon, max_candidates)
    
    def _probabilistic_directional_selection(self, candidates: List[FastSemanticCandidate],
                                           from_lat: float, from_lon: float, max_candidates: int) -> List[FastSemanticCandidate]:
        """Probabilistic candidate selection with diversity enforcement"""
        # Strategy 1: Score-weighted probabilistic selection (40%)
        score_candidates = self._score_weighted_selection(candidates, max(1, int(max_candidates * 0.4)))
        
        # Strategy 2: Diversity-enforced selection (40%)
        remaining = [c for c in candidates if c.node_id not in {sc.node_id for sc in score_candidates}]
        diversity_candidates = self._diversity_enforced_selection(remaining, score_candidates, from_lat, from_lon, max(1, int(max_candidates * 0.4)))
        
        # Strategy 3: Pure exploration (20%)
        remaining = [c for c in candidates if c.node_id not in {sc.node_id for sc in score_candidates + diversity_candidates}]
        exploration_candidates = self._exploration_selection(remaining, max_candidates - len(score_candidates) - len(diversity_candidates))
        
        # Combine and apply final directional diversity check
        all_selected = score_candidates + diversity_candidates + exploration_candidates
        
        # Final directional diversity enforcement
        final_candidates = self._enforce_final_directional_spread(all_selected[:max_candidates], from_lat, from_lon, max_candidates)
        
        # Shuffle for randomness
        random.shuffle(final_candidates)
        
        return final_candidates
    
    def _score_weighted_selection(self, candidates: List[FastSemanticCandidate], count: int) -> List[FastSemanticCandidate]:
        """Score-weighted probabilistic selection"""
        if len(candidates) <= count:
            return candidates.copy()
        
        # Create weights with exploration factor
        weights = []
        for candidate in candidates:
            # Base score weight
            score_weight = math.exp(1.5 * candidate.overall_score)
            # Add exploration randomness
            exploration_bonus = random.uniform(0, self.exploration_factor)
            weights.append(score_weight + exploration_bonus)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.sample(candidates, count)
        
        probabilities = np.array(weights) / total_weight
        selected_indices = np.random.choice(len(candidates), size=count, replace=False, p=probabilities)
        
        return [candidates[i] for i in selected_indices]
    
    def _diversity_enforced_selection(self, candidates: List[FastSemanticCandidate], 
                                    already_selected: List[FastSemanticCandidate],
                                    from_lat: float, from_lon: float, count: int) -> List[FastSemanticCandidate]:
        """Selection with spatial and angular diversity enforcement"""
        if len(candidates) <= count:
            return candidates.copy()
        
        selected = []
        remaining = candidates.copy()
        
        # Use configurable diversity thresholds
        distance_threshold = self.distance_threshold
        min_angle_separation = self.min_angular_separation
        
        for _ in range(count):
            if not remaining:
                break
            
            # Calculate diversity weights
            weights = []
            for candidate in remaining:
                # Base score weight
                score_weight = math.exp(0.8 * candidate.overall_score)
                
                # Distance and angular diversity bonus/penalty
                diversity_weight = 1.0
                for prev_candidate in already_selected + selected:
                    # Distance-based diversity
                    distance = haversine_cached(candidate.lat, candidate.lon, prev_candidate.lat, prev_candidate.lon)
                    if distance < distance_threshold:
                        diversity_weight *= 0.3  # Penalty for nearby candidates
                    else:
                        diversity_weight *= 1.1  # Small bonus for distant candidates
                    
                    # Angular diversity (from current position)
                    candidate_bearing = self._calculate_bearing_fast(from_lat, from_lon, candidate.lat, candidate.lon)
                    prev_bearing = self._calculate_bearing_fast(from_lat, from_lon, prev_candidate.lat, prev_candidate.lon)
                    
                    angle_diff = abs(candidate_bearing - prev_bearing)
                    angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wraparound
                    
                    if angle_diff < min_angle_separation:
                        # Strong penalty for candidates in similar direction
                        angle_penalty = 0.1 + 0.4 * (angle_diff / min_angle_separation)  # 0.1 to 0.5 penalty
                        diversity_weight *= angle_penalty
                    else:
                        # Bonus for candidates in different directions
                        angle_bonus = 1.0 + 0.3 * min(1.0, (angle_diff - min_angle_separation) / 60)
                        diversity_weight *= angle_bonus
                
                # Add exploration randomness
                exploration_bonus = random.uniform(0, self.exploration_factor * 0.5)
                
                weights.append(score_weight * diversity_weight + exploration_bonus)
            
            # Probabilistic selection with diversity bias
            if sum(weights) > 0:
                probabilities = np.array(weights) / sum(weights)
                selected_idx = np.random.choice(len(remaining), p=probabilities)
                selected.append(remaining.pop(selected_idx))
            else:
                selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))
        
        return selected
    
    def _exploration_selection(self, candidates: List[FastSemanticCandidate], count: int) -> List[FastSemanticCandidate]:
        """Pure exploration selection for discovering new areas"""
        if len(candidates) <= count:
            return candidates.copy()
        
        # Add random exploration scores
        exploration_candidates = []
        for candidate in candidates:
            exploration_score = candidate.overall_score + random.uniform(0, self.exploration_factor * 0.5)
            exploration_candidates.append((candidate, exploration_score))
        
        # Sort by exploration score and add randomness
        exploration_candidates.sort(key=lambda x: x[1], reverse=True)
        top_count = min(count * 3, len(exploration_candidates))
        top_candidates = [candidate for candidate, _ in exploration_candidates[:top_count]]
        
        return random.sample(top_candidates, min(count, len(top_candidates)))
    
    def _enforce_final_directional_spread(self, candidates: List[FastSemanticCandidate], 
                                        from_lat: float, from_lon: float, max_candidates: int) -> List[FastSemanticCandidate]:
        """
        Final enforcement of directional spread to ensure candidates aren't clustered in one direction.
        This is the last step to guarantee good angular distribution.
        """
        if len(candidates) <= 1:
            return candidates
        
        # Calculate ideal angular separation for even distribution
        ideal_separation = 360 / max_candidates if max_candidates > 1 else 180
        min_acceptable_separation = max(self.min_angular_separation, ideal_separation * 0.6)  # Use configured minimum
        
        # Start with the highest scoring candidate
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        final_selection = [candidates[0]]
        
        # Add candidates that maintain good angular separation
        for candidate in candidates[1:]:
            if len(final_selection) >= max_candidates:
                break
            
            candidate_bearing = self._calculate_bearing_fast(from_lat, from_lon, candidate.lat, candidate.lon)
            
            # Check angular separation from all selected candidates
            sufficient_separation = True
            for selected in final_selection:
                selected_bearing = self._calculate_bearing_fast(from_lat, from_lon, selected.lat, selected.lon)
                angle_diff = abs(candidate_bearing - selected_bearing)
                angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wraparound
                
                if angle_diff < min_acceptable_separation:
                    sufficient_separation = False
                    break
            
            if sufficient_separation:
                final_selection.append(candidate)
        
        # If we don't have enough candidates due to strict angular requirements,
        # fill remaining slots with best remaining candidates (relaxed requirements)
        if len(final_selection) < max_candidates and len(candidates) > len(final_selection):
            remaining = [c for c in candidates if c.node_id not in {s.node_id for s in final_selection}]
            relaxed_separation = max(60, min_acceptable_separation * 0.5)  # Relax to 60 degrees minimum
            
            for candidate in remaining:
                if len(final_selection) >= max_candidates:
                    break
                
                candidate_bearing = self._calculate_bearing_fast(from_lat, from_lon, candidate.lat, candidate.lon)
                
                # Check with relaxed separation requirements
                sufficient_separation = True
                for selected in final_selection:
                    selected_bearing = self._calculate_bearing_fast(from_lat, from_lon, selected.lat, selected.lon)
                    angle_diff = abs(candidate_bearing - selected_bearing)
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    
                    if angle_diff < relaxed_separation:
                        sufficient_separation = False
                        break
                
                if sufficient_separation:
                    final_selection.append(candidate)
        
        # Log directional spread for debugging
        if len(final_selection) > 1:
            bearings = [self._calculate_bearing_fast(from_lat, from_lon, c.lat, c.lon) for c in final_selection]
            bearings.sort()
            min_gap = min((bearings[i+1] - bearings[i]) for i in range(len(bearings)-1))
            max_gap = max((bearings[i+1] - bearings[i]) for i in range(len(bearings)-1))
            wraparound_gap = 360 - bearings[-1] + bearings[0]
            min_gap = min(min_gap, wraparound_gap)
            max_gap = max(max_gap, wraparound_gap)
            
            logger.debug(f"Directional spread: {len(final_selection)} candidates, "
                        f"angular gaps: {min_gap:.0f}°-{max_gap:.0f}°, "
                        f"bearings: {[f'{b:.0f}°' for b in bearings]}")
        
        return final_selection
    
    def _ensure_minimum_candidates(self, selected_candidates: List[FastSemanticCandidate],
                                 all_candidates: List[FastSemanticCandidate],
                                 from_lat: float, from_lon: float, target_radius: float,
                                 exclude_nodes: Set[int], min_score: float, 
                                 required_count: int) -> List[FastSemanticCandidate]:
        """
        Guarantee minimum number of candidates through progressive fallback strategies.
        
        This method ensures we always return the requested number of candidates by:
        1. Using selected candidates if we have enough
        2. Relaxing distance constraints progressively 
        3. Relaxing score constraints progressively
        4. Relaxing angular diversity constraints
        5. Final fallback to any available candidates
        """
        if len(selected_candidates) >= required_count:
            return selected_candidates
        
        logger.info(f"Need {required_count} candidates but only have {len(selected_candidates)}, applying fallback strategies")
        
        # Start with what we have
        result = selected_candidates.copy()
        used_node_ids = {c.node_id for c in result}
        
        # Strategy 1: Relax distance constraints progressively
        if len(result) < required_count:
            result = self._fallback_relax_distance(
                result, all_candidates, from_lat, from_lon, target_radius, 
                exclude_nodes, min_score, required_count, used_node_ids
            )
            used_node_ids = {c.node_id for c in result}
        
        # Strategy 2: Relax score constraints progressively  
        if len(result) < required_count:
            result = self._fallback_relax_score(
                result, all_candidates, from_lat, from_lon, target_radius,
                exclude_nodes, min_score, required_count, used_node_ids
            )
            used_node_ids = {c.node_id for c in result}
        
        # Strategy 3: Relax angular diversity constraints
        if len(result) < required_count:
            result = self._fallback_relax_angular_diversity(
                result, all_candidates, from_lat, from_lon, target_radius,
                exclude_nodes, required_count, used_node_ids
            )
            used_node_ids = {c.node_id for c in result}
        
        # Strategy 4: Final fallback - take any remaining candidates
        if len(result) < required_count:
            result = self._fallback_any_candidates(
                result, all_candidates, exclude_nodes, required_count, used_node_ids
            )
        
        logger.info(f"Fallback strategies resulted in {len(result)} candidates (target: {required_count})")
        
        return result
    
    def _fallback_relax_distance(self, current_result: List[FastSemanticCandidate],
                               all_candidates: List[FastSemanticCandidate],
                               from_lat: float, from_lon: float, target_radius: float,
                               exclude_nodes: Set[int], min_score: float,
                               required_count: int, used_node_ids: Set[int]) -> List[FastSemanticCandidate]:
        """Fallback 1: Progressively relax distance constraints"""
        result = current_result.copy()
        
        # Try expanding distance constraints: 2x, 3x, 5x, then unlimited
        distance_multipliers = [2.0, 3.0, 5.0, float('inf')]
        
        for multiplier in distance_multipliers:
            if len(result) >= required_count:
                break
                
            expanded_radius = target_radius * multiplier
            logger.debug(f"Fallback: trying {multiplier}x distance ({expanded_radius:.0f}m)")
            
            for candidate in all_candidates:
                if len(result) >= required_count:
                    break
                    
                if (candidate.node_id in used_node_ids or 
                    candidate.node_id in exclude_nodes or
                    candidate.overall_score < min_score):
                    continue
                
                distance = haversine_cached(from_lat, from_lon, candidate.lat, candidate.lon)
                if distance <= expanded_radius:
                    result.append(candidate)
                    used_node_ids.add(candidate.node_id)
        
        return result
    
    def _fallback_relax_score(self, current_result: List[FastSemanticCandidate],
                            all_candidates: List[FastSemanticCandidate],
                            from_lat: float, from_lon: float, target_radius: float,
                            exclude_nodes: Set[int], min_score: float,
                            required_count: int, used_node_ids: Set[int]) -> List[FastSemanticCandidate]:
        """Fallback 2: Progressively relax score constraints"""
        result = current_result.copy()
        
        # Try relaxing score thresholds: 75%, 50%, 25%, 0%
        score_multipliers = [0.75, 0.5, 0.25, 0.0]
        
        for multiplier in score_multipliers:
            if len(result) >= required_count:
                break
                
            relaxed_min_score = min_score * multiplier
            logger.debug(f"Fallback: trying {multiplier*100:.0f}% score threshold ({relaxed_min_score:.3f})")
            
            for candidate in all_candidates:
                if len(result) >= required_count:
                    break
                    
                if (candidate.node_id in used_node_ids or 
                    candidate.node_id in exclude_nodes or
                    candidate.overall_score < relaxed_min_score):
                    continue
                
                distance = haversine_cached(from_lat, from_lon, candidate.lat, candidate.lon)
                if distance <= target_radius * 3.0:  # Use expanded radius from previous fallback
                    result.append(candidate)
                    used_node_ids.add(candidate.node_id)
        
        return result
    
    def _fallback_relax_angular_diversity(self, current_result: List[FastSemanticCandidate],
                                        all_candidates: List[FastSemanticCandidate],
                                        from_lat: float, from_lon: float, target_radius: float,
                                        exclude_nodes: Set[int], required_count: int, 
                                        used_node_ids: Set[int]) -> List[FastSemanticCandidate]:
        """Fallback 3: Relax angular diversity constraints"""
        result = current_result.copy()
        
        # Try relaxing angular separation: 60°, 45°, 30°, 0°
        angular_thresholds = [60, 45, 30, 0]
        
        for min_angle in angular_thresholds:
            if len(result) >= required_count:
                break
                
            logger.debug(f"Fallback: trying {min_angle}° angular separation")
            
            for candidate in all_candidates:
                if len(result) >= required_count:
                    break
                    
                if (candidate.node_id in used_node_ids or 
                    candidate.node_id in exclude_nodes):
                    continue
                
                distance = haversine_cached(from_lat, from_lon, candidate.lat, candidate.lon)
                if distance > target_radius * 5.0:  # Reasonable distance limit
                    continue
                
                # Check angular separation if required
                if min_angle > 0:
                    candidate_bearing = self._calculate_bearing_fast(from_lat, from_lon, candidate.lat, candidate.lon)
                    too_close = False
                    
                    for existing in result:
                        existing_bearing = self._calculate_bearing_fast(from_lat, from_lon, existing.lat, existing.lon)
                        angle_diff = abs(candidate_bearing - existing_bearing)
                        angle_diff = min(angle_diff, 360 - angle_diff)
                        
                        if angle_diff < min_angle:
                            too_close = True
                            break
                    
                    if too_close:
                        continue
                
                result.append(candidate)
                used_node_ids.add(candidate.node_id)
        
        return result
    
    def _fallback_any_candidates(self, current_result: List[FastSemanticCandidate],
                               all_candidates: List[FastSemanticCandidate],
                               exclude_nodes: Set[int], required_count: int,
                               used_node_ids: Set[int]) -> List[FastSemanticCandidate]:
        """Fallback 4: Take any remaining candidates (last resort)"""
        result = current_result.copy()
        
        logger.debug(f"Final fallback: taking any available candidates")
        
        # Sort by score and take the best remaining candidates
        remaining_candidates = [
            c for c in all_candidates 
            if c.node_id not in used_node_ids and c.node_id not in exclude_nodes
        ]
        remaining_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        needed = required_count - len(result)
        result.extend(remaining_candidates[:needed])
        
        return result
    
    def _deterministic_directional_selection(self, candidates: List[FastSemanticCandidate],
                                           from_lat: float, from_lon: float, max_candidates: int) -> List[FastSemanticCandidate]:
        """Original deterministic directional selection"""
        selected = []
        min_angle_separation = 360 / max_candidates  # Even distribution
        
        # Sort by score first
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Select first candidate
        if candidates:
            selected.append(candidates[0])
        
        # Add candidates with sufficient directional separation
        for candidate in candidates[1:]:
            if len(selected) >= max_candidates:
                break
            
            candidate_bearing = self._calculate_bearing_fast(from_lat, from_lon, candidate.lat, candidate.lon)
            
            # Check separation from existing candidates
            sufficient_separation = True
            for selected_candidate in selected:
                selected_bearing = self._calculate_bearing_fast(from_lat, from_lon, 
                                                              selected_candidate.lat, selected_candidate.lon)
                angle_diff = abs(candidate_bearing - selected_bearing)
                angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wraparound
                
                if angle_diff < min_angle_separation:
                    sufficient_separation = False
                    break
            
            if sufficient_separation:
                selected.append(candidate)
        
        return selected
    
    @lru_cache(maxsize=1000)
    def _calculate_bearing_fast(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Cached bearing calculation"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def _calculate_overall_score_fast(self, semantic_scores: Dict[str, float], preference: str) -> Tuple[float, str, str]:
        """Fast overall score calculation with simplified logic"""
        if not semantic_scores:
            return 0.3, "Basic walkable area", "No natural features detected."
        
        # Simplified preference weighting
        weights = {'forests': 1.0, 'rivers': 1.0, 'lakes': 1.0}
        preference_lower = preference.lower()
        
        if 'forest' in preference_lower or 'park' in preference_lower or 'nature' in preference_lower:
            weights['forests'] *= 1.3
        if 'water' in preference_lower or 'river' in preference_lower:
            weights['rivers'] *= 1.3
        if 'lake' in preference_lower or 'pond' in preference_lower:
            weights['lakes'] *= 1.3
        
        # Calculate weighted score
        total_weighted = sum(semantic_scores.get(prop, 0) * weight 
                           for prop, weight in weights.items())
        total_weight = sum(weights.values())
        
        overall_score = min(1.0, total_weighted / total_weight) if total_weight > 0 else 0.3
        
        # Quick explanation
        best_features = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        if best_features and best_features[0][1] > 0.1:
            explanation = f"Near {best_features[0][0]}"
            if len(best_features) > 1 and best_features[1][1] > 0.1:
                explanation += f" and {best_features[1][0]}"
        else:
            explanation = "Basic walkable area"
        
        details = f"Scores: " + ", ".join([f"{k}:{v:.1f}" for k, v in semantic_scores.items() if v > 0])
        
        return overall_score, explanation, details
    
    def _calculate_bounds(self, locations: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Calculate bounding box for locations"""
        if not locations:
            return (0, 0, 0, 0)
        
        lats = [lat for lat, lon in locations]
        lons = [lon for lat, lon in locations]
        
        return (min(lats), min(lons), max(lats), max(lons))
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        optimizer_stats = self.optimized_scorer.get_performance_stats()
        
        return {
            **self.perf_stats,
            'optimizer_stats': optimizer_stats,
            'cache_sizes': {
                'precomputed_cache': len(self.precomputed_cache),
                'node_selection_cache': len(self.node_selection_cache)
            }
        }
    
    def clear_cache(self, cache_key: str = None):
        """Clear caches"""
        if cache_key:
            self.precomputed_cache.pop(cache_key, None)
        else:
            self.precomputed_cache.clear()
            self.node_selection_cache.clear()
            self.optimized_scorer.clear_cache()
        
        logger.info(f"Cleared fast candidate generator cache: {cache_key or 'all'}")


def haversine_vectorized(lat1: float, lon1: float, lat2_array: np.ndarray, lon2_array: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance calculation - moved here to avoid circular imports"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2_array)
    dlat_rad = np.radians(lat2_array - lat1)
    dlon_rad = np.radians(lon2_array - lon1)
    
    a = (np.sin(dlat_rad/2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * 
         np.sin(dlon_rad/2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c