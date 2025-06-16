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
    
    def precompute_area_scores_fast(self, graph: nx.MultiGraph, center_lat: float, center_lon: float, 
                                   radius_m: float = 8000, preference: str = "scenic nature",
                                   max_nodes: int = 1000) -> str:
        """
        Fast precomputation using intelligent sampling and optimized scoring.
        
        Target: Complete in 0.5-2 seconds instead of 10-30 seconds.
        """
        cache_key = f"fast_{center_lat:.4f}_{center_lon:.4f}_{radius_m}_{hash(preference) % 10000}"
        
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
        
        elapsed = time.time() - start_time
        logger.debug(f"Ultra-fast generation: {len(diverse_candidates)} candidates in {elapsed*1000:.1f}ms")
        
        return diverse_candidates[:max_candidates]
    
    def _intelligent_node_selection(self, graph: nx.MultiGraph, center_lat: float, center_lon: float,
                                   radius_m: float, max_nodes: int) -> Tuple[List[Tuple[float, float]], List[int]]:
        """Intelligent node selection using multiple sampling strategies"""
        cache_key = f"nodes_{center_lat:.4f}_{center_lon:.4f}_{radius_m}_{max_nodes}"
        
        if cache_key in self.node_selection_cache:
            logger.debug(f"Using cached node selection: {cache_key}")
            return self.node_selection_cache[cache_key]
        
        # Use intelligent sampler
        locations, node_ids = self.node_sampler.sample_nodes_intelligently(
            graph, center_lat, center_lon, radius_m, max_nodes, strategy='hybrid'
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
        """Fast directional diversity with minimal computation"""
        if len(candidates) <= max_candidates:
            return candidates
        
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