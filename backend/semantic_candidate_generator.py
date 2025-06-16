"""
Efficient Semantic Candidate Generator
Uses pre-loaded semantic features to quickly identify high-value candidate nodes
without expensive per-node proximity calculations.
"""

import math
import time
import random
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import networkx as nx
from shapely.geometry import Point
from loguru import logger


@dataclass
class PrecomputedSemanticNode:
    """Node with precomputed semantic scores."""
    node_id: int
    lat: float
    lon: float
    semantic_scores: Dict[str, float]
    overall_score: float
    explanation: str
    semantic_details: str


class SemanticCandidateGenerator:
    """
    Fast candidate generation using precomputed semantic scores.
    Avoids expensive per-candidate proximity calculations.
    """
    
    def __init__(self, semantic_overlay_manager):
        self.semantic_overlay_manager = semantic_overlay_manager
        self.node_cache: Dict[str, List[PrecomputedSemanticNode]] = {}
        self.cache_bounds: Dict[str, Tuple[float, float, float, float]] = {}  # (min_lat, min_lon, max_lat, max_lon)
        
        # Probabilistic generation parameters
        self.probabilistic_mode = True  # Enable probabilistic selection
        self.exploration_factor = 0.3  # How much randomness to inject (0.0 = deterministic, 1.0 = random)
        self.diversity_enforcement = True  # Enforce spatial diversity in selection
        
    def precompute_area_scores(self, graph: nx.MultiGraph, center_lat: float, center_lon: float, 
                              radius_m: float = 8000, preference: str = "scenic nature") -> str:
        """
        Precompute semantic scores for all nodes in an area.
        
        Args:
            graph: NetworkX graph with nodes
            center_lat: Center latitude  
            center_lon: Center longitude
            radius_m: Radius in meters
            preference: User preference string
            
        Returns:
            Cache key for the precomputed area
        """
        cache_key = f"{center_lat:.4f}_{center_lon:.4f}_{radius_m}_{hash(preference) % 10000}"
        
        if cache_key in self.node_cache:
            logger.info(f"Using existing precomputed semantic scores: {cache_key}")
            return cache_key
        
        logger.info(f"Precomputing semantic scores for {len(graph.nodes)} nodes in area")
        start_time = time.time()
        
        # Ensure semantic features are loaded for the area
        bbox = self.semantic_overlay_manager.calculate_bbox_from_center(
            center_lat, center_lon, radius_m / 1000.0
        )
        
        # Load all semantic features for the area
        for feature_type in ['forests', 'rivers', 'lakes']:
            try:
                self.semantic_overlay_manager.get_semantic_overlays(feature_type, bbox, use_cache=True)
            except Exception as e:
                logger.warning(f"Failed to load {feature_type} for precomputation: {e}")
        
        # Strategic node selection for efficient semantic analysis
        node_locations, node_ids = self._select_strategic_nodes(
            graph, center_lat, center_lon, radius_m
        )
        
        logger.info(f"Scoring {len(node_locations)} strategically selected nodes (vs {len(graph.nodes)} total nodes)")
        
        # Batch score all locations at once
        if node_locations:
            try:
                semantic_scores_list = self.semantic_overlay_manager.score_multiple_locations(
                    node_locations, ['forests', 'rivers', 'lakes']
                )
            except Exception as e:
                logger.error(f"Failed to batch score locations: {e}")
                semantic_scores_list = [{}] * len(node_locations)
        else:
            semantic_scores_list = []
        
        # Convert to precomputed nodes
        precomputed_nodes = []
        bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # min_lat, min_lon, max_lat, max_lon
        
        for i, (node_id, (lat, lon)) in enumerate(zip(node_ids, node_locations)):
            semantic_scores = semantic_scores_list[i] if i < len(semantic_scores_list) else {}
            
            overall_score, explanation, details = self._calculate_overall_score(
                semantic_scores, preference
            )
            
            precomputed_node = PrecomputedSemanticNode(
                node_id=node_id,
                lat=lat,
                lon=lon,
                semantic_scores=semantic_scores,
                overall_score=overall_score,
                explanation=explanation,
                semantic_details=details
            )
            
            precomputed_nodes.append(precomputed_node)
            
            # Update bounds
            bounds[0] = min(bounds[0], lat)  # min_lat
            bounds[1] = min(bounds[1], lon)  # min_lon
            bounds[2] = max(bounds[2], lat)  # max_lat
            bounds[3] = max(bounds[3], lon)  # max_lon
        
        # Cache results
        self.node_cache[cache_key] = precomputed_nodes
        self.cache_bounds[cache_key] = tuple(bounds)
        
        elapsed = time.time() - start_time
        logger.info(f"Precomputed {len(precomputed_nodes)} semantic scores in {elapsed:.2f}s")
        
        return cache_key
    
    def generate_candidates_fast(self, cache_key: str, from_lat: float, from_lon: float,
                                target_radius: float, exclude_nodes: Set[int] = None,
                                min_score: float = 0.2, randomize: bool = True) -> List[PrecomputedSemanticNode]:
        """
        Fast candidate generation using precomputed scores with probabilistic selection.
        
        Args:
            cache_key: Key for precomputed area
            from_lat: Starting latitude
            from_lon: Starting longitude  
            target_radius: Target radius in meters
            exclude_nodes: Set of node IDs to exclude
            min_score: Minimum semantic score threshold
            randomize: If True, add probabilistic selection for non-deterministic results
            
        Returns:
            List of candidate nodes with probabilistic diversity
        """
        if cache_key not in self.node_cache:
            logger.warning(f"Cache key {cache_key} not found, returning empty candidates")
            return []
        
        precomputed_nodes = self.node_cache[cache_key]
        exclude_nodes = exclude_nodes or set()
        
        candidates = []
        
        # Filter candidates efficiently using precomputed scores
        for node in precomputed_nodes:
            if node.node_id in exclude_nodes:
                continue
                
            # Distance filter
            distance = self._haversine_distance(from_lat, from_lon, node.lat, node.lon)
            if distance < target_radius * 0.5 or distance > target_radius * 1.5:
                continue
            
            # Score filter
            if node.overall_score < min_score:
                continue
            
            # Calculate combined score (semantic score + distance preference)
            distance_score = self._calculate_distance_score(distance, target_radius)
            combined_score = (node.overall_score * 0.7) + (distance_score * 0.3)
            
            # Create candidate with distance info
            candidate = PrecomputedSemanticNode(
                node_id=node.node_id,
                lat=node.lat,
                lon=node.lon,
                semantic_scores=node.semantic_scores,
                overall_score=combined_score,  # Use combined score
                explanation=node.explanation,
                semantic_details=f"{node.semantic_details} (Distance: {distance:.0f}m)"
            )
            
            candidates.append(candidate)
        
        # Sort by combined score 
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Apply probabilistic selection if randomize is enabled
        if randomize and len(candidates) > 15:  # Only apply if we have enough candidates
            candidates = self._probabilistic_selection(candidates, selection_size=min(50, len(candidates)))
        
        # First, ensure we have directional diversity by selecting candidates from different directions
        directionally_diverse = self._ensure_directional_coverage(
            candidates, from_lat, from_lon, target_directions=4, candidates_per_direction=5
        )
        
        # Then apply final directional diversity filtering with probabilistic component
        diverse_candidates = self._apply_directional_diversity(
            directionally_diverse, from_lat, from_lon, max_candidates=5, randomize=randomize
        )
        
        # Debug: Log candidate directions
        if diverse_candidates:
            logger.info(f"Candidate directions from ({from_lat:.6f}, {from_lon:.6f}):")
            for i, candidate in enumerate(diverse_candidates[:3]):
                bearing = self._calculate_bearing(from_lat, from_lon, candidate.lat, candidate.lon)
                distance = self._haversine_distance(from_lat, from_lon, candidate.lat, candidate.lon)
                logger.info(f"  {i+1}. Bearing: {bearing:6.1f}°, Distance: {distance:4.0f}m, Score: {candidate.overall_score:.3f}")
        
        logger.info(f"Generated {len(diverse_candidates)} candidates from {len(precomputed_nodes)} precomputed nodes")
        return diverse_candidates[:3]  # Return top 3 candidates to avoid choice overload
    
    def _calculate_distance_score(self, distance: float, target_radius: float) -> float:
        """Calculate preference score based on distance to target radius."""
        optimal_distance = target_radius
        distance_diff = abs(distance - optimal_distance)
        max_diff = target_radius * 0.5
        
        if distance_diff >= max_diff:
            return 0.0
        
        return 1.0 - (distance_diff / max_diff)
    
    def _probabilistic_selection(self, candidates: List[PrecomputedSemanticNode], 
                                selection_size: int = 50) -> List[PrecomputedSemanticNode]:
        """
        Apply probabilistic selection to introduce randomness while favoring higher scores.
        Uses score-weighted probability to maintain quality bias while adding variability.
        
        Args:
            candidates: Sorted list of candidates (best scores first)
            selection_size: Number of candidates to select for further processing
            
        Returns:
            Probabilistically selected candidates
        """
        if len(candidates) <= selection_size:
            return candidates
        
        # Create probability weights based on scores with exponential decay
        # Higher scores get exponentially higher probability
        scores = [c.overall_score for c in candidates]
        min_score = min(scores)
        max_score = max(scores)
        
        # Normalize scores to 0-1 range
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            normalized_scores = [1.0] * len(scores)
        
        # Apply exponential weighting with exploration factor
        # Higher exploration_factor = more randomness, lower = more deterministic
        if self.probabilistic_mode:
            # Mix score-based and random weighting based on exploration factor
            temperature = 1.5 * (1.0 - self.exploration_factor) + 0.3 * self.exploration_factor
            weights = [math.exp(temperature * score) + random.uniform(0, self.exploration_factor) for score in normalized_scores]
        else:
            # Pure score-based selection
            weights = [math.exp(2.0 * score) for score in normalized_scores]
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Randomly select candidates based on weighted probabilities
        selected_indices = np.random.choice(
            len(candidates), 
            size=min(selection_size, len(candidates)), 
            replace=False, 
            p=probabilities
        )
        
        selected_candidates = [candidates[i] for i in selected_indices]
        
        # Sort selected candidates by score to maintain some order
        selected_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        logger.debug(f"Probabilistic selection: {len(selected_candidates)} from {len(candidates)} candidates")
        return selected_candidates
    
    def _ensure_directional_coverage(self, candidates: List[PrecomputedSemanticNode], 
                                   from_lat: float, from_lon: float,
                                   target_directions: int = 4, 
                                   candidates_per_direction: int = 5) -> List[PrecomputedSemanticNode]:
        """
        Ensure directional coverage by selecting candidates from different directional quadrants.
        This prevents clustering by semantic bias before applying final diversity filtering.
        
        Args:
            candidates: Sorted list of candidates (best semantic scores first)
            from_lat: Starting latitude
            from_lon: Starting longitude
            target_directions: Number of directional sectors (4 = N/E/S/W quadrants)
            candidates_per_direction: Max candidates to select per direction
            
        Returns:
            List of candidates with good directional spread
        """
        if len(candidates) <= target_directions:
            return candidates
        
        # Divide 360° into directional sectors
        sector_size = 360.0 / target_directions
        directional_buckets = {i: [] for i in range(target_directions)}
        
        # Sort candidates into directional buckets
        for candidate in candidates:
            bearing = self._calculate_bearing(from_lat, from_lon, candidate.lat, candidate.lon)
            sector = int(bearing / sector_size) % target_directions
            directional_buckets[sector].append(candidate)
        
        # Select best candidates from each direction
        directionally_diverse = []
        
        # First pass: Take the best candidate from each non-empty direction
        for sector in range(target_directions):
            if directional_buckets[sector]:
                directionally_diverse.append(directional_buckets[sector][0])
        
        # Second pass: Fill remaining slots by rotating through directions
        remaining_slots = candidates_per_direction * target_directions - len(directionally_diverse)
        sector_idx = 0
        candidate_idx = {i: 1 for i in range(target_directions)}  # Start from index 1 (already took 0)
        
        while remaining_slots > 0 and any(len(directional_buckets[i]) > candidate_idx[i] for i in range(target_directions)):
            if (len(directional_buckets[sector_idx]) > candidate_idx[sector_idx] and 
                sum(1 for c in directionally_diverse if self._get_candidate_sector(c, from_lat, from_lon, target_directions) == sector_idx) < candidates_per_direction):
                
                directionally_diverse.append(directional_buckets[sector_idx][candidate_idx[sector_idx]])
                candidate_idx[sector_idx] += 1
                remaining_slots -= 1
            
            sector_idx = (sector_idx + 1) % target_directions
        
        # Log directional distribution for debugging
        sector_counts = {}
        for candidate in directionally_diverse:
            sector = self._get_candidate_sector(candidate, from_lat, from_lon, target_directions)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        logger.info(f"Directional coverage: {sector_counts} (target: {target_directions} directions)")
        
        return directionally_diverse
    
    def _get_candidate_sector(self, candidate: PrecomputedSemanticNode, 
                             from_lat: float, from_lon: float, target_directions: int) -> int:
        """Get the directional sector for a candidate."""
        bearing = self._calculate_bearing(from_lat, from_lon, candidate.lat, candidate.lon)
        sector_size = 360.0 / target_directions
        return int(bearing / sector_size) % target_directions

    def _apply_directional_diversity(self, candidates: List[PrecomputedSemanticNode], 
                                   from_lat: float, from_lon: float, 
                                   max_candidates: int = 3, randomize: bool = True) -> List[PrecomputedSemanticNode]:
        """Apply directional diversity to avoid clustering candidates with optional randomization."""
        if len(candidates) <= max_candidates:
            return candidates
        
        selected = []
        min_angle_separation = 90  # degrees - ensure good directional spread for 3 candidates
        
        # If randomizing, shuffle the top candidates within score tiers to add variability
        if randomize and len(candidates) > max_candidates:
            # Group candidates into score tiers and randomize within tiers
            candidates = self._randomize_within_score_tiers(candidates, tier_size=5)
        
        # Always include the top candidate (after potential randomization)
        if candidates:
            selected.append(candidates[0])
        
        for candidate in candidates[1:]:
            if len(selected) >= max_candidates:
                break
            
            # Calculate angle from origin to this candidate
            candidate_angle = self._calculate_bearing(from_lat, from_lon, candidate.lat, candidate.lon)
            
            # Check if this candidate is sufficiently separated from existing ones
            too_close = False
            for selected_candidate in selected:
                selected_angle = self._calculate_bearing(from_lat, from_lon, 
                                                       selected_candidate.lat, selected_candidate.lon)
                angle_diff = abs(candidate_angle - selected_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wraparound
                
                if angle_diff < min_angle_separation:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(candidate)
        
        return selected
    
    def _randomize_within_score_tiers(self, candidates: List[PrecomputedSemanticNode], 
                                     tier_size: int = 5) -> List[PrecomputedSemanticNode]:
        """
        Enhanced randomization within score tiers with probabilistic selection.
        
        Args:
            candidates: Sorted list of candidates
            tier_size: Size of each score tier to randomize within
            
        Returns:
            Candidates with enhanced randomization within score tiers
        """
        if not self.probabilistic_mode:
            return candidates  # Return deterministic order
            
        if len(candidates) <= tier_size:
            randomized = candidates.copy()
            random.shuffle(randomized)
            return randomized
        
        result = []
        for i in range(0, len(candidates), tier_size):
            tier = candidates[i:i + tier_size]
            
            if self.exploration_factor > 0.5:
                # High exploration: completely shuffle tier
                random.shuffle(tier)
            else:
                # Medium exploration: probabilistic reordering within tier
                tier_scores = [c.overall_score for c in tier]
                if len(set(tier_scores)) > 1:  # If scores differ in tier
                    # Add small random values to scores for reordering
                    random_values = [random.uniform(0, self.exploration_factor * 0.1) for _ in tier]
                    scored_tier = [(tier[j], tier_scores[j] + random_values[j]) for j in range(len(tier))]
                    scored_tier.sort(key=lambda x: x[1], reverse=True)
                    tier = [candidate for candidate, _ in scored_tier]
                else:
                    random.shuffle(tier)
            
            result.extend(tier)
        
        return result
    
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
        
        logger.info(f"Probabilistic mode: {enabled}, exploration: {self.exploration_factor:.2f}, diversity: {diversity_enforcement}")
    
    def clear_cache(self):
        """Clear all cached precomputed nodes to ensure fresh probabilistic results."""
        cache_size = len(self.node_cache)
        self.node_cache.clear()
        self.cache_bounds.clear()
        
        if cache_size > 0:
            logger.info(f"Cleared {cache_size} cached precomputed areas for fresh probabilistic generation")
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing in degrees from point 1 to point 2."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def _calculate_overall_score(self, semantic_scores: dict, preference: str) -> Tuple[float, str, str]:
        """Calculate overall score from semantic scores - same logic as before."""
        if not semantic_scores:
            return 0.3, "Basic walkable area", "No natural features detected in this area."
        
        # Weight scores based on preference keywords
        preference_lower = preference.lower()
        weights = {
            'forests': 1.0,
            'rivers': 1.0, 
            'lakes': 1.0
        }
        
        # Adjust weights based on user preferences
        if any(word in preference_lower for word in ['park', 'forest', 'tree', 'green', 'nature', 'wood']):
            weights['forests'] *= 1.5
        if any(word in preference_lower for word in ['water', 'river', 'stream', 'canal']):
            weights['rivers'] *= 1.5
        if any(word in preference_lower for word in ['lake', 'pond', 'water', 'swimming']):
            weights['lakes'] *= 1.5
        if any(word in preference_lower for word in ['scenic', 'beautiful', 'view']):
            for key in weights:
                weights[key] *= 1.2
        
        # Calculate weighted average
        total_weighted_score = 0.0
        total_weight = 0.0
        feature_details = []
        
        for property_name, score in semantic_scores.items():
            if property_name in weights and score > 0:
                weight = weights[property_name]
                total_weighted_score += score * weight
                total_weight += weight
                
                # Add feature detail
                if score > 0.7:
                    feature_details.append(f"Very close to {property_name} (score: {score:.1f})")
                elif score > 0.4:
                    feature_details.append(f"Near {property_name} (score: {score:.1f})")
                elif score > 0.1:
                    feature_details.append(f"Some {property_name} nearby (score: {score:.1f})")
        
        if total_weight > 0:
            overall_score = min(1.0, total_weighted_score / total_weight)
        else:
            overall_score = 0.3
        
        # Generate explanation
        if not feature_details:
            explanation = "Basic walkable area with minimal natural features"
            details = "This location has limited access to natural features but is still walkable."
        else:
            top_features = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            top_names = [name.rstrip('s') for name, score in top_features if score > 0.1]
            
            if len(top_names) >= 2:
                explanation = f"Great location near {top_names[0]} and {top_names[1]}"
            elif len(top_names) == 1:
                explanation = f"Nice location near {top_names[0]}"
            else:
                explanation = "Decent walkable area"
                
            details = "; ".join(feature_details)
        
        return overall_score, explanation, details
    
    def _select_strategic_nodes(self, graph, center_lat: float, center_lon: float, radius_m: float):
        """
        Select strategically important nodes for semantic analysis instead of all nodes.
        
        This reduces computation from ~15k-30k nodes to ~1k-3k strategic nodes while
        maintaining or improving candidate quality.
        """
        strategic_nodes = []
        strategic_ids = []
        
        # Step 1: Filter all nodes to radius first
        radius_nodes = []
        radius_ids = []
        for node_id, data in graph.nodes(data=True):
            lat, lon = data['y'], data['x']
            if self._haversine_distance(center_lat, center_lon, lat, lon) <= radius_m:
                radius_nodes.append((node_id, lat, lon, data))
                radius_ids.append(node_id)
        
        logger.info(f"Found {len(radius_nodes)} nodes within {radius_m}m radius")
        
        # Step 2: Priority 1 - Major intersections (high connectivity = interesting for routes)
        intersection_threshold = 3  # Nodes with 3+ connections
        intersections = []
        for node_id, lat, lon, data in radius_nodes:
            degree = graph.degree(node_id)
            if degree >= intersection_threshold:
                intersections.append((node_id, lat, lon))
        
        strategic_nodes.extend([(lat, lon) for _, lat, lon in intersections])
        strategic_ids.extend([node_id for node_id, _, _ in intersections])
        logger.info(f"Selected {len(intersections)} intersection nodes")
        
        # Step 3: Priority 2 - Grid sampling for coverage (every ~200m instead of every node)
        grid_spacing = 200  # meters
        grid_nodes = self._sample_grid_nodes(radius_nodes, center_lat, center_lon, grid_spacing)
        
        # Avoid duplicates from intersections
        existing_ids = set(strategic_ids)
        for node_id, lat, lon in grid_nodes:
            if node_id not in existing_ids:
                strategic_nodes.append((lat, lon))
                strategic_ids.append(node_id)
                existing_ids.add(node_id)
        
        logger.info(f"Added {len(grid_nodes)} grid-sampled nodes (total: {len(strategic_nodes)})")
        
        # Step 4: Priority 3 - Random sampling for diversity if we still have budget
        target_total = min(2000, len(radius_nodes))  # Cap at 2000 nodes max
        if len(strategic_nodes) < target_total:
            remaining_budget = target_total - len(strategic_nodes)
            random_candidates = [
                (node_id, lat, lon) for node_id, lat, lon, _ in radius_nodes 
                if node_id not in existing_ids
            ]
            
            if random_candidates:
                import random
                random_sample = random.sample(
                    random_candidates, 
                    min(remaining_budget, len(random_candidates))
                )
                
                for node_id, lat, lon in random_sample:
                    strategic_nodes.append((lat, lon))
                    strategic_ids.append(node_id)
                
                logger.info(f"Added {len(random_sample)} random nodes for diversity")
        
        logger.info(f"Strategic selection: {len(strategic_nodes)} nodes from {len(radius_nodes)} total "
                   f"({100*len(strategic_nodes)/len(radius_nodes):.1f}% reduction)")
        
        return strategic_nodes, strategic_ids
    
    def _sample_grid_nodes(self, radius_nodes, center_lat: float, center_lon: float, spacing_m: float):
        """Sample nodes on a regular grid for coverage."""
        grid_nodes = []
        
        # Convert spacing to approximate degrees (rough approximation)
        lat_spacing = spacing_m / 111000  # ~111km per degree latitude
        lon_spacing = spacing_m / (111000 * math.cos(math.radians(center_lat)))
        
        # Create grid points
        grid_points = set()
        for node_id, lat, lon, data in radius_nodes:
            # Snap to grid
            grid_lat = round(lat / lat_spacing) * lat_spacing
            grid_lon = round(lon / lon_spacing) * lon_spacing
            grid_key = (grid_lat, grid_lon)
            
            if grid_key not in grid_points:
                grid_points.add(grid_key)
                grid_nodes.append((node_id, lat, lon))
        
        return grid_nodes
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters."""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat_rad = math.radians(lat2 - lat1)
        dlon_rad = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat_rad/2) * math.sin(dlat_rad/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon_rad/2) * math.sin(dlon_rad/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def clear_cache(self, cache_key: str = None):
        """Clear precomputed cache."""
        if cache_key:
            self.node_cache.pop(cache_key, None)
            self.cache_bounds.pop(cache_key, None)
            logger.info(f"Cleared semantic cache for key: {cache_key}")
        else:
            self.node_cache.clear()
            self.cache_bounds.clear()
            logger.info("Cleared all semantic caches")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_nodes = sum(len(nodes) for nodes in self.node_cache.values())
        return {
            "cached_areas": len(self.node_cache),
            "total_precomputed_nodes": total_nodes,
            "cache_keys": list(self.node_cache.keys())
        }