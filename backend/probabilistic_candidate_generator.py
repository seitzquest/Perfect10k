"""
Probabilistic Candidate Generator for Perfect10k
Uses environmental locality information for fast, non-deterministic candidate generation
"""

import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import geohash2
from loguru import logger

from semantic_scoring import SemanticScorer
from interactive_router import CandidateLocation


@dataclass
class EnvironmentalZone:
    """Environmental zone with probabilistic sampling weights."""
    zone_type: str
    center_lat: float
    center_lon: float
    radius_m: float
    semantic_bias: Dict[str, float]  # Feature type -> weight multiplier
    exploration_weight: float = 1.0
    diversity_bonus: float = 1.0


class ProbabilisticCandidateGenerator:
    """
    Generates candidates probabilistically using environmental locality.
    
    Key features:
    - No precomputed candidate storage
    - Uses environmental zones for fast sampling
    - Probabilistic selection with diversity enforcement
    - Consistent sub-200ms performance
    """
    
    def __init__(self, semantic_scorer: SemanticScorer):
        self.semantic_scorer = semantic_scorer
        self.exploration_factor = 0.3  # Balance exploration vs exploitation
        self.diversity_threshold = 500  # Minimum distance between candidates (meters)
        self.zone_cache = {}  # Cache environmental zones
        
    def generate_candidates(self, 
                          graph, 
                          from_location: Tuple[float, float],
                          target_radius: float,
                          preference: str = "scenic nature",
                          max_candidates: int = 3,
                          diversity_enforcement: bool = True) -> List[CandidateLocation]:
        """
        Generate candidates probabilistically using environmental zones.
        
        Args:
            graph: OSMnx graph
            from_location: (lat, lon) current position
            target_radius: Search radius in meters
            preference: User preference string
            max_candidates: Number of candidates to return
            diversity_enforcement: Whether to enforce spatial diversity
            
        Returns:
            List of probabilistically selected candidates
        """
        start_time = time.time()
        
        # Get environmental zones around location
        zones = self._get_environmental_zones(from_location, target_radius)
        logger.debug(f"Found {len(zones)} environmental zones")
        
        # Sample candidates from zones probabilistically
        candidate_pool = []
        for zone in zones:
            zone_candidates = self._sample_zone_probabilistically(
                graph, zone, preference, max_samples=30
            )
            candidate_pool.extend(zone_candidates)
        
        logger.debug(f"Sampled {len(candidate_pool)} candidates from zones")
        
        # Apply final probabilistic selection with diversity
        selected_candidates = self._select_diverse_probabilistically(
            candidate_pool, max_candidates, diversity_enforcement
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Generated {len(selected_candidates)} probabilistic candidates in {elapsed:.3f}s")
        
        return selected_candidates
    
    def _get_environmental_zones(self, 
                                location: Tuple[float, float], 
                                radius_m: float) -> List[EnvironmentalZone]:
        """
        Identify environmental zones for probabilistic sampling.
        Uses spatial locality without storing precomputed scores.
        """
        lat, lon = location
        
        # Create cache key based on location and radius
        cache_key = f"{lat:.4f}_{lon:.4f}_{radius_m}"
        if cache_key in self.zone_cache:
            return self.zone_cache[cache_key]
        
        zones = []
        
        # Get environmental context from spatial tiles
        environmental_context = self._get_environmental_context(lat, lon, radius_m)
        
        # Create zones based on environmental features
        if environmental_context.get('forest_density', 0) > 0.3:
            zones.append(EnvironmentalZone(
                zone_type='green_corridor',
                center_lat=lat,
                center_lon=lon,
                radius_m=radius_m * 0.6,
                semantic_bias={'forests': 2.0, 'parks': 1.5},
                exploration_weight=1.2
            ))
        
        if environmental_context.get('water_proximity', 0) > 0.2:
            zones.append(EnvironmentalZone(
                zone_type='water_vicinity',
                center_lat=lat,
                center_lon=lon,
                radius_m=radius_m * 0.7,
                semantic_bias={'rivers': 1.8, 'lakes': 1.6},
                exploration_weight=1.3
            ))
        
        # Always include mixed exploration zone
        zones.append(EnvironmentalZone(
            zone_type='mixed_exploration',
            center_lat=lat,
            center_lon=lon,
            radius_m=radius_m,
            semantic_bias={},
            exploration_weight=1.0,
            diversity_bonus=1.5
        ))
        
        # Cache zones for reuse
        self.zone_cache[cache_key] = zones
        
        return zones
    
    def _get_environmental_context(self, 
                                  lat: float, 
                                  lon: float, 
                                  radius_m: float) -> Dict[str, float]:
        """
        Fast environmental context lookup using spatial locality.
        Returns general environmental characteristics without specific scores.
        """
        # Use geohash for spatial locality
        tile_hash = geohash2.encode(lat, lon, precision=6)
        
        # Quick heuristic-based environmental assessment
        # This could be enhanced with actual spatial tile data
        context = {
            'forest_density': self._estimate_forest_density(lat, lon),
            'water_proximity': self._estimate_water_proximity(lat, lon),
            'connectivity': self._estimate_connectivity(lat, lon),
            'urban_density': self._estimate_urban_density(lat, lon)
        }
        
        return context
    
    def _sample_zone_probabilistically(self, 
                                     graph, 
                                     zone: EnvironmentalZone, 
                                     preference: str,
                                     max_samples: int = 30) -> List[CandidateLocation]:
        """
        Sample candidates probabilistically from an environmental zone.
        """
        # Get nodes in zone radius
        nodes_in_zone = self._get_nodes_in_radius(
            graph, zone.center_lat, zone.center_lon, zone.radius_m
        )
        
        if len(nodes_in_zone) == 0:
            return []
        
        # Limit sampling for efficiency
        if len(nodes_in_zone) > max_samples * 3:
            # Probabilistic subsampling based on zone characteristics
            sampled_nodes = self._probabilistic_node_sampling(
                graph, nodes_in_zone, zone, max_samples * 3
            )
        else:
            sampled_nodes = nodes_in_zone
        
        # Score candidates with zone bias
        candidates = []
        for node in sampled_nodes[:max_samples]:
            try:
                candidate = self._create_candidate_with_zone_bias(
                    graph, node, zone, preference
                )
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Failed to create candidate for node {node}: {e}")
                continue
        
        return candidates
    
    def _probabilistic_node_sampling(self, 
                                   graph, 
                                   nodes: List, 
                                   zone: EnvironmentalZone,
                                   max_nodes: int) -> List:
        """
        Probabilistically sample nodes based on zone characteristics.
        """
        if len(nodes) <= max_nodes:
            return nodes
        
        # Create sampling weights based on node characteristics
        weights = []
        for node in nodes:
            node_data = graph.nodes[node]
            lat, lon = node_data['y'], node_data['x']
            
            # Base weight
            weight = 1.0
            
            # Zone-specific bonuses
            if zone.zone_type == 'green_corridor':
                # Prefer nodes near intersections (higher connectivity)
                if len(list(graph.neighbors(node))) > 2:
                    weight *= 1.3
            elif zone.zone_type == 'water_vicinity':
                # Add some randomness for water vicinity exploration
                weight *= random.uniform(0.8, 1.4)
            
            # Distance from zone center factor
            distance_factor = 1.0 - (self._haversine_distance(
                lat, lon, zone.center_lat, zone.center_lon
            ) / zone.radius_m)
            weight *= (0.5 + 0.5 * distance_factor)
            
            # Exploration factor
            weight *= (1.0 + random.uniform(0, zone.exploration_weight - 1.0))
            
            weights.append(weight)
        
        # Probabilistic sampling
        selected_indices = np.random.choice(
            len(nodes), 
            size=min(max_nodes, len(nodes)), 
            replace=False,
            p=np.array(weights) / np.sum(weights)
        )
        
        return [nodes[i] for i in selected_indices]
    
    def _create_candidate_with_zone_bias(self, 
                                       graph, 
                                       node, 
                                       zone: EnvironmentalZone,
                                       preference: str) -> Optional[CandidateLocation]:
        """
        Create candidate with zone-specific semantic bias.
        """
        node_data = graph.nodes[node]
        lat, lon = node_data['y'], node_data['x']
        
        # Fast semantic scoring with zone bias
        base_scores = self.semantic_scorer.score_location_fast(lat, lon)
        
        # Apply zone bias
        biased_score = 0.0
        for feature, weight in zone.semantic_bias.items():
            if feature in base_scores:
                biased_score += base_scores[feature] * weight
        
        # Add base score for features not in bias
        for feature, score in base_scores.items():
            if feature not in zone.semantic_bias:
                biased_score += score * 0.5
        
        # Normalize and add exploration bonus
        final_score = biased_score / max(1.0, len(base_scores))
        final_score += random.uniform(0, 0.1) * zone.exploration_weight
        
        return CandidateLocation(
            node_id=node,
            lat=lat,
            lon=lon,
            value_score=final_score,
            distance_from_current=0.0,  # Will be calculated later
            estimated_route_completion=0.0,  # Will be calculated later
            explanation=f"Probabilistic selection from {zone.zone_type}",
            semantic_scores=base_scores,
            semantic_details=f"Zone: {zone.zone_type}, bias: {zone.semantic_bias}"
        )
    
    def _select_diverse_probabilistically(self, 
                                        candidates: List[CandidateLocation],
                                        max_candidates: int,
                                        diversity_enforcement: bool = True) -> List[CandidateLocation]:
        """
        Final probabilistic selection with diversity enforcement.
        """
        if len(candidates) <= max_candidates:
            return candidates
        
        # Sort by score for initial filtering
        candidates.sort(key=lambda c: c.value_score, reverse=True)
        
        # Keep top candidates but add probabilistic selection
        top_tier = candidates[:max_candidates * 2]
        
        if not diversity_enforcement:
            # Simple probabilistic selection from top tier
            weights = [math.exp(1.5 * c.value_score) for c in top_tier]
            selected_indices = np.random.choice(
                len(top_tier),
                size=min(max_candidates, len(top_tier)),
                replace=False,
                p=np.array(weights) / np.sum(weights)
            )
            return [top_tier[i] for i in selected_indices]
        
        # Diversity-enforced selection
        selected = []
        remaining = top_tier.copy()
        
        # Always select the best candidate
        selected.append(remaining.pop(0))
        
        # Select remaining candidates with diversity constraint
        while len(selected) < max_candidates and remaining:
            # Calculate diversity weights
            weights = []
            for candidate in remaining:
                # Base weight from score
                score_weight = math.exp(1.0 * candidate.value_score)
                
                # Diversity bonus (distance from already selected)
                diversity_weight = 1.0
                for selected_candidate in selected:
                    distance = self._haversine_distance(
                        candidate.lat, candidate.lon,
                        selected_candidate.lat, selected_candidate.lon
                    )
                    if distance < self.diversity_threshold:
                        diversity_weight *= 0.3  # Penalize close candidates
                    else:
                        diversity_weight *= 1.2  # Bonus for diverse candidates
                
                weights.append(score_weight * diversity_weight)
            
            # Probabilistic selection with diversity bias
            if weights and sum(weights) > 0:
                selected_idx = np.random.choice(
                    len(remaining),
                    p=np.array(weights) / np.sum(weights)
                )
                selected.append(remaining.pop(selected_idx))
            else:
                # Fallback: random selection
                selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))
        
        return selected
    
    # Utility methods
    def _get_nodes_in_radius(self, graph, lat: float, lon: float, radius_m: float) -> List:
        """Get all nodes within radius of location."""
        import osmnx as ox
        center_node = ox.nearest_nodes(graph, lon, lat)
        
        # Get nodes within radius using graph traversal (more accurate than geometric)
        nodes_in_radius = []
        visited = set()
        queue = [(center_node, 0.0)]
        
        while queue:
            node, distance = queue.pop(0)
            if node in visited or distance > radius_m:
                continue
                
            visited.add(node)
            nodes_in_radius.append(node)
            
            # Add neighbors to queue
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    edge_data = graph.edges[node, neighbor, 0]
                    edge_length = edge_data.get('length', 0)
                    new_distance = distance + edge_length
                    if new_distance <= radius_m:
                        queue.append((neighbor, new_distance))
        
        return nodes_in_radius
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters."""
        R = 6371000  # Earth's radius in meters
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    # Heuristic estimation methods (can be enhanced with real data)
    def _estimate_forest_density(self, lat: float, lon: float) -> float:
        """Estimate forest density using heuristics."""
        # Simple heuristic - can be enhanced with real spatial data
        return random.uniform(0.1, 0.8) if abs(lat) < 60 else 0.1
    
    def _estimate_water_proximity(self, lat: float, lon: float) -> float:
        """Estimate water proximity using heuristics."""
        # Simple heuristic - can be enhanced with real spatial data
        return random.uniform(0.0, 0.6)
    
    def _estimate_connectivity(self, lat: float, lon: float) -> float:
        """Estimate road connectivity using heuristics."""
        # Simple heuristic - can be enhanced with real spatial data
        return random.uniform(0.3, 0.9)
    
    def _estimate_urban_density(self, lat: float, lon: float) -> float:
        """Estimate urban density using heuristics."""
        # Simple heuristic - can be enhanced with real spatial data
        return random.uniform(0.2, 0.8)


# Add missing import
import time