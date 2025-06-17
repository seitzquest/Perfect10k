"""
Interactive Route Planner - Optimized for Perfect10k
Client-session based approach with graph caching for fast performance.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Set, List

from loguru import logger

# Lazy imports for heavy dependencies to improve startup performance
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import networkx as nx
    import osmnx as ox
    from core.graph_cache import PersistentGraphCache
    from core.semantic_matcher import SemanticMatcher
    from semantic_overlays import SemanticOverlayManager
    from semantic_candidate_generator import SemanticCandidateGenerator


@dataclass
class RouteCandidate:
    """A candidate destination node."""
    node_id: int
    lat: float
    lon: float
    value_score: float
    distance_from_current: float
    estimated_route_completion: float  # Estimated total route length if chosen as final
    explanation: str = "Basic walkable area"  # Explanation for the score
    semantic_scores: dict = None  # Individual scores per semantic property
    semantic_details: str = ""  # Detailed semantic explanation


@dataclass
class ClientSession:
    """State of a client session with cached graph and routing capability."""
    client_id: str
    graph: 'nx.MultiGraph'
    graph_center: tuple[float, float]  # (lat, lon) center of cached graph
    graph_radius: float  # radius in meters
    semantic_matcher: 'SemanticMatcher'

    # Current route state (None if no active route)
    active_route: Optional['RouteState'] = None
    created_at: float = 0
    last_access: float = 0
    
    # Semantic scoring cache
    semantic_cache_key: str = None

@dataclass
class RouteState:
    """State of an active route being built."""
    start_node: int
    start_location: tuple[float, float]  # (lat, lon)
    current_waypoints: list[int]  # Ordered list of waypoints
    current_path: list[int]  # Full path through all waypoints
    total_distance: float
    estimated_final_distance: float
    target_distance: float
    preference: str
    value_function: dict
    used_edges: set[tuple[int, int]]  # Track used edges to avoid conflicts


class InteractiveRouteBuilder:
    """Builds routes interactively with client-session based approach."""

    def __init__(self, cache_dir: str = "cache/graphs"):
        self.client_sessions: dict[str, ClientSession] = {}
        self.graph_cache: dict[str, 'nx.MultiGraph'] = {}  # Legacy in-memory cache
        self.cache_dir = cache_dir
        
        # Lazy initialization for faster startup performance
        self._persistent_cache = None
        self._semantic_matcher = None
        self._semantic_overlay_manager = None
        self._semantic_candidate_generator = None

        logger.info("Initialized InteractiveRouteBuilder (lazy mode for faster startup)")

    @property
    def persistent_cache(self):
        """Get enhanced spatial cache instance (lazy initialization)."""
        if self._persistent_cache is None:
            from core.enhanced_graph_cache import EnhancedGraphCache
            self._persistent_cache = EnhancedGraphCache(self.cache_dir)
        return self._persistent_cache

    @property
    def semantic_matcher(self):
        """Get semantic matcher instance (lazy initialization)."""
        if self._semantic_matcher is None:
            from core.semantic_matcher import SemanticMatcher
            self._semantic_matcher = SemanticMatcher()
        return self._semantic_matcher

    @property
    def semantic_overlay_manager(self):
        """Get semantic overlay manager instance (lazy initialization)."""
        if self._semantic_overlay_manager is None:
            from semantic_overlays import SemanticOverlayManager
            self._semantic_overlay_manager = SemanticOverlayManager()
        return self._semantic_overlay_manager

    @property
    def semantic_candidate_generator(self):
        """Get semantic candidate generator instance (lazy initialization)."""
        if self._semantic_candidate_generator is None:
            from semantic_candidate_generator import SemanticCandidateGenerator
            self._semantic_candidate_generator = SemanticCandidateGenerator(self.semantic_overlay_manager)
        return self._semantic_candidate_generator
    
    @property
    def fast_candidate_generator(self):
        """Get fast candidate generator instance (lazy initialization)."""
        if not hasattr(self, '_fast_candidate_generator') or self._fast_candidate_generator is None:
            try:
                from fast_candidate_generator import FastCandidateGenerator
                self._fast_candidate_generator = FastCandidateGenerator(self.semantic_overlay_manager)
                logger.info("Initialized fast candidate generator")
            except ImportError as e:
                logger.warning(f"Fast candidate generator not available: {e}")
                self._fast_candidate_generator = None
        return self._fast_candidate_generator

    def get_or_create_client_session(self, client_id: str, lat: float, lon: float, radius: float = 8000) -> ClientSession:
        """Get existing client session or create new one with cached graph."""

        # Check if client already has a session
        if client_id in self.client_sessions:
            session = self.client_sessions[client_id]
            session.last_access = time.time()

            # Check if requested location is within cached graph area
            if self._is_location_in_graph_area(lat, lon, session.graph_center, session.graph_radius):
                logger.info(f"Using existing session for client {client_id}")
                return session
            else:
                logger.info(f"Location outside cached area, creating new session for client {client_id}")
                # Remove old session
                del self.client_sessions[client_id]

        # Create new session with cached graph
        logger.info(f"Creating new session for client {client_id} at ({lat:.6f}, {lon:.6f})")

        # Load or get cached graph with specified radius
        graph = self._get_cached_graph(lat, lon, radius)

        session = ClientSession(
            client_id=client_id,
            graph=graph,
            graph_center=(lat, lon),
            graph_radius=radius,
            semantic_matcher=self.semantic_matcher,
            created_at=time.time(),
            last_access=time.time()
        )

        self.client_sessions[client_id] = session
        return session

    def _is_location_in_graph_area(self, lat: float, lon: float, center: tuple[float, float], radius: float) -> bool:
        """Check if location is within the cached graph area."""
        distance = self._haversine_distance(lat, lon, center[0], center[1])
        return distance <= radius * 0.8  # Use 80% of radius for safety margin

    def _get_cached_graph(self, lat: float, lon: float, radius: float = 8000) -> 'nx.MultiGraph':
        """Get graph from persistent cache or load new one with fallback caching."""

        # First, try to get from persistent cache
        graph = self.persistent_cache.get_graph(lat, lon, radius)

        if graph is not None:
            logger.info(f"Using persistent cached graph for ({lat:.6f}, {lon:.6f})")
            return graph

        # Fallback: check legacy in-memory cache
        cache_key = f"{round(lat, 2)}_{round(lon, 2)}"
        if cache_key in self.graph_cache:
            logger.info(f"Using legacy cached graph for key {cache_key}")
            return self.graph_cache[cache_key]

        # Load new graph from OSM
        logger.info(f"Loading new graph from OSM for ({lat:.6f}, {lon:.6f})")
        graph = self._load_graph(lat, lon, radius)

        # Store in spatial tile cache for permanent future use
        try:
            semantic_features = {'forests', 'rivers', 'lakes'}  # Available semantic features
            success = self.persistent_cache.store_graph(graph, lat, lon, radius, semantic_features)
            if success:
                logger.info(f"Stored graph as spatial tiles for permanent caching")
            else:
                logger.warning(f"Failed to store graph in spatial tiles")
        except Exception as e:
            logger.warning(f"Failed to store graph in spatial tile cache: {e}")

        # Also store in legacy cache for immediate reuse
        if len(self.graph_cache) >= 5:  # Reduced from 10 since we have persistent cache
            oldest_key = list(self.graph_cache.keys())[0]
            del self.graph_cache[oldest_key]
            logger.debug(f"Removed old legacy cached graph {oldest_key}")

        self.graph_cache[cache_key] = graph
        return graph

    def _generate_candidates_for_session(self, session: ClientSession, from_node: int, use_fast_generator: bool = True) -> list[RouteCandidate]:
        """Generate candidates using fast or regular semantic precomputation."""
        if not session.active_route:
            raise ValueError("No active route in session")

        route = session.active_route

        # Choose generation method based on availability and preference
        if use_fast_generator and self.fast_candidate_generator is not None:
            return self._generate_candidates_fast(session, from_node)
        else:
            return self._generate_candidates_regular(session, from_node)
    
    def _generate_candidates_fast(self, session: ClientSession, from_node: int) -> list[RouteCandidate]:
        """Generate candidates using ultra-fast optimized generator with enhanced probabilistic features."""
        route = session.active_route
        
        # Configure probabilistic mode for variety
        if hasattr(self.fast_candidate_generator, 'set_probabilistic_mode'):
            # Higher exploration factor for more variety in candidate selection
            exploration_factor = 0.4  # Increased from default 0.3 for more diversity
            self.fast_candidate_generator.set_probabilistic_mode(
                enabled=True, 
                exploration_factor=exploration_factor,
                diversity_enforcement=True
            )
        
        # Ensure fast semantic scores are precomputed for current waypoint area
        from_lat = session.graph.nodes[from_node]["y"]
        from_lon = session.graph.nodes[from_node]["x"]
        
        # Create dynamic cache key based on current waypoint location
        # Add randomization when probabilistic mode is enabled to prevent deterministic caching
        base_cache_key = f"waypoint_{from_lat:.4f}_{from_lon:.4f}_{route.preference}"
        
        # Check if fast generator is in probabilistic mode
        probabilistic_mode = getattr(self.fast_candidate_generator, 'probabilistic_mode', False) if self.fast_candidate_generator else False
        
        if probabilistic_mode:
            # Add random suffix to cache key to enable probabilistic variety
            import random
            cache_variant = random.randint(1, 12)  # 12 variants for good diversity
            waypoint_cache_key = f"{base_cache_key}_prob{cache_variant}"
            logger.info(f"Probabilistic mode: using cache variant {cache_variant} for waypoint")
        else:
            waypoint_cache_key = base_cache_key
        
        # Check if we have precomputed scores for this waypoint area
        if not hasattr(session, 'waypoint_cache_keys'):
            session.waypoint_cache_keys = {}
            
        if waypoint_cache_key not in session.waypoint_cache_keys:
            # Limit cache size to prevent memory issues (keep last 5 waypoint areas)
            if len(session.waypoint_cache_keys) >= 5:
                # Remove oldest cache entry
                oldest_key = next(iter(session.waypoint_cache_keys))
                logger.debug(f"Removing old waypoint cache: {oldest_key}")
                del session.waypoint_cache_keys[oldest_key]
            logger.info(f"Fast precomputing semantic scores for waypoint area at ({from_lat:.4f}, {from_lon:.4f})")
            
            # Precompute around current waypoint with reasonable radius
            precompute_radius = min(session.graph_radius, 4000)  # Max 4km radius for efficiency
            
            session.waypoint_cache_keys[waypoint_cache_key] = self.fast_candidate_generator.precompute_area_scores_fast(
                session.graph, 
                from_lat,  # Use current waypoint location
                from_lon,  # Use current waypoint location
                precompute_radius,
                route.preference,
                max_nodes=1000  # Limit nodes for speed
            )
        
        # Use the cache key for current waypoint
        current_cache_key = session.waypoint_cache_keys[waypoint_cache_key]

        # Calculate target radius based on remaining distance
        remaining_distance = route.target_distance - route.total_distance
        slack_factor = 1.2
        calculated_radius = slack_factor * remaining_distance / (2 * math.pi)
        target_radius = max(800, min(calculated_radius, 2000))  # Increased minimum from 200m to 800m

        logger.info(f"DEBUG RADIUS: target_distance={route.target_distance:.0f}m, total_distance={route.total_distance:.0f}m, remaining={remaining_distance:.0f}m, calculated_radius={calculated_radius:.0f}m, final_radius={target_radius:.0f}m")
        logger.info(f"Ultra-fast candidate generation from node {from_node} ({from_lat:.4f}, {from_lon:.4f}) with radius {target_radius:.0f}m")

        # Get candidates using ultra-fast generator
        exclude_nodes = set(route.current_waypoints)
        
        # Also exclude nodes too close to existing waypoints
        for node, data in session.graph.nodes(data=True):
            if self._too_close_to_existing_waypoints_in_session(session, node, min_distance=100):
                exclude_nodes.add(node)

        fast_candidates = self.fast_candidate_generator.generate_candidates_ultra_fast(
            current_cache_key,  # Use waypoint-specific cache key
            from_lat,
            from_lon, 
            target_radius,
            exclude_nodes,
            min_score=0.15,  # Slightly lower threshold for more options
            max_candidates=3
        )
        
        # Ensure we have exactly 3 candidates with additional fallback if needed
        if len(fast_candidates) < 3:
            logger.warning(f"Fast generator returned only {len(fast_candidates)} candidates, applying emergency fallback")
            fast_candidates = self._emergency_candidate_fallback(
                session, from_node, from_lat, from_lon, target_radius, exclude_nodes, fast_candidates
            )

        # Convert to RouteCandidate format
        candidates = []
        for fc in fast_candidates:
            distance = self._haversine_distance(from_lat, from_lon, fc.lat, fc.lon)
            
            # Estimate total route completion
            estimated_completion = route.total_distance + distance
            if route.start_node != from_node:
                return_estimate = self._haversine_distance(fc.lat, fc.lon,
                                                         session.graph.nodes[route.start_node]["y"],
                                                         session.graph.nodes[route.start_node]["x"])
                estimated_completion += return_estimate

            candidates.append(RouteCandidate(
                node_id=fc.node_id,
                lat=fc.lat,
                lon=fc.lon,
                value_score=fc.overall_score,
                distance_from_current=distance,
                estimated_route_completion=estimated_completion,
                explanation=fc.explanation,
                semantic_scores=fc.semantic_scores,
                semantic_details=fc.semantic_details
            ))

        logger.info(f"Generated {len(candidates)} candidates using ultra-fast semantic scoring")
        
        # Fallback: If fast generator returns no candidates, try regular generator
        if len(candidates) == 0:
            logger.warning("Fast generator returned 0 candidates, falling back to regular generator")
            return self._generate_candidates_regular(session, from_node)
        
        return candidates
    
    def _generate_candidates_regular(self, session: ClientSession, from_node: int) -> list[RouteCandidate]:
        """Generate candidates using regular semantic precomputation (fallback)."""
        route = session.active_route

        # Ensure semantic scores are precomputed for this area
        # Check if semantic candidate generator is in probabilistic mode
        probabilistic_mode = getattr(self.semantic_candidate_generator, 'probabilistic_mode', False) if self.semantic_candidate_generator else False
        
        if not session.semantic_cache_key or (probabilistic_mode and random.random() < 0.3):
            # In probabilistic mode, occasionally force recomputation for variety (30% chance)
            if probabilistic_mode and session.semantic_cache_key:
                logger.info("Probabilistic mode: forcing semantic recomputation for variety")
            else:
                logger.info("Precomputing semantic scores for session area (regular)")
            
            session.semantic_cache_key = self.semantic_candidate_generator.precompute_area_scores(
                session.graph, 
                session.graph_center[0], 
                session.graph_center[1],
                session.graph_radius,
                route.preference
            )

        # Calculate target radius based on remaining distance
        remaining_distance = route.target_distance - route.total_distance
        slack_factor = 1.2
        calculated_radius = slack_factor * remaining_distance / (2 * math.pi)
        target_radius = max(800, min(calculated_radius, 2000))  # Increased minimum from 200m to 800m

        logger.info(f"DEBUG RADIUS (regular): target_distance={route.target_distance:.0f}m, total_distance={route.total_distance:.0f}m, remaining={remaining_distance:.0f}m, calculated_radius={calculated_radius:.0f}m, final_radius={target_radius:.0f}m")

        from_lat = session.graph.nodes[from_node]["y"]
        from_lon = session.graph.nodes[from_node]["x"]

        logger.info(f"Regular candidate generation from node {from_node} with radius {target_radius:.0f}m")

        # Get candidates using precomputed semantic scores
        exclude_nodes = set(route.current_waypoints)
        
        # Also exclude nodes too close to existing waypoints
        for node, data in session.graph.nodes(data=True):
            if self._too_close_to_existing_waypoints_in_session(session, node, min_distance=100):
                exclude_nodes.add(node)

        precomputed_candidates = self.semantic_candidate_generator.generate_candidates_fast(
            session.semantic_cache_key,
            from_lat,
            from_lon, 
            target_radius,
            exclude_nodes,
            min_score=0.2,
            randomize=True  # Enable probabilistic selection for non-deterministic results
        )

        # Convert to RouteCandidate format
        candidates = []
        for pc in precomputed_candidates:
            distance = self._haversine_distance(from_lat, from_lon, pc.lat, pc.lon)
            
            # Estimate total route completion
            estimated_completion = route.total_distance + distance
            if route.start_node != from_node:
                return_estimate = self._haversine_distance(pc.lat, pc.lon,
                                                         session.graph.nodes[route.start_node]["y"],
                                                         session.graph.nodes[route.start_node]["x"])
                estimated_completion += return_estimate

            candidates.append(RouteCandidate(
                node_id=pc.node_id,
                lat=pc.lat,
                lon=pc.lon,
                value_score=pc.overall_score,
                distance_from_current=distance,
                estimated_route_completion=estimated_completion,
                explanation=pc.explanation,
                semantic_scores=pc.semantic_scores,
                semantic_details=pc.semantic_details
            ))

        logger.info(f"Generated {len(candidates)} candidates using regular semantic scoring")
        return candidates

    def _generate_candidates_distance_only(self, session: ClientSession, from_node: int) -> list[RouteCandidate]:
        """Generate candidates quickly using distance-based approach only (emergency fallback)."""
        if not session.active_route:
            raise ValueError("No active route in session")

        route = session.active_route

        # Calculate target radius based on remaining distance
        remaining_distance = route.target_distance - route.total_distance
        slack_factor = 1.2
        target_radius = slack_factor * remaining_distance / (2 * math.pi)
        target_radius = max(800, min(target_radius, 2000))  # Increased minimum from 200m to 800m

        from_lat = session.graph.nodes[from_node]["y"]
        from_lon = session.graph.nodes[from_node]["x"]

        logger.info(f"Emergency distance-only candidate generation from node {from_node} with radius {target_radius:.0f}m")

        # Get nearby nodes based on distance only (no semantic computation)
        candidates = []
        exclude_nodes = set(route.current_waypoints)
        
        # Find nodes within target radius
        for node, data in session.graph.nodes(data=True):
            if node in exclude_nodes:
                continue
                
            node_lat, node_lon = data['y'], data['x']
            distance = self._haversine_distance(from_lat, from_lon, node_lat, node_lon)
            
            # Filter by distance and minimum spacing
            if distance < target_radius and distance > 100:  # At least 100m away
                if not self._too_close_to_existing_waypoints_in_session(session, node, min_distance=150):
                    # Simple score based on distance (prefer moderate distances)
                    optimal_distance = target_radius * 0.6
                    distance_score = 1.0 - abs(distance - optimal_distance) / target_radius
                    distance_score = max(0.1, distance_score)
                    
                    # Estimate total route completion
                    estimated_completion = route.total_distance + distance
                    if route.start_node != from_node:
                        return_estimate = self._haversine_distance(node_lat, node_lon,
                                                                 session.graph.nodes[route.start_node]["y"],
                                                                 session.graph.nodes[route.start_node]["x"])
                        estimated_completion += return_estimate

                    candidates.append(RouteCandidate(
                        node_id=node,
                        lat=node_lat,
                        lon=node_lon,
                        value_score=distance_score,
                        distance_from_current=distance,
                        estimated_route_completion=estimated_completion,
                        explanation=f"Distance-based candidate ({distance:.0f}m away)",
                        semantic_scores={"distance": distance_score},
                        semantic_details="Emergency fallback - no semantic analysis"
                    ))

        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x.value_score, reverse=True)
        top_candidates = candidates[:3]

        logger.info(f"Generated {len(top_candidates)} distance-only candidates (emergency fallback)")
        return top_candidates

    def _too_close_to_existing_waypoints_in_session(self, session: ClientSession, node: int, min_distance: float) -> bool:
        """Check if node is too close to existing waypoints in session."""
        if not session.active_route:
            return False

        route = session.active_route
        node_lat = session.graph.nodes[node]["y"]
        node_lon = session.graph.nodes[node]["x"]

        for waypoint in route.current_waypoints:
            wp_lat = session.graph.nodes[waypoint]["y"]
            wp_lon = session.graph.nodes[waypoint]["x"]

            if self._haversine_distance(node_lat, node_lon, wp_lat, wp_lon) < min_distance:
                return True

        return False

    def start_route(self, client_id: str, lat: float, lon: float, preference: str, target_distance: int = 8000) -> dict:
        """
        Start a new route within client session.
        Uses cached graph for fast performance with optimized initial loading.
        """
        logger.info(f"Starting route for client {client_id} at ({lat:.6f}, {lon:.6f}) - optimized mode")
        
        loading_phases = []
        start_time = time.time()

        # Phase 1: Graph loading
        phase_start = time.time()
        session = self.get_or_create_client_session(client_id, lat, lon, radius=6000)  # 6km instead of 8km
        loading_phases.append({
            "phase": "graph_loading",
            "description": "Loading street network graph",
            "duration_ms": int((time.time() - phase_start) * 1000),
            "completed": True
        })

        # Phase 2: Route initialization
        phase_start = time.time()
        import osmnx as ox
        start_node = ox.nearest_nodes(session.graph, lon, lat)

        # Create value function from preferences
        value_function = session.semantic_matcher.create_value_function(preference)

        # Create route state
        route_state = RouteState(
            start_node=start_node,
            start_location=(lat, lon),
            current_waypoints=[start_node],
            current_path=[start_node],
            total_distance=0.0,
            estimated_final_distance=0.0,
            target_distance=target_distance,
            preference=preference,
            value_function=value_function,
            used_edges=set()
        )

        # Set active route in session
        session.active_route = route_state
        loading_phases.append({
            "phase": "route_initialization",
            "description": "Initializing route parameters",
            "duration_ms": int((time.time() - phase_start) * 1000),
            "completed": True
        })

        # Phase 3: Semantic data processing (this is the potentially slow part)
        phase_start = time.time()
        was_semantic_cached = session.semantic_cache_key is not None
        candidates = self._generate_candidates_for_session(session, start_node)
        semantic_duration = int((time.time() - phase_start) * 1000)
        
        loading_phases.append({
            "phase": "semantic_analysis",
            "description": "Analyzing natural features and generating candidates",
            "duration_ms": semantic_duration,
            "completed": True,
            "was_cached": was_semantic_cached,
            "nodes_analyzed": len(session.graph.nodes) if hasattr(session, 'graph') else 0
        })

        total_duration = int((time.time() - start_time) * 1000)
        
        return {
            "session_id": client_id,  # Use client_id as session_id for frontend compatibility
            "start_location": {
                "lat": session.graph.nodes[start_node]["y"],
                "lon": session.graph.nodes[start_node]["x"]
            },
            "candidates": [
                {
                    "node_id": c.node_id,
                    "lat": c.lat,
                    "lon": c.lon,
                    "value_score": c.value_score,
                    "distance": c.distance_from_current,
                    "estimated_completion": c.estimated_route_completion,
                    "explanation": c.explanation,
                    "semantic_scores": c.semantic_scores or {},
                    "semantic_details": c.semantic_details or "",
                    "score_breakdown": {
                        "forests": c.semantic_scores.get('forests', 0.0) if c.semantic_scores else 0.0,
                        "rivers": c.semantic_scores.get('rivers', 0.0) if c.semantic_scores else 0.0,
                        "lakes": c.semantic_scores.get('lakes', 0.0) if c.semantic_scores else 0.0,
                        "overall": c.value_score
                    }
                }
                for c in candidates
            ],
            "route_stats": {
                "current_distance": 0.0,
                "target_distance": target_distance,
                "progress": 0.0
            },
            "loading_phases": loading_phases,
            "performance": {
                "total_duration_ms": total_duration,
                "semantic_cached": was_semantic_cached,
                "graph_nodes": len(session.graph.nodes),
                "candidates_generated": len(candidates)
            },
            "semantic_precomputation": {
                "was_cached": was_semantic_cached,
                "computation_time_ms": semantic_duration,
                "nodes_processed": len(session.graph.nodes),
                "cache_key": session.semantic_cache_key or ""
            }
        }

    def add_waypoint(self, client_id: str, node_id: int) -> dict:
        """
        Add a waypoint to the route and generate new candidates.
        """
        if client_id not in self.client_sessions:
            raise ValueError("Client session not found")

        session = self.client_sessions[client_id]
        session.last_access = time.time()

        if not session.active_route:
            raise ValueError("No active route found")

        route = session.active_route

        logger.info(f"Adding waypoint {node_id} to client session {client_id}")

        # Plan path from current end to new waypoint
        current_end = route.current_waypoints[-1]
        path_segment = self._plan_path_segment(session.graph, current_end, node_id, route.used_edges)

        if not path_segment:
            raise ValueError("Cannot reach waypoint - no valid path")

        # Update route state
        segment_distance = self._calculate_path_distance(session.graph, path_segment)
        route.current_waypoints.append(node_id)

        # Extend current path (remove duplicate connection node)
        route.current_path.extend(path_segment[1:])
        route.total_distance += segment_distance

        # Track used edges
        self._add_edges_to_used(route, path_segment)

        # Estimate completion distance (shortest path back to start)
        completion_path = self._plan_shortest_disjunct_path(session.graph, node_id, route.start_node, route.used_edges)
        completion_distance = self._calculate_path_distance(session.graph, completion_path) if completion_path else 0
        route.estimated_final_distance = route.total_distance + completion_distance

        # Generate new candidates from this waypoint
        candidates = self._generate_candidates_for_session(session, node_id)

        return {
            "success": True,
            "current_path": [
                {
                    "lat": session.graph.nodes[n]["y"],
                    "lon": session.graph.nodes[n]["x"]
                }
                for n in route.current_path
            ],
            "candidates": [
                {
                    "node_id": c.node_id,
                    "lat": c.lat,
                    "lon": c.lon,
                    "value_score": c.value_score,
                    "distance": c.distance_from_current,
                    "estimated_completion": c.estimated_route_completion,
                    "explanation": c.explanation,
                    "semantic_scores": c.semantic_scores or {},
                    "semantic_details": c.semantic_details or "",
                    "score_breakdown": {
                        "forests": c.semantic_scores.get('forests', 0.0) if c.semantic_scores else 0.0,
                        "rivers": c.semantic_scores.get('rivers', 0.0) if c.semantic_scores else 0.0,
                        "lakes": c.semantic_scores.get('lakes', 0.0) if c.semantic_scores else 0.0,
                        "overall": c.value_score
                    }
                }
                for c in candidates
            ],
            "route_stats": {
                "current_distance": route.total_distance,
                "estimated_final_distance": route.estimated_final_distance,
                "target_distance": route.target_distance,
                "progress": route.total_distance / route.target_distance,
                "waypoints_count": len(route.current_waypoints)
            }
        }

    def finalize_route(self, client_id: str, final_node_id: int) -> dict:
        """
        Complete the route by connecting to final destination and back to start.
        """
        if client_id not in self.client_sessions:
            raise ValueError("Client session not found")

        session = self.client_sessions[client_id]
        session.last_access = time.time()

        if not session.active_route:
            raise ValueError("No active route found")

        route = session.active_route

        logger.info(f"Finalizing route for client {client_id} with destination {final_node_id}")

        current_end = route.current_waypoints[-1]

        # Path to final destination
        path_to_final = self._plan_path_segment(session.graph, current_end, final_node_id, route.used_edges)
        if not path_to_final:
            raise ValueError("Cannot reach final destination")

        # Add final destination segment
        final_segment_distance = self._calculate_path_distance(session.graph, path_to_final)
        route.current_path.extend(path_to_final[1:])
        route.total_distance += final_segment_distance
        self._add_edges_to_used(route, path_to_final)

        # Plan return path to start (must be disjunct)
        return_path = self._plan_shortest_disjunct_path(session.graph, final_node_id, route.start_node, route.used_edges)
        if not return_path:
            raise ValueError("Cannot find disjunct return path to start")

        # Complete the cycle
        return_distance = self._calculate_path_distance(session.graph, return_path)
        route.current_path.extend(return_path[1:])  # Exclude duplicate final node
        route.total_distance += return_distance

        # Mark as complete
        route.current_waypoints.append(final_node_id)

        final_coordinates = [
            {
                "lat": session.graph.nodes[n]["y"],
                "lon": session.graph.nodes[n]["x"]
            }
            for n in route.current_path
        ]

        # Calculate additional route statistics for final route
        accuracy = 1.0 - (abs(route.total_distance - route.target_distance) / route.target_distance)
        progress = min(route.total_distance / route.target_distance, 1.0)

        # Calculate enclosed area (simplified polygon area)
        area = self._calculate_polygon_area(final_coordinates)

        # Calculate route quality score
        distance_score = max(0, 1 - abs(route.total_distance - route.target_distance) / route.target_distance)
        complexity_score = min(len(route.current_waypoints) / 10, 1.0)  # Reward reasonable complexity
        route_score = (distance_score * 0.7 + complexity_score * 0.3)

        # Clear active route since it's completed
        session.active_route = None

        return {
            "success": True,
            "completed_route": final_coordinates,
            "route_stats": {
                "total_distance": route.total_distance,
                "current_distance": route.total_distance,  # For compatibility
                "target_distance": route.target_distance,
                "accuracy": accuracy,
                "progress": progress,
                "waypoints_count": len(route.current_waypoints),
                "total_nodes": len(route.current_path),
                "area": area,
                "score": route_score,
                "conflicts": 0,  # No conflicts in interactive approach
                "convexity": 0.8  # Placeholder convexity score
            },
            "message": f"Route completed: {route.total_distance:.0f}m cycle with {len(route.current_waypoints)} waypoints"
        }

    def get_route_status(self, client_id: str) -> dict:
        """Get current route status and statistics."""
        if client_id not in self.client_sessions:
            raise ValueError("Client session not found")

        session = self.client_sessions[client_id]
        session.last_access = time.time()

        if not session.active_route:
            return {
                "session_id": client_id,
                "route_stats": {
                    "current_distance": 0.0,
                    "estimated_final_distance": 0.0,
                    "target_distance": 0.0,
                    "progress": 0.0,
                    "waypoints_count": 0,
                    "total_nodes": 0
                },
                "current_path": []
            }

        route = session.active_route
        return {
            "session_id": client_id,
            "route_stats": {
                "current_distance": route.total_distance,
                "estimated_final_distance": route.estimated_final_distance,
                "target_distance": route.target_distance,
                "progress": route.total_distance / route.target_distance,
                "waypoints_count": len(route.current_waypoints),
                "total_nodes": len(route.current_path)
            },
            "current_path": [
                {
                    "lat": session.graph.nodes[n]["y"],
                    "lon": session.graph.nodes[n]["x"]
                }
                for n in route.current_path
            ] if route.current_path else []
        }


    def _plan_path_segment(self, graph: 'nx.MultiGraph', start: int, end: int,
                          used_edges: set[tuple[int, int]]) -> list[int]:
        """Plan shortest path avoiding used edges where possible."""
        try:
            # Create a copy of graph with used edges weighted heavily (not removed completely)
            temp_graph = graph.copy()

            # Increase weight of used edges to discourage reuse
            for edge in used_edges:
                if temp_graph.has_edge(edge[0], edge[1]):
                    for key in temp_graph[edge[0]][edge[1]]:
                        temp_graph[edge[0]][edge[1]][key]['length'] *= 10  # Heavy penalty

            import networkx as nx
            path = nx.shortest_path(temp_graph, start, end, weight='length')
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {start} to {end}")
            return []

    def _plan_shortest_disjunct_path(self, graph: 'nx.MultiGraph', start: int, end: int,
                                   used_edges: set[tuple[int, int]]) -> list[int]:
        """Plan path avoiding used edges completely for final return path."""
        try:
            # Create copy and remove used edges completely
            temp_graph = graph.copy()

            edges_to_remove = []
            for edge in used_edges:
                if temp_graph.has_edge(edge[0], edge[1]):
                    edges_to_remove.append((edge[0], edge[1]))

            temp_graph.remove_edges_from(edges_to_remove)

            import networkx as nx
            path = nx.shortest_path(temp_graph, start, end, weight='length')
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No disjunct path found from {start} to {end}")
            # Fallback: allow edge reuse with penalty
            return self._plan_path_segment(graph, start, end, used_edges)

    def _add_edges_to_used(self, route: RouteState, path: list[int]):
        """Add edges from path to used edges set."""
        for i in range(len(path) - 1):
            edge = (min(path[i], path[i+1]), max(path[i], path[i+1]))
            route.used_edges.add(edge)

    def _calculate_path_distance(self, graph: 'nx.MultiGraph', path: list[int]) -> float:
        """Calculate total distance of a path."""
        if len(path) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(path) - 1):
            try:
                edge_data = graph[path[i]][path[i+1]]
                # Get the shortest edge if multiple exist
                distance = min(data.get('length', 0) for data in edge_data.values())
                total_distance += distance
            except KeyError:
                logger.warning(f"No edge found between {path[i]} and {path[i+1]}")

        return total_distance

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 in degrees (0-360)."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)

        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # Normalize to 0-360

        return bearing

    def _select_diverse_candidates(self, candidates: list[RouteCandidate], from_lat: float, from_lon: float) -> list[RouteCandidate]:
        """Select 3 candidates that are spread across different directions for better diversity."""
        if len(candidates) <= 3:
            return candidates

        # Calculate bearings for all candidates
        candidates_with_bearing = []
        for candidate in candidates:
            bearing = self._calculate_bearing(from_lat, from_lon, candidate.lat, candidate.lon)
            candidates_with_bearing.append((candidate, bearing))

        # Sort by value score (descending)
        candidates_with_bearing.sort(key=lambda x: x[0].value_score, reverse=True)

        # Select diverse candidates
        selected = []
        selected_bearings = []

        for candidate, bearing in candidates_with_bearing:
            if len(selected) == 0:
                # First candidate: highest value score
                selected.append(candidate)
                selected_bearings.append(bearing)
            elif len(selected) < 3:
                # Check if this candidate is sufficiently different in direction
                min_bearing_diff = min(
                    min(abs(bearing - sb), 360 - abs(bearing - sb))
                    for sb in selected_bearings
                )

                # Require at least 60 degrees separation (360/6 for good spread)
                if min_bearing_diff >= 60:
                    selected.append(candidate)
                    selected_bearings.append(bearing)
                # If we can't find diverse candidates, add high-value ones anyway
                elif len(selected) < 3 and len(candidates_with_bearing) - len(selected) <= 3 - len(selected):
                    selected.append(candidate)
                    selected_bearings.append(bearing)

        # If we still don't have 3 candidates, fill with best remaining
        if len(selected) < 3:
            remaining = [c for c, _ in candidates_with_bearing if c not in selected]
            remaining.sort(key=lambda x: x.value_score, reverse=True)
            selected.extend(remaining[:3-len(selected)])

        logger.info(f"Selected {len(selected)} candidates with bearings: {[round(b) for b in selected_bearings[:len(selected)]]}")
        return selected

    def _calculate_polygon_area(self, coordinates: list[dict]) -> float:
        """Calculate the area of a polygon using the shoelace formula."""
        if len(coordinates) < 3:
            return 0.0

        # Convert to lat/lon lists
        lats = [coord["lat"] for coord in coordinates]
        lons = [coord["lon"] for coord in coordinates]

        # Ensure polygon is closed
        if lats[0] != lats[-1] or lons[0] != lons[-1]:
            lats.append(lats[0])
            lons.append(lons[0])

        # Convert to meters using approximate projection
        # This is a simplified calculation - for more accuracy, use proper projection
        avg_lat = sum(lats) / len(lats)
        lat_scale = 111000  # meters per degree latitude
        lon_scale = 111000 * math.cos(math.radians(avg_lat))  # meters per degree longitude

        # Convert to relative coordinates in meters
        x_coords = [(lon - lons[0]) * lon_scale for lon in lons]
        y_coords = [(lat - lats[0]) * lat_scale for lat in lats]

        # Shoelace formula
        area = 0.0
        n = len(x_coords)
        for i in range(n - 1):
            area += x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]

        return abs(area) / 2.0

    def _load_graph(self, lat: float, lon: float, radius: float = 8000) -> 'nx.MultiGraph':
        """Load OSM graph for the given location."""
        logger.info(f"Loading OSM graph for ({lat:.6f}, {lon:.6f}) with radius {radius}m")

        try:
            import osmnx as ox
            import networkx as nx
            
            # Use simplified filter for faster loading
            if radius <= 6000:
                # For smaller areas, use simplified filter for faster loading
                custom_filter = '["highway"~"footway|path|residential"]'
            else:
                # For larger areas, use full filter
                custom_filter = (
                    '["highway"~"path|track|footway|steps|bridleway|cycleway|'
                    'pedestrian|living_street|residential|tertiary"]'
                )
                
            # Set global timeout for requests
            import requests
            ox.settings.timeout = 60
            
            G = ox.graph_from_point(
                (lat, lon),
                dist=radius,
                network_type="walk",
                custom_filter=custom_filter
            )

            logger.info(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")

            # Use largest connected component
            if not nx.is_connected(G.to_undirected()):
                largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
                G = G.subgraph(largest_cc).copy()
                logger.info(f"Using largest connected component: {len(G.nodes())} nodes")

            return G

        except Exception as e:
            logger.error(f"Failed to load map data: {str(e)}")
            raise

    def _emergency_candidate_fallback(self, session: ClientSession, from_node: int, 
                                     from_lat: float, from_lon: float, target_radius: float,
                                     exclude_nodes: Set[int], existing_candidates: List) -> List:
        """
        Emergency fallback to ensure we always have 3 candidates.
        This method uses progressively more relaxed criteria to find candidates.
        """
        logger.warning(f"Emergency fallback: need 3 candidates but only have {len(existing_candidates)}")
        
        # Start with existing candidates
        result = existing_candidates.copy()
        used_nodes = {getattr(c, 'node_id', None) for c in result}
        used_nodes.discard(None)
        
        # Strategy 1: Expand radius progressively  
        radius_multipliers = [2.0, 3.0, 5.0, 10.0]
        
        for multiplier in radius_multipliers:
            if len(result) >= 3:
                break
                
            expanded_radius = target_radius * multiplier
            logger.debug(f"Emergency fallback: trying {multiplier}x radius ({expanded_radius:.0f}m)")
            
            # Find all nodes within expanded radius
            for node in session.graph.nodes():
                if len(result) >= 3:
                    break
                    
                if node in exclude_nodes or node in used_nodes:
                    continue
                
                node_data = session.graph.nodes[node]
                node_lat, node_lon = node_data['y'], node_data['x']
                
                distance = self._haversine_distance(from_lat, from_lon, node_lat, node_lon)
                if distance <= expanded_radius:
                    # Create basic candidate
                    try:
                        import osmnx as ox
                        route_candidate = RouteCandidate(
                            node_id=node,
                            lat=node_lat,
                            lon=node_lon,
                            value_score=0.3,  # Basic score
                            distance_from_current=distance,
                            estimated_route_completion=0.0,
                            explanation="Emergency fallback candidate",
                            semantic_scores={'basic': 0.3},
                            semantic_details="Generated via emergency fallback system"
                        )
                        result.append(route_candidate)
                        used_nodes.add(node)
                    except Exception as e:
                        logger.debug(f"Failed to create emergency candidate for node {node}: {e}")
                        continue
        
        # Strategy 2: If still not enough, use any remaining nodes (no distance limit)
        if len(result) < 3:
            logger.debug("Final emergency fallback: using any available nodes")
            
            all_nodes = list(session.graph.nodes())
            # Sort by distance from current position
            node_distances = []
            for node in all_nodes:
                if node in exclude_nodes or node in used_nodes:
                    continue
                    
                node_data = session.graph.nodes[node]
                node_lat, node_lon = node_data['y'], node_data['x']
                distance = self._haversine_distance(from_lat, from_lon, node_lat, node_lon)
                node_distances.append((node, distance, node_lat, node_lon))
            
            # Sort by distance and take the closest remaining nodes
            node_distances.sort(key=lambda x: x[1])
            
            for node, distance, node_lat, node_lon in node_distances:
                if len(result) >= 3:
                    break
                    
                try:
                    route_candidate = RouteCandidate(
                        node_id=node,
                        lat=node_lat,
                        lon=node_lon,
                        value_score=0.2,  # Lower score for emergency candidates
                        distance_from_current=distance,
                        estimated_route_completion=0.0,
                        explanation="Final emergency fallback candidate",
                        semantic_scores={'emergency': 0.2},
                        semantic_details="Generated via final emergency fallback"
                    )
                    result.append(route_candidate)
                    used_nodes.add(node)
                except Exception as e:
                    logger.debug(f"Failed to create final emergency candidate for node {node}: {e}")
                    continue
        
        logger.info(f"Emergency fallback completed: {len(result)} candidates (target: 3)")
        return result

    def get_cache_statistics(self) -> dict:
        """Get comprehensive cache statistics."""
        persistent_stats = self.persistent_cache.get_cache_statistics()

        return {
            "persistent_cache": persistent_stats,
            "memory_cache": {
                "legacy_cache_size": len(self.graph_cache),
                "active_sessions": len(self.client_sessions),
                "memory_cached_graphs": len(self.persistent_cache.memory_cache)
            },
            "session_info": [
                {
                    "client_id": session.client_id,
                    "created_at": session.created_at,
                    "last_access": session.last_access,
                    "graph_center": session.graph_center,
                    "graph_radius": session.graph_radius,
                    "has_active_route": session.active_route is not None,
                    "graph_nodes": len(session.graph.nodes()),
                    "graph_edges": len(session.graph.edges())
                }
                for session in self.client_sessions.values()
            ]
        }

    def cleanup_old_sessions(self, max_age_hours: float = 24.0):
        """Clean up old client sessions to free memory."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        sessions_to_remove = []
        for client_id, session in self.client_sessions.items():
            age = current_time - session.last_access
            if age > max_age_seconds:
                sessions_to_remove.append(client_id)

        for client_id in sessions_to_remove:
            del self.client_sessions[client_id]
            logger.info(f"Cleaned up old session for client {client_id}")

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
