"""
Clean Interactive Router for Perfect10k
Simple, fast router using the new clean candidate generation system.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from async_job_manager import JobPriority, job_manager, submit_cache_warming_job, submit_route_job
from clean_candidate_generator import CleanCandidateGenerator
from core.spatial_tile_storage import SpatialTileStorage
from loguru import logger
from performance_profiler import profile_function, profile_operation
from smart_cache_manager import cache_manager


@dataclass
class RouteCandidate:
    """A route candidate for the API."""

    node_id: int
    lat: float
    lon: float
    value_score: float
    explanation: str
    distance_from_current: float
    estimated_route_completion: float
    semantic_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ActiveRoute:
    """Active route being built by user."""

    start_location: dict[str, float]
    current_waypoints: list[int]
    current_path: list[int]
    total_distance: float
    target_distance: float
    preference: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ClientSession:
    """Client session for route building."""

    client_id: str
    graph_center: tuple
    active_route: ActiveRoute | None
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class CleanRouter:
    """
    Clean, simple interactive router using the new candidate generation system.

    Replaces the complex InteractiveRouteBuilder with a streamlined implementation
    focused on real-time performance and interpretable results.
    """

    def __init__(self, semantic_overlay_manager=None):
        """
        Initialize the clean router.

        Args:
            semantic_overlay_manager: Optional semantic overlay manager for nature features
        """
        self.semantic_overlay_manager = semantic_overlay_manager
        # Use proper storage location for spatial tiles
        self.spatial_storage = SpatialTileStorage(storage_dir="storage/tiles")
        self.client_sessions: dict[str, ClientSession] = {}
        self.candidate_generators: dict[str, CleanCandidateGenerator] = {}  # Per area

        logger.info("Initialized clean router")

        # Start async job manager
        self._async_ready_task = asyncio.create_task(self._ensure_async_ready())

    def start_route(
        self, client_id: str, lat: float, lon: float, preference: str, target_distance: int
    ) -> dict[str, Any]:
        """
        Start a new interactive route.

        Args:
            client_id: Unique client identifier
            lat: Starting latitude
            lon: Starting longitude
            preference: User preference text
            target_distance: Target route distance in meters

        Returns:
            Dictionary with initial candidates and session info
        """
        with profile_operation("start_route") as profiler:
            start_time = time.time()

            logger.info(f"Starting route for client {client_id} at ({lat:.6f}, {lon:.6f})")

            profiler.step("Load graph for area")
            # Load graph for the area
            graph = self._get_graph_for_location(lat, lon)
            if not graph or len(graph.nodes) == 0:
                raise ValueError(
                    f"No graph data available for location ({lat:.6f}, {lon:.6f}). "
                    f"Please load OSM data for this area first using the data loading utilities."
                )

            profiler.step("Initialize candidate generator")
            # Get or create candidate generator for this area
            area_key = f"{lat:.3f}_{lon:.3f}"  # Rough area grouping
            if area_key not in self.candidate_generators:
                self.candidate_generators[area_key] = CleanCandidateGenerator(
                    graph, self.semantic_overlay_manager, area_key
                )
                # Initialize in background if not already done
                if not self.candidate_generators[area_key].is_initialized:
                    init_success = self.candidate_generators[area_key].initialize()
                    if not init_success:
                        raise ValueError("Failed to initialize candidate generator")

            generator = self.candidate_generators[area_key]

            profiler.step("Find starting node")
            # Find starting node
            start_node = self._find_nearest_node(graph, lat, lon)
            if start_node is None:
                raise ValueError("No walkable node found near starting location")

            profiler.step("Create route session")
            # Create active route
            active_route = ActiveRoute(
                start_location={"lat": lat, "lon": lon},
                current_waypoints=[start_node],
                current_path=[start_node],
                total_distance=0.0,
                target_distance=float(target_distance),
                preference=preference,
            )

            # Create or update client session
            self.client_sessions[client_id] = ClientSession(
                client_id=client_id, graph_center=(lat, lon), active_route=active_route
            )

            profiler.step("Generate initial candidates")
            # Generate initial candidates
            step_distance = target_distance / 8.0
            logger.info(
                f"Generating candidates from ({lat:.6f}, {lon:.6f}) with step distance {step_distance:.0f}m"
            )
            logger.info(f"Start node: {start_node}, excluding nodes: {[start_node]}")

            candidates_result = generator.generate_candidates(
                from_lat=lat,
                from_lon=lon,
                target_distance=step_distance,  # First step is ~1/8 of total
                preference=preference,
                exclude_nodes=[start_node],
                existing_route_nodes=[],  # No existing route yet
            )

            logger.info(
                f"Candidate generation returned {len(candidates_result.candidates)} candidates"
            )
            if len(candidates_result.candidates) == 0:
                logger.warning(
                    f"Zero candidates returned! Search stats: {candidates_result.search_stats}"
                )
                logger.warning(f"Generation time: {candidates_result.generation_time_ms}ms")

            profiler.step("Convert to API format")
            # Convert to API format
            api_candidates = []
            for scored_candidate in candidates_result.candidates:
                # Convert feature scores to string keys for JSON
                feature_scores = {
                    feature_type.value: score
                    for feature_type, score in scored_candidate.feature_scores.items()
                }

                route_candidate = RouteCandidate(
                    node_id=scored_candidate.node_id,
                    lat=scored_candidate.lat,
                    lon=scored_candidate.lon,
                    value_score=scored_candidate.overall_score,
                    explanation=scored_candidate.explanation,
                    distance_from_current=self._calculate_distance(
                        lat, lon, scored_candidate.lat, scored_candidate.lon
                    ),
                    estimated_route_completion=target_distance * 0.15,  # Rough estimate
                    semantic_scores=feature_scores,
                )
                api_candidates.append(route_candidate)

            elapsed_ms = (time.time() - start_time) * 1000

            result = {
                "session_id": client_id,
                "start_location": {"lat": lat, "lon": lon},
                "candidates": [self._candidate_to_dict(c) for c in api_candidates],
                "route_stats": {
                    "current_distance": 0.0,
                    "target_distance": target_distance,
                    "waypoints_count": 1,
                    "completion_percentage": 0.0,
                },
                "generation_info": {
                    "generation_time_ms": candidates_result.generation_time_ms,
                    "total_time_ms": elapsed_ms,
                    "search_stats": candidates_result.search_stats,
                    "preference_analysis": candidates_result.preference_analysis,
                },
            }

            logger.info(
                f"Started route with {len(api_candidates)} candidates in {elapsed_ms:.1f}ms"
            )
            return result

    def add_waypoint(self, client_id: str, node_id: int) -> dict[str, Any]:
        """
        Add a waypoint to the current route.

        Args:
            client_id: Client identifier
            node_id: Node ID to add as waypoint

        Returns:
            Dictionary with updated route and new candidates
        """
        if client_id not in self.client_sessions:
            raise ValueError(f"Client session {client_id} not found")

        session = self.client_sessions[client_id]
        if not session.active_route:
            raise ValueError("No active route found")

        route = session.active_route

        # Get graph for this area
        lat, lon = session.graph_center
        graph = self._get_graph_for_location(lat, lon)

        if graph is None:
            raise ValueError("Failed to load graph for this area")

        # Get candidate generator
        area_key = f"{lat:.3f}_{lon:.3f}"
        if area_key not in self.candidate_generators:
            raise ValueError("Candidate generator not available for this area")

        generator = self.candidate_generators[area_key]

        # Get node coordinates
        if node_id not in graph.nodes:
            raise ValueError(f"Node {node_id} not found in graph")

        node_data = graph.nodes[node_id]
        node_lat, node_lon = node_data["y"], node_data["x"]

        # Calculate distance from last waypoint
        last_waypoint = route.current_waypoints[-1]
        last_data = graph.nodes[last_waypoint]
        last_lat, last_lon = last_data["y"], last_data["x"]

        self._calculate_distance(last_lat, last_lon, node_lat, node_lon)

        # Find actual path between last waypoint and new waypoint, avoiding already visited nodes when possible
        path_between = self._find_path_between_nodes(
            graph, last_waypoint, node_id, avoid_nodes=route.current_path
        )
        if not path_between:
            raise ValueError(f"No path found between waypoints {last_waypoint} and {node_id}")

        # Validate path connectivity
        self._validate_path_connectivity(
            graph, path_between, f"path from {last_waypoint} to {node_id}"
        )

        # Calculate actual path distance
        actual_distance = self._calculate_path_distance(graph, path_between)

        # Update route
        route.current_waypoints.append(node_id)
        # Add the path nodes (excluding the first one since it's already in current_path)
        route.current_path.extend(path_between[1:])
        route.total_distance += actual_distance
        session.last_access = time.time()

        # Calculate remaining distance for next candidates
        remaining_distance = route.target_distance - route.total_distance
        completion_percentage = (route.total_distance / route.target_distance) * 100

        # Use more generous step distance when close to completion or allow exceeding target
        if completion_percentage >= 85:
            # When close to completion, use a minimum step distance to ensure candidates exist
            # Allow routes to exceed target by up to 20% for better user experience
            min_step_distance = 400.0  # Minimum 400m to ensure candidates can be found
            base_step_distance = route.target_distance / 8.0  # Slightly larger base step
            next_step_distance = max(min_step_distance, base_step_distance)
            logger.info(
                f"Near completion ({completion_percentage:.1f}%), using generous step distance: {next_step_distance:.0f}m"
            )
        else:
            # Normal logic for early/mid route
            base_step = min(remaining_distance * 0.3, route.target_distance / 6.0)
            min_step_distance = 200.0  # Always use at least 200m step
            next_step_distance = max(min_step_distance, base_step)

        logger.info(
            f"Generating next candidates from {node_id} with remaining distance {remaining_distance:.0f}m, next step {next_step_distance:.0f}m"
        )

        # Generate next candidates
        candidates_result = generator.generate_candidates(
            from_lat=node_lat,
            from_lon=node_lon,
            target_distance=next_step_distance,
            preference=route.preference,
            exclude_nodes=list(route.current_waypoints),
            existing_route_nodes=route.current_path,  # Pass entire route path for proximity filtering
        )

        # Convert to API format
        api_candidates = []
        for scored_candidate in candidates_result.candidates:
            # Convert feature scores to string keys for JSON
            feature_scores = {
                feature_type.value: score
                for feature_type, score in scored_candidate.feature_scores.items()
            }

            route_candidate = RouteCandidate(
                node_id=scored_candidate.node_id,
                lat=scored_candidate.lat,
                lon=scored_candidate.lon,
                value_score=scored_candidate.overall_score,
                explanation=scored_candidate.explanation,
                distance_from_current=self._calculate_distance(
                    node_lat, node_lon, scored_candidate.lat, scored_candidate.lon
                ),
                estimated_route_completion=self._calculate_estimated_completion(
                    route.total_distance,
                    route.target_distance,
                    completion_percentage,
                    next_step_distance,
                ),
                semantic_scores=feature_scores,
            )
            api_candidates.append(route_candidate)

        completion_percentage = (route.total_distance / route.target_distance) * 100

        # Generate route coordinates using actual way geometry for better alignment
        route_coordinates = self._extract_route_geometry(graph, route.current_path)

        result = {
            "waypoint_added": {
                "node_id": node_id,
                "lat": node_lat,
                "lon": node_lon,
                "distance_added": actual_distance,
            },
            "candidates": [self._candidate_to_dict(c) for c in api_candidates],
            "route_stats": {
                "current_distance": route.total_distance,
                "target_distance": route.target_distance,
                "waypoints_count": len(route.current_waypoints),
                "completion_percentage": completion_percentage,
            },
            "route_coordinates": route_coordinates,
            "current_path": route.current_path,  # Node IDs for debugging
            "generation_info": {
                "generation_time_ms": candidates_result.generation_time_ms,
                "search_stats": candidates_result.search_stats,
            },
        }

        logger.info(
            f"Added waypoint {node_id}, route now {route.total_distance:.0f}m ({completion_percentage:.1f}%)"
        )
        return result

    def finalize_route(self, client_id: str, final_node_id: int) -> dict[str, Any]:
        """
        Finalize the route by adding final waypoint and connecting back to start.

        Args:
            client_id: Client identifier
            final_node_id: Final destination node ID

        Returns:
            Dictionary with finalized route information
        """
        if client_id not in self.client_sessions:
            raise ValueError(f"Client session {client_id} not found")

        session = self.client_sessions[client_id]
        if not session.active_route:
            raise ValueError("No active route found")

        route = session.active_route

        # Get graph for this area
        lat, lon = session.graph_center
        graph = self._get_graph_for_location(lat, lon)

        if not graph or final_node_id not in graph.nodes:
            raise ValueError(f"Final node {final_node_id} not found in graph")

        # Get coordinates of final node
        final_data = graph.nodes[final_node_id]
        final_lat, final_lon = final_data["y"], final_data["x"]

        # Find actual path from last waypoint to final destination, avoiding visited nodes when possible
        last_waypoint = route.current_waypoints[-1]
        path_to_final = self._find_path_between_nodes(
            graph, last_waypoint, final_node_id, avoid_nodes=route.current_path
        )
        if not path_to_final:
            raise ValueError(
                f"No walkable path found from waypoint {last_waypoint} to final destination {final_node_id}"
            )

        # Find the start node (closest to start location)
        start_lat, start_lon = route.start_location["lat"], route.start_location["lon"]
        start_node = self._find_nearest_node(graph, start_lat, start_lon)
        if not start_node:
            raise ValueError("Cannot find start node to close the loop")

        # For the return path, avoid the route we've built so far but allow some overlap for closing the loop
        extended_path = route.current_path + path_to_final[1:]  # Current path + path to final

        # First try: Avoid most of the existing route to prevent out-and-back
        path_to_start = self._find_path_between_nodes(
            graph, final_node_id, start_node, avoid_nodes=extended_path[:-5]
        )

        # Check if this creates an out-and-back route (>70% path overlap)
        if path_to_start and self._is_out_and_back_route(extended_path, path_to_start):
            logger.warning("Detected potential out-and-back route, trying alternative path")
            # Try with less aggressive avoidance to find a different route
            path_to_start = self._find_alternative_return_path(
                graph, final_node_id, start_node, extended_path
            )

        if not path_to_start:
            raise ValueError(
                f"No walkable path found from final destination {final_node_id} back to start {start_node}"
            )

        # Calculate actual distances along paths
        distance_to_final = self._calculate_path_distance(graph, path_to_final)
        distance_to_start = self._calculate_path_distance(graph, path_to_start)

        logger.info(
            f"Finalize route pathfinding: {len(path_to_final)} nodes to final, {len(path_to_start)} nodes back to start"
        )
        logger.info(
            f"Path distances: {distance_to_final:.0f}m to final, {distance_to_start:.0f}m back to start"
        )

        # Analyze final route quality
        total_path = route.current_path + path_to_final[1:] + path_to_start[1:]
        unique_nodes = len(set(total_path))
        total_nodes = len(total_path)
        route_efficiency = unique_nodes / max(total_nodes, 1)

        logger.info(
            f"Route quality: {unique_nodes}/{total_nodes} unique nodes ({route_efficiency:.1%} efficiency)"
        )

        # Validate that paths are connected before updating route
        self._validate_path_connectivity(graph, path_to_final, "path to final")
        self._validate_path_connectivity(graph, path_to_start, "path back to start")

        # Update route with actual paths
        route.current_waypoints.append(final_node_id)
        # Add path to final (excluding first node since it's already in current_path)
        route.current_path.extend(path_to_final[1:])
        # Add path back to start (excluding first node which is the final_node_id)
        route.current_path.extend(path_to_start[1:])
        route.total_distance += distance_to_final + distance_to_start
        session.last_access = time.time()

        completion_percentage = (route.total_distance / route.target_distance) * 100

        # Generate route coordinates using actual way geometry for better alignment
        route_coordinates = self._extract_route_geometry(graph, route.current_path)

        # The path back to start should already be included in current_path, no need to add separate coordinate

        result = {
            "route_completed": True,
            "final_waypoint": {
                "node_id": final_node_id,
                "lat": final_lat,
                "lon": final_lon,
                "distance_added": distance_to_final + distance_to_start,
            },
            "route_stats": {
                "current_distance": route.total_distance,
                "total_distance": route.total_distance,  # Add missing field
                "target_distance": route.target_distance,
                "waypoints_count": len(route.current_waypoints),
                "completion_percentage": completion_percentage,
            },
            "route_coordinates": route_coordinates,
            "total_waypoints": len(route.current_waypoints),
            "route_summary": {
                "start_location": route.start_location,
                "preference": route.preference,
                "created_at": route.created_at,
                "finalized_at": time.time(),
            },
            "route_quality": {
                "total_nodes": total_nodes,
                "unique_nodes": unique_nodes,
                "efficiency": route_efficiency,
                "path_to_final_nodes": len(path_to_final),
                "path_to_start_nodes": len(path_to_start),
                "is_loop": path_to_start[-1] == route.current_path[0]
                if path_to_start and route.current_path
                else False,
            },
        }

        logger.info(
            f"Finalized route for client {client_id}: {route.total_distance:.0f}m with {len(route.current_waypoints)} waypoints"
        )
        return result

    def get_route_status(self, client_id: str) -> dict[str, Any]:
        """Get current route status."""
        if client_id not in self.client_sessions:
            raise ValueError(f"Client session {client_id} not found")

        session = self.client_sessions[client_id]
        if not session.active_route:
            raise ValueError("No active route found")

        route = session.active_route
        completion_percentage = (route.total_distance / route.target_distance) * 100

        return {
            "session_id": client_id,
            "route_stats": {
                "current_distance": route.total_distance,
                "target_distance": route.target_distance,
                "waypoints_count": len(route.current_waypoints),
                "completion_percentage": completion_percentage,
            },
            "start_location": route.start_location,
            "preference": route.preference,
            "created_at": route.created_at,
            "last_access": session.last_access,
        }

    # === ASYNC METHODS FOR BACKGROUND PROCESSING ===

    async def _ensure_async_ready(self):
        """Ensure async job manager is running."""
        try:
            await job_manager.start()
        except Exception as e:
            logger.error(f"Failed to start async job manager: {e}")

    async def start_route_async(
        self, client_id: str, lat: float, lon: float, preference: str, target_distance: int
    ) -> dict[str, Any]:
        """
        Async route start with immediate response + background processing.

        Returns immediate response with cached data, then submits background
        jobs for cache warming and route refinement.
        """
        logger.info(f"🚀 Async route start for client {client_id} at ({lat:.6f}, {lon:.6f})")

        # Step 1: Try immediate response from cache
        try:
            cached_result = await self._try_immediate_cached_response(
                client_id, lat, lon, preference, target_distance
            )

            if cached_result:
                # Start background cache warming for nearby areas
                await self._start_background_cache_warming(lat, lon)

                cached_result["response_type"] = "cached"
                cached_result["background_jobs"] = "started"
                return cached_result

        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

        # Step 2: Submit background job for full processing
        job_id = await submit_route_job(
            "start_route_background",
            self._background_route_generation,
            client_id,
            lat,
            lon,
            preference,
            target_distance,
            priority=JobPriority.HIGH,
        )

        # Step 3: Start cache warming in parallel
        await self._start_background_cache_warming(lat, lon)

        # Return job tracking info
        return {
            "session_id": client_id,
            "response_type": "async",
            "job_id": job_id,
            "status": "processing",
            "background_jobs": "started",
            "estimated_completion_ms": 30000,  # 30 seconds estimate
            "polling_interval_ms": 1000,  # Check every 1 second
        }

    async def _try_immediate_cached_response(
        self, client_id: str, lat: float, lon: float, preference: str, target_distance: int
    ) -> dict[str, Any] | None:
        """Try to provide immediate response from cache."""

        # Check if we have a cached graph for this area
        graph = cache_manager.get_cached_graph(lat, lon)
        if not graph:
            return None

        logger.info(f"🚀 Fast cached response for {client_id}")

        # Check if we have a cached candidate generator
        area_key = f"{lat:.3f}_{lon:.3f}"
        if area_key not in self.candidate_generators:
            # Try to initialize quickly from cache
            generator = CleanCandidateGenerator(graph, self.semantic_overlay_manager)
            # Note: quick_init_from_cache is not implemented yet
            # Just skip if not already cached
            return None
        else:
            generator = self.candidate_generators[area_key]

        # Generate candidates quickly
        try:
            start_time = time.time()

            # Find starting node
            start_node = self._find_nearest_node(graph, lat, lon)
            if start_node is None:
                return None

            # Create session
            active_route = ActiveRoute(
                start_location={"lat": lat, "lon": lon},
                current_waypoints=[start_node],
                current_path=[start_node],
                total_distance=0.0,
                target_distance=float(target_distance),
                preference=preference,
            )

            self.client_sessions[client_id] = ClientSession(
                client_id=client_id, graph_center=(lat, lon), active_route=active_route
            )

            # Generate candidates
            candidates_result = generator.generate_candidates(
                from_lat=lat,
                from_lon=lon,
                target_distance=target_distance / 8.0,
                preference=preference,
                exclude_nodes=[start_node],
                existing_route_nodes=[],
            )

            # Convert to API format
            api_candidates = []
            for scored_candidate in candidates_result.candidates:
                feature_scores = {
                    feature_type.value: score
                    for feature_type, score in scored_candidate.feature_scores.items()
                }

                route_candidate = RouteCandidate(
                    node_id=scored_candidate.node_id,
                    lat=scored_candidate.lat,
                    lon=scored_candidate.lon,
                    value_score=scored_candidate.overall_score,
                    explanation=scored_candidate.explanation,
                    distance_from_current=self._calculate_distance(
                        lat, lon, scored_candidate.lat, scored_candidate.lon
                    ),
                    estimated_route_completion=target_distance * 0.15,
                    semantic_scores=feature_scores,
                )
                api_candidates.append(route_candidate)

            elapsed_ms = (time.time() - start_time) * 1000

            result = {
                "session_id": client_id,
                "start_location": {"lat": lat, "lon": lon},
                "candidates": [self._candidate_to_dict(c) for c in api_candidates],
                "route_stats": {
                    "current_distance": 0.0,
                    "target_distance": target_distance,
                    "waypoints_count": 1,
                    "completion_percentage": 0.0,
                },
                "generation_info": {
                    "generation_time_ms": candidates_result.generation_time_ms,
                    "total_time_ms": elapsed_ms,
                    "search_stats": candidates_result.search_stats,
                    "preference_analysis": candidates_result.preference_analysis,
                },
            }

            logger.info(
                f"⚡ Cached response in {elapsed_ms:.1f}ms with {len(api_candidates)} candidates"
            )
            return result

        except Exception as e:
            logger.warning(f"Fast cached response failed: {e}")
            return None

    def _background_route_generation(
        self, client_id: str, lat: float, lon: float, preference: str, target_distance: int
    ) -> dict[str, Any]:
        """Background route generation for async processing."""
        logger.info(f"🔄 Background route generation for {client_id}")

        # This is the full route generation (same as sync version)
        return self.start_route(client_id, lat, lon, preference, target_distance)

    async def _start_background_cache_warming(self, center_lat: float, center_lon: float):
        """Start background cache warming for nearby areas."""

        # Define warming radius (areas around current location)
        warming_offsets = [
            (0.01, 0.01),  # NE
            (0.01, -0.01),  # NW
            (-0.01, 0.01),  # SE
            (-0.01, -0.01),  # SW
            (0.02, 0.0),  # N
            (-0.02, 0.0),  # S
            (0.0, 0.02),  # E
            (0.0, -0.02),  # W
        ]

        warming_jobs = []

        for lat_offset, lon_offset in warming_offsets:
            warm_lat = center_lat + lat_offset
            warm_lon = center_lon + lon_offset

            # Check if already cached
            if cache_manager.get_cached_graph(warm_lat, warm_lon):
                continue  # Skip if already cached

            # Submit warming job
            job_id = await submit_cache_warming_job(warm_lat, warm_lon, self._warm_area_cache)
            warming_jobs.append(job_id)

        if warming_jobs:
            logger.info(
                f"🔥 Started {len(warming_jobs)} cache warming jobs around ({center_lat:.6f}, {center_lon:.6f})"
            )
        else:
            logger.debug(
                f"✅ All nearby areas already cached for ({center_lat:.6f}, {center_lon:.6f})"
            )

        return warming_jobs

    def _warm_area_cache(self, lat: float, lon: float) -> bool:
        """Warm cache for a specific area."""
        try:
            logger.info(f"🔥 Warming cache for area ({lat:.6f}, {lon:.6f})")

            # Load graph (will be cached automatically)
            graph = self._get_graph_for_location(lat, lon)
            if not graph:
                logger.warning(f"Failed to load graph for warming at ({lat:.6f}, {lon:.6f})")
                return False

            # Initialize candidate generator (will cache features)
            area_key = f"{lat:.3f}_{lon:.3f}"
            if area_key not in self.candidate_generators:
                generator = CleanCandidateGenerator(graph, self.semantic_overlay_manager)
                if generator.initialize():
                    self.candidate_generators[area_key] = generator
                    logger.info(
                        f"✅ Cache warmed for area ({lat:.6f}, {lon:.6f}) - {len(graph.nodes)} nodes"
                    )
                    return True
                else:
                    logger.warning(
                        f"Failed to initialize generator for warming at ({lat:.6f}, {lon:.6f})"
                    )
                    return False
            else:
                logger.debug(f"✅ Area ({lat:.6f}, {lon:.6f}) already initialized")
                return True

        except Exception as e:
            logger.error(f"Cache warming failed for ({lat:.6f}, {lon:.6f}): {e}")
            return False

    async def get_job_status_async(self, job_id: str) -> dict[str, Any] | None:
        """Get async job status with progress."""
        result = job_manager.get_job_status(job_id)
        if not result:
            return None

        return {
            "job_id": job_id,
            "status": result.status.value,
            "progress": result.progress,
            "result": result.result,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "created_at": result.created_at,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
        }

    async def wait_for_job_async(self, job_id: str, timeout: float = 30.0) -> dict[str, Any] | None:
        """Wait for job completion with timeout."""
        result = await job_manager.wait_for_job(job_id, timeout)
        if result:
            return await self.get_job_status_async(job_id)
        return None

    # === END ASYNC METHODS ===

    def get_scoring_visualization_data(
        self, client_id: str, score_type: str = "overall"
    ) -> dict[str, Any]:
        """
        Get scored nodes data for semantic overlay visualization.

        Args:
            client_id: Client session ID
            score_type: Type of score to visualize ("overall", "nature", "water", etc.)

        Returns:
            Dictionary with scored nodes for map visualization
        """
        if client_id not in self.client_sessions:
            return {
                "error": "Client session not found",
                "nodes": [],
                "score_range": {"min": 0, "max": 1},
                "total_nodes": 0,
            }

        session = self.client_sessions[client_id]
        if not session.active_route:
            return {
                "error": "No active route found",
                "nodes": [],
                "score_range": {"min": 0, "max": 1},
                "total_nodes": 0,
            }

        # Get the area key for this session
        lat, lon = session.graph_center
        area_key = f"{lat:.3f}_{lon:.3f}"

        if area_key not in self.candidate_generators:
            return {
                "error": "No candidate generator found for this area",
                "nodes": [],
                "score_range": {"min": 0, "max": 1},
                "total_nodes": 0,
            }

        generator = self.candidate_generators[area_key]
        visualization_data = generator.get_scored_nodes_for_visualization(score_type)

        # Add session context
        visualization_data["client_id"] = client_id
        visualization_data["area_center"] = {"lat": lat, "lon": lon}
        visualization_data["route_preference"] = session.active_route.preference

        return visualization_data

    def _has_immediate_cache(self, lat: float, lon: float, preference: str) -> bool:
        """Fast check if we have immediate cache data for a location without expensive loading."""
        try:
            # Check if candidate generator is already initialized
            area_key = f"{lat:.3f}_{lon:.3f}"
            return area_key in self.candidate_generators
        except Exception:
            return False

    @profile_function("get_graph_for_location")
    def _get_graph_for_location(self, lat: float, lon: float) -> nx.MultiGraph | None:
        """Get graph data for a location using smart caching with fallback to spatial tile storage."""
        try:
            # First, try smart cache for instant loading
            graph = cache_manager.get_cached_graph(lat, lon)
            if graph is not None:
                logger.info(
                    f"🚀 Graph loaded from cache with {len(graph.nodes)} nodes for ({lat:.6f}, {lon:.6f})"
                )
                return graph

            # Cache miss - load from spatial tile storage
            logger.info(f"📥 Cache miss, loading graph from tiles for ({lat:.6f}, {lon:.6f})")
            graph = self.spatial_storage.load_graph_for_area(lat, lon, radius_m=5000)

            if graph is None:
                logger.error(f"Failed to load or create graph data for ({lat:.6f}, {lon:.6f})")
                return None

            # Store in smart cache for next time
            cache_manager.store_graph(lat, lon, graph)
            logger.info(
                f"💾 Graph cached and loaded with {len(graph.nodes)} nodes for ({lat:.6f}, {lon:.6f})"
            )
            return graph

        except Exception as e:
            logger.error(f"Exception while loading graph for ({lat:.6f}, {lon:.6f}): {e}")
            return None

    def _find_nearest_node(self, graph: nx.MultiGraph, lat: float, lon: float) -> int | None:
        """Find the nearest walkable node to a location."""
        min_distance = float("inf")
        nearest_node = None

        for node_id, data in graph.nodes(data=True):
            node_lat, node_lon = data["y"], data["x"]
            distance = self._calculate_distance(lat, lon, node_lat, node_lon)

            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id

        return nearest_node

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters."""
        import math

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

    def _find_path_between_nodes(
        self,
        graph: nx.MultiGraph,
        start_node: int,
        end_node: int,
        avoid_nodes: list[int] | None = None,
    ) -> list[int]:
        """Find shortest path between two nodes using NetworkX, optionally avoiding certain nodes."""
        try:
            # First try to find a path avoiding already visited nodes
            if avoid_nodes:
                # Create a subgraph excluding avoided nodes (except start and end)
                nodes_to_remove = [n for n in avoid_nodes if n != start_node and n != end_node]
                if nodes_to_remove:
                    subgraph = graph.copy()
                    subgraph.remove_nodes_from(nodes_to_remove)

                    try:
                        # Try with length weights on subgraph
                        path = nx.shortest_path(subgraph, start_node, end_node, weight="length")
                        logger.debug(f"Found path avoiding {len(nodes_to_remove)} visited nodes")
                        return path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        try:
                            # Try unweighted on subgraph
                            path = nx.shortest_path(subgraph, start_node, end_node)
                            logger.debug(
                                f"Found unweighted path avoiding {len(nodes_to_remove)} visited nodes"
                            )
                            return path
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            logger.debug(
                                "No path found avoiding visited nodes, falling back to full graph"
                            )

            # Fallback to full graph with length weights
            path = nx.shortest_path(graph, start_node, end_node, weight="length")
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning(
                f"No weighted path found between {start_node} and {end_node}, trying unweighted: {e}"
            )
            try:
                # Final fallback to unweighted path on full graph
                path = nx.shortest_path(graph, start_node, end_node)
                return path
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e2:
                logger.error(f"No path found between {start_node} and {end_node}: {e2}")
                return []

    def _calculate_path_distance(self, graph: nx.MultiGraph, path: list[int]) -> float:
        """Calculate total distance of a path through the graph."""
        if len(path) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]

            # Get edge data (handle MultiGraph with multiple edges)
            edge_data = graph[node1][node2]
            if isinstance(edge_data, dict):
                # Multiple edges - use the first one or find shortest
                edge_lengths = [edge.get("length", 0) for edge in edge_data.values()]
                edge_length = min(edge_lengths) if edge_lengths else 0
            else:
                edge_length = edge_data.get("length", 0)

            # If no length data, calculate from coordinates
            if edge_length == 0:
                node1_data = graph.nodes[node1]
                node2_data = graph.nodes[node2]
                edge_length = self._calculate_distance(
                    node1_data["y"], node1_data["x"], node2_data["y"], node2_data["x"]
                )

            total_distance += edge_length

        return total_distance

    def _calculate_estimated_completion(
        self,
        current_distance: float,
        target_distance: float,
        completion_percentage: float,
        next_step_distance: float,
    ) -> float:
        """Calculate estimated route completion distance accounting for ability to exceed target."""

        if completion_percentage >= 85:
            # When close to completion, estimate based on next step plus some buffer
            # Since we allow exceeding target by up to 20%, show realistic estimate
            estimated_final = current_distance + next_step_distance
            max_allowed = target_distance * 1.2  # 20% overage allowed
            return min(estimated_final, max_allowed)
        else:
            # Normal estimation - assume we'll use about 70% of remaining distance
            remaining_distance = target_distance - current_distance
            return current_distance + (remaining_distance * 0.7)

    def _validate_path_connectivity(self, graph: nx.MultiGraph, path: list[int], path_name: str):
        """Validate that all consecutive nodes in a path are connected by edges."""
        if len(path) < 2:
            return

        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            if not graph.has_edge(node1, node2):
                raise ValueError(
                    f"Invalid path in {path_name}: nodes {node1} and {node2} are not connected by an edge"
                )

        logger.debug(f"Validated {path_name}: all {len(path) - 1} edges are connected")

    def _is_out_and_back_route(self, outbound_path: list[int], return_path: list[int]) -> bool:
        """Detect if the return path largely retraces the outbound path (out-and-back route)."""
        if len(return_path) < 3 or len(outbound_path) < 3:
            return False

        # Reverse the return path to check for overlap with outbound
        reversed_return = list(reversed(return_path))

        # Count overlapping nodes
        outbound_set = set(outbound_path)
        return_set = set(reversed_return)
        overlap = len(outbound_set.intersection(return_set))

        # Calculate overlap percentage
        total_unique_nodes = len(outbound_set.union(return_set))
        overlap_percentage = overlap / max(total_unique_nodes, 1)

        logger.debug(
            f"Route overlap analysis: {overlap}/{total_unique_nodes} nodes ({overlap_percentage:.1%})"
        )

        # Consider it out-and-back if >60% overlap
        return overlap_percentage > 0.6

    def _find_alternative_return_path(
        self, graph: nx.MultiGraph, start_node: int, end_node: int, outbound_path: list[int]
    ) -> list[int]:
        """Find an alternative return path that minimizes retracing the outbound route."""

        # Strategy 1: Try different intermediate waypoints to force a different route
        intermediate_candidates = []

        # Find nodes that are roughly perpendicular to the outbound direction
        start_data = graph.nodes[start_node]
        end_data = graph.nodes[end_node]
        start_lat, start_lon = start_data["y"], start_data["x"]
        end_lat, end_lon = end_data["y"], end_data["x"]

        # Get nearby nodes as potential intermediate waypoints
        for node_id in graph.nodes():
            if node_id in outbound_path or node_id in (start_node, end_node):
                continue

            node_data = graph.nodes[node_id]
            node_lat, node_lon = node_data["y"], node_data["x"]

            # Check if node is within reasonable distance
            dist_to_start = self._calculate_distance(start_lat, start_lon, node_lat, node_lon)
            dist_to_end = self._calculate_distance(end_lat, end_lon, node_lat, node_lon)

            if 200 <= dist_to_start <= 1000 and 200 <= dist_to_end <= 1000:
                intermediate_candidates.append((node_id, dist_to_start + dist_to_end))

        # Sort by total distance and try a few options
        intermediate_candidates.sort(key=lambda x: x[1])

        for intermediate_node, _ in intermediate_candidates[:5]:
            try:
                # Try path: start -> intermediate -> end
                path1 = self._find_path_between_nodes(graph, start_node, intermediate_node)
                if not path1:
                    continue

                path2 = self._find_path_between_nodes(graph, intermediate_node, end_node)
                if not path2:
                    continue

                # Combine paths
                alternative_path = path1 + path2[1:]  # Remove duplicate intermediate node

                # Check if this is better (less overlap)
                if not self._is_out_and_back_route(outbound_path, alternative_path):
                    logger.info(
                        f"Found alternative return path via intermediate node {intermediate_node}"
                    )
                    return alternative_path

            except Exception as e:
                logger.debug(f"Failed to route via intermediate {intermediate_node}: {e}")
                continue

        # Strategy 2: If no good intermediate found, try with reduced avoidance
        logger.warning("No good intermediate path found, trying with reduced avoidance")
        avoid_nodes = (
            outbound_path[:-10] if len(outbound_path) > 10 else []
        )  # Avoid less of the route

        return self._find_path_between_nodes(graph, start_node, end_node, avoid_nodes=avoid_nodes)

    def _extract_route_geometry(
        self, graph: nx.MultiGraph, path: list[int]
    ) -> list[dict[str, float]]:
        """Extract detailed route coordinates using OSM way geometry for better map alignment."""
        if len(path) < 2:
            # Single node, just return its coordinates
            if path and path[0] in graph.nodes:
                node_data = graph.nodes[path[0]]
                return [{"lat": node_data["y"], "lon": node_data["x"]}]
            return []

        route_coordinates = []
        edges_with_geometry = 0
        total_edges = len(path) - 1

        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]

            if node1 not in graph.nodes or node2 not in graph.nodes:
                continue

            # Add first node coordinates (only for first segment)
            if i == 0:
                node1_data = graph.nodes[node1]
                route_coordinates.append({"lat": node1_data["y"], "lon": node1_data["x"]})

            # Get edge data with geometry
            try:
                edge_data = graph[node1][node2]

                # Handle MultiGraph (multiple edges between same nodes)
                edge_geom = None
                if isinstance(edge_data, dict):
                    # Multiple edges - find one with geometry
                    for edge_attrs in edge_data.values():
                        if "geometry" in edge_attrs and edge_attrs["geometry"] is not None:
                            edge_geom = edge_attrs["geometry"]
                            break
                else:
                    # Single edge
                    edge_geom = edge_data.get("geometry")

                if edge_geom is not None:
                    edges_with_geometry += 1
                    # Extract coordinates from geometry (Shapely LineString)
                    try:
                        # Get coordinate pairs from the LineString
                        coords = list(edge_geom.coords)

                        # Add intermediate points (skip first point to avoid duplication)
                        for coord in coords[1:]:
                            lon, lat = coord[0], coord[1]  # Shapely uses (lon, lat) order
                            route_coordinates.append({"lat": lat, "lon": lon})

                    except Exception as e:
                        logger.debug(f"Failed to extract geometry from edge {node1}-{node2}: {e}")
                        # Fallback to end node coordinates
                        node2_data = graph.nodes[node2]
                        route_coordinates.append({"lat": node2_data["y"], "lon": node2_data["x"]})
                else:
                    # No geometry available, use end node coordinates
                    node2_data = graph.nodes[node2]
                    route_coordinates.append({"lat": node2_data["y"], "lon": node2_data["x"]})

            except Exception as e:
                logger.debug(f"Failed to get edge data for {node1}-{node2}: {e}")
                # Fallback to end node coordinates
                if node2 in graph.nodes:
                    node2_data = graph.nodes[node2]
                    route_coordinates.append({"lat": node2_data["y"], "lon": node2_data["x"]})

        geometry_percentage = (edges_with_geometry / total_edges * 100) if total_edges > 0 else 0
        logger.info(
            f"Route geometry: {edges_with_geometry}/{total_edges} edges have geometry ({geometry_percentage:.1f}%)"
        )
        logger.info(
            f"Extracted {len(route_coordinates)} coordinate points from {len(path)} path nodes"
        )

        if geometry_percentage < 50:
            logger.warning(
                f"Low geometry coverage ({geometry_percentage:.1f}%) - route may show straight line segments"
            )

        return route_coordinates

    def _candidate_to_dict(self, candidate: RouteCandidate) -> dict[str, Any]:
        """Convert RouteCandidate to dictionary for API."""
        return {
            "node_id": candidate.node_id,
            "lat": candidate.lat,
            "lon": candidate.lon,
            "value_score": candidate.value_score,
            "explanation": candidate.explanation,
            "distance_from_current": candidate.distance_from_current,
            "estimated_completion": candidate.estimated_route_completion,
            "feature_scores": candidate.semantic_scores,  # Detailed feature breakdown
            "semantic_scores": candidate.semantic_scores,  # Keep for backward compatibility
        }

    def cleanup_old_sessions(self, max_age_hours: float = 24.0):
        """Clean up old client sessions."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        sessions_to_remove = []
        for client_id, session in self.client_sessions.items():
            if current_time - session.last_access > max_age_seconds:
                sessions_to_remove.append(client_id)

        for client_id in sessions_to_remove:
            del self.client_sessions[client_id]

        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def get_statistics(self) -> dict[str, Any]:
        """Get router statistics."""
        cache_stats = cache_manager.get_cache_statistics()
        return {
            "active_sessions": len(self.client_sessions),
            "candidate_generators": len(self.candidate_generators),
            "generator_stats": {
                area_key: generator.get_statistics()
                for area_key, generator in self.candidate_generators.items()
            },
            "cache_performance": {
                "hit_rate": cache_stats["overall_hit_rate"],
                "total_requests": cache_stats["total_requests"],
                "cache_sizes": cache_stats["cache_sizes"],
            },
        }
