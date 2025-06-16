"""
Perfect10k Backend - Interactive Route Builder
New simplified approach for user-driven route construction.
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

# Lazy imports to improve startup performance
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from interactive_router import InteractiveRouteBuilder
    from semantic_overlays import SemanticOverlayManager, BoundingBox

# Configure minimal logging for faster startup (file logging configured on first use)
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),  # Console output only during startup
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> | {message}",
)

# File logging will be added on first API call to avoid startup I/O
_file_logging_configured = False

def ensure_file_logging():
    """Configure file logging on first use to improve startup performance."""
    global _file_logging_configured
    if not _file_logging_configured:
        logger.add(
            "perfect10k.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )
        _file_logging_configured = True

app = FastAPI(
    title="Perfect10k Interactive Route Builder",
    description="User-driven route planning with step-by-step construction",
    version="2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend (only if directory exists)
if Path("../frontend").exists():
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Lazy-loaded global instances to improve startup performance
route_builder = None
overlay_manager = None


def get_route_builder():
    """Get route builder instance (lazy initialization for faster startup)."""
    global route_builder
    if route_builder is None:
        from interactive_router import InteractiveRouteBuilder
        route_builder = InteractiveRouteBuilder()
    return route_builder


def get_overlay_manager():
    """Get overlay manager instance (lazy initialization for faster startup)."""
    global overlay_manager
    if overlay_manager is None:
        from semantic_overlays import SemanticOverlayManager
        overlay_manager = SemanticOverlayManager()
    return overlay_manager


# Request/Response Models
class StartRouteRequest(BaseModel):
    lat: float
    lon: float
    preference: str = "scenic parks and nature"
    target_distance: int = 8000
    client_id: str | None = None  # Auto-generated if not provided


class AddWaypointRequest(BaseModel):
    session_id: str
    node_id: int


class FinalizeRouteRequest(BaseModel):
    session_id: str
    final_node_id: int


class SemanticOverlayRequest(BaseModel):
    lat: float
    lon: float
    radius_km: float = 2.0
    feature_types: list[str] = ["forests", "rivers", "lakes"]
    use_cache: bool = True


class SemanticScoringRequest(BaseModel):
    locations: list[tuple[float, float]]  # List of (lat, lon) tuples
    property_names: list[str] = ["forests", "rivers", "lakes"]
    ensure_loaded_radius: float = 2.0  # km radius to ensure features are loaded


# Utility function to generate client ID
def get_client_id(request: Request, provided_id: str | None = None) -> str:
    """Generate or use provided client ID."""
    if provided_id:
        return provided_id

    # Generate client ID based on IP and user agent for consistency
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    return f"{client_ip}_{hash(user_agent) % 10000}"

# API Endpoints
@app.post("/api/start-session")
async def start_session(request_data: StartRouteRequest, request: Request):
    """Initialize a new interactive routing session (optimized with caching)."""
    ensure_file_logging()  # Configure file logging on first API call
    client_id = get_client_id(request, request_data.client_id)
    logger.info(f"Starting route for client {client_id} at ({request_data.lat:.6f}, {request_data.lon:.6f})")

    try:
        result = get_route_builder().start_route(
            client_id=client_id,
            lat=request_data.lat,
            lon=request_data.lon,
            preference=request_data.preference,
            target_distance=request_data.target_distance
        )

        # Add loading state information to help frontend show appropriate feedback
        result["loading_info"] = {
            "was_cached": result.get("semantic_precomputation", {}).get("was_cached", False),
            "computation_time_ms": result.get("semantic_precomputation", {}).get("computation_time_ms", 0),
            "nodes_processed": result.get("semantic_precomputation", {}).get("nodes_processed", 0),
            "cache_key": result.get("semantic_precomputation", {}).get("cache_key", ""),
            "loading_phases": result.get("loading_phases", [])
        }

        logger.info(f"Route started for client {client_id} with {len(result['candidates'])} candidates")
        return result

    except Exception as e:
        logger.error(f"Failed to start route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start route: {str(e)}") from e


@app.post("/api/start-session-async")
async def start_session_async(request_data: StartRouteRequest, request: Request):
    """Initialize a new interactive routing session asynchronously."""
    ensure_file_logging()
    client_id = get_client_id(request, request_data.client_id)
    logger.info(f"Starting async route for client {client_id} at ({request_data.lat:.6f}, {request_data.lon:.6f})")

    try:
        from async_job_manager import start_route_analysis_async
        
        job_id = start_route_analysis_async(
            get_route_builder(),
            client_id=client_id,
            lat=request_data.lat,
            lon=request_data.lon,
            preference=request_data.preference,
            target_distance=request_data.target_distance
        )

        return {
            "job_id": job_id,
            "status": "started",
            "message": "Route analysis started in background",
            "client_id": client_id,
            "poll_url": f"/api/job-status/{job_id}"
        }

    except Exception as e:
        logger.error(f"Failed to start async route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start async route: {str(e)}") from e


@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a background job."""
    try:
        from async_job_manager import job_manager
        
        status = job_manager.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}") from e


@app.get("/api/jobs/stats")
async def get_job_stats():
    """Get job manager statistics."""
    try:
        from async_job_manager import job_manager
        return job_manager.get_stats()
    except Exception as e:
        logger.error(f"Failed to get job stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job stats: {str(e)}") from e


@app.post("/api/add-waypoint")
async def add_waypoint(request: AddWaypointRequest):
    """Add a waypoint to the current route."""
    # Use session_id as client_id for compatibility
    client_id = request.session_id
    logger.info(f"Adding waypoint {request.node_id} to client {client_id}")

    try:
        result = get_route_builder().add_waypoint(client_id, request.node_id)

        logger.info(f"Waypoint added. Route now {result['route_stats']['current_distance']:.0f}m")
        return result

    except ValueError as e:
        logger.warning(f"Waypoint addition failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to add waypoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add waypoint: {str(e)}") from e


@app.post("/api/finalize-route")
async def finalize_route(request: FinalizeRouteRequest):
    """Complete the route by connecting to final destination and back to start."""
    # Use session_id as client_id for compatibility
    client_id = request.session_id
    logger.info(f"Finalizing route for client {client_id} with destination {request.final_node_id}")

    try:
        result = get_route_builder().finalize_route(client_id, request.final_node_id)

        logger.success(f"Route finalized: {result['route_stats']['total_distance']:.0f}m")
        return result

    except ValueError as e:
        logger.warning(f"Route finalization failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to finalize route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to finalize route: {str(e)}") from e


@app.get("/api/route-status/{client_id}")
async def get_route_status(client_id: str):
    """Get current route status and statistics."""
    try:
        result = get_route_builder().get_route_status(client_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get route status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get route status: {str(e)}") from e


@app.get("/api/sessions")
async def list_sessions():
    """List all active client sessions (for debugging)."""
    sessions = []
    current_time = time.time()

    for client_id, session in get_route_builder().client_sessions.items():
        session_info = {
            "client_id": client_id,
            "created_at": session.created_at,
            "last_access": session.last_access,
            "age_minutes": (current_time - session.created_at) / 60,
            "graph_center": session.graph_center,
            "has_active_route": session.active_route is not None
        }

        if session.active_route:
            route = session.active_route
            session_info.update({
                "waypoints_count": len(route.current_waypoints),
                "current_distance": route.total_distance,
                "target_distance": route.target_distance,
                "start_location": route.start_location
            })

        sessions.append(session_info)

    return {
        "active_sessions": sessions,
        "cached_graphs": len(get_route_builder().graph_cache)
    }


@app.delete("/api/session/{client_id}")
async def delete_session(client_id: str):
    """Delete a client session (cleanup)."""
    if client_id in get_route_builder().client_sessions:
        del get_route_builder().client_sessions[client_id]
        return {"success": True, "message": f"Client session {client_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Client session not found")


# Backwards compatibility with old API (for existing frontend)
@app.post("/api/plan-route")
async def plan_route_legacy(request: dict, http_request: Request):
    """Legacy endpoint for backwards compatibility."""
    logger.info("Legacy plan-route endpoint called - redirecting to optimized approach")

    # Convert old request to new format
    start_request = StartRouteRequest(
        lat=request.get("lat"),
        lon=request.get("lon"),
        preference=request.get("preference", "scenic parks and nature"),
        target_distance=request.get("target_distance", 8000)
    )

    # Start route and return first candidates as if it were a planned route
    result = await start_session(start_request, http_request)

    # Convert to old format for compatibility
    if result["candidates"]:
        # Use first candidate as a simple route
        candidate = result["candidates"][0]

        return {
            "coordinates": [
                [result["start_location"]["lat"], result["start_location"]["lon"]],
                [candidate["lat"], candidate["lon"]],
                [result["start_location"]["lat"], result["start_location"]["lon"]]  # Back to start
            ],
            "distance": candidate["estimated_completion"],
            "value_score": candidate["value_score"],
            "message": "Interactive routing started - use new endpoints for full functionality",
            "session_id": result["session_id"],
            "interactive_mode": True,
            "candidates": result["candidates"]
        }
    else:
        raise HTTPException(status_code=500, detail="No candidates found")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main application page."""
    try:
        with open("../frontend/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Perfect10k - Interactive Route Builder</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .method { color: #28a745; font-weight: bold; }
                .path { color: #007bff; font-family: monospace; }
            </style>
        </head>
        <body>
            <h1>Perfect10k Interactive Route Builder</h1>
            <p>User-driven route planning with step-by-step construction</p>

            <h2>New Interactive API Endpoints:</h2>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/start-session</span>
                <p>Initialize routing session with start point and get 3 initial candidates</p>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/add-waypoint</span>
                <p>Add waypoint to route and get 3 new candidates from that point</p>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/finalize-route</span>
                <p>Complete circular route by connecting to final destination and back to start</p>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/route-status/{session_id}</span>
                <p>Get current route state and statistics</p>
            </div>

            <h2>Workflow Example:</h2>
            <ol>
                <li>POST /api/start-session → Get 3 candidates ~1km away</li>
                <li>POST /api/add-waypoint → Add chosen candidate, get 3 new candidates</li>
                <li>Repeat step 2 as needed, or...</li>
                <li>POST /api/finalize-route → Complete circular route back to start</li>
            </ol>

            <p><a href="/docs">Full API Documentation</a></p>
        </body>
        </html>
        """


@app.get("/api/cache-stats")
async def get_cache_statistics():
    """Get comprehensive cache statistics."""
    try:
        stats = get_route_builder().get_cache_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}") from e


@app.get("/api/performance-stats")
async def get_performance_statistics():
    """Get comprehensive performance statistics including fast generation metrics."""
    try:
        route_builder = get_route_builder()
        stats = {
            "cache_stats": route_builder.get_cache_statistics(),
            "session_count": len(route_builder.client_sessions)
        }
        
        # Add fast generator stats if available
        if hasattr(route_builder, 'fast_candidate_generator') and route_builder.fast_candidate_generator:
            stats["fast_generator_stats"] = route_builder.fast_candidate_generator.get_performance_stats()
        
        # Add regular generator stats if available  
        if hasattr(route_builder, 'semantic_candidate_generator') and route_builder.semantic_candidate_generator:
            stats["regular_generator_stats"] = route_builder.semantic_candidate_generator.get_cache_stats()
            
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get performance statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance statistics: {str(e)}") from e


@app.post("/api/toggle-fast-generation")
async def toggle_fast_generation(enable_fast: bool = True):
    """Toggle between fast and regular candidate generation."""
    try:
        # This could be stored in a global setting or per-session
        # For now, we'll just return the current capability
        route_builder = get_route_builder()
        
        fast_available = (hasattr(route_builder, 'fast_candidate_generator') and 
                         route_builder.fast_candidate_generator is not None)
        
        return {
            "success": True,
            "fast_generation_requested": enable_fast,
            "fast_generation_available": fast_available,
            "message": f"Fast generation {'enabled' if enable_fast and fast_available else 'disabled/unavailable'}",
            "fallback_available": True
        }
        
    except Exception as e:
        logger.error(f"Failed to toggle fast generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to toggle fast generation: {str(e)}") from e


@app.post("/api/cleanup-sessions")
async def cleanup_old_sessions(max_age_hours: float = 24.0):
    """Clean up old client sessions."""
    try:
        get_route_builder().cleanup_old_sessions(max_age_hours)
        return {
            "success": True,
            "message": f"Cleaned up sessions older than {max_age_hours} hours",
            "remaining_sessions": len(get_route_builder().client_sessions)
        }
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup sessions: {str(e)}") from e


@app.post("/api/clear-semantic-cache")
async def clear_semantic_candidate_cache():
    """Clear semantic candidate cache to get fresh probabilistic results."""
    try:
        route_builder = get_route_builder()
        
        # Clear cache in semantic candidate generator
        route_builder.semantic_candidate_generator.clear_cache()
        
        # Clear fast generator cache if available
        if hasattr(route_builder, 'fast_candidate_generator') and route_builder.fast_candidate_generator:
            route_builder.fast_candidate_generator.clear_cache()
        
        # Also clear cache keys in active sessions to force recomputation
        cleared_sessions = 0
        for session in route_builder.client_sessions.values():
            session.semantic_cache_key = None
            if hasattr(session, 'fast_cache_key'):
                session.fast_cache_key = None
            cleared_sessions += 1
        
        return {
            "success": True,
            "message": "Cleared all semantic candidate caches - next requests will get fresh probabilistic results",
            "active_sessions_reset": cleared_sessions,
            "caches_cleared": ["semantic_cache", "fast_cache", "session_cache_keys"]
        }
    except Exception as e:
        logger.error(f"Failed to clear semantic cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear semantic cache: {str(e)}") from e


@app.post("/api/semantic-overlays")
async def get_semantic_overlays(request: SemanticOverlayRequest):
    """Get semantic overlay data for specified area and feature types."""
    try:
        # Calculate bounding box from center point and radius
        bbox = get_overlay_manager().calculate_bbox_from_center(
            request.lat, request.lon, request.radius_km
        )
        
        logger.info(f"Getting semantic overlays for {request.feature_types} around ({request.lat:.6f}, {request.lon:.6f}) with radius {request.radius_km}km")
        
        # Get overlays for requested feature types
        overlays = {}
        for feature_type in request.feature_types:
            if feature_type in get_overlay_manager().feature_configs:
                overlays[feature_type] = get_overlay_manager().get_semantic_overlays(
                    feature_type, bbox, request.use_cache
                )
            else:
                logger.warning(f"Unknown feature type requested: {feature_type}")
        
        return {
            "success": True,
            "bbox": {
                "south": bbox.south,
                "west": bbox.west, 
                "north": bbox.north,
                "east": bbox.east
            },
            "overlays": overlays,
            "metadata": {
                "center": {"lat": request.lat, "lon": request.lon},
                "radius_km": request.radius_km,
                "requested_features": request.feature_types,
                "returned_features": list(overlays.keys())
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get semantic overlays: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get semantic overlays: {str(e)}") from e


@app.get("/api/semantic-overlays/{feature_type}")
async def get_single_semantic_overlay(
    feature_type: str,
    lat: float,
    lon: float,
    radius_km: float = 2.0,
    use_cache: bool = True
):
    """Get semantic overlay data for a single feature type."""
    try:
        # Calculate bounding box from center point and radius
        bbox = get_overlay_manager().calculate_bbox_from_center(lat, lon, radius_km)
        
        logger.info(f"Getting {feature_type} overlay around ({lat:.6f}, {lon:.6f}) with radius {radius_km}km")
        
        if feature_type not in get_overlay_manager().feature_configs:
            raise HTTPException(status_code=400, detail=f"Unknown feature type: {feature_type}")
        
        overlay_data = get_overlay_manager().get_semantic_overlays(feature_type, bbox, use_cache)
        
        return {
            "success": True,
            "feature_type": feature_type,
            "bbox": {
                "south": bbox.south,
                "west": bbox.west,
                "north": bbox.north, 
                "east": bbox.east
            },
            "data": overlay_data,
            "metadata": {
                "center": {"lat": lat, "lon": lon},
                "radius_km": radius_km
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get {feature_type} overlay: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get {feature_type} overlay: {str(e)}") from e


@app.get("/api/semantic-overlays-info")
async def get_semantic_overlays_info():
    """Get information about available semantic overlay types."""
    feature_info = {}
    
    for feature_type, config in get_overlay_manager().feature_configs.items():
        feature_info[feature_type] = {
            "style": config["style"],
            "description": f"OSM {feature_type} features"
        }
    
    cache_stats = get_overlay_manager().get_cache_stats()
    
    return {
        "available_features": feature_info,
        "cache_stats": cache_stats,
        "api_info": {
            "endpoints": [
                "POST /api/semantic-overlays - Get multiple overlay types",
                "GET /api/semantic-overlays/{feature_type} - Get single overlay type",
                "GET /api/semantic-overlays-info - Get this info"
            ],
            "supported_features": list(get_overlay_manager().feature_configs.keys())
        }
    }


@app.post("/api/semantic-overlays/clear-cache")
async def clear_semantic_overlay_cache(older_than_hours: int = None):
    """Clear semantic overlay cache."""
    try:
        cleared_count = get_overlay_manager().clear_cache(older_than_hours)
        return {
            "success": True,
            "cleared_files": cleared_count,
            "message": f"Cleared {cleared_count} cache files" + 
                      (f" older than {older_than_hours} hours" if older_than_hours else "")
        }
    except Exception as e:
        logger.error(f"Failed to clear semantic overlay cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}") from e


@app.post("/api/semantic-scoring")
async def score_locations_semantically(request: SemanticScoringRequest):
    """Score multiple locations based on proximity to semantic features."""
    try:
        # Ensure features are loaded for the area
        if request.locations:
            # Use centroid of locations to determine loading area
            center_lat = sum(lat for lat, lon in request.locations) / len(request.locations)
            center_lon = sum(lon for lat, lon in request.locations) / len(request.locations)
            
            get_overlay_manager().ensure_features_loaded_for_area(
                center_lat, center_lon, request.ensure_loaded_radius
            )
        
        # Score all locations
        scores = get_overlay_manager().score_multiple_locations(
            request.locations, request.property_names
        )
        
        return {
            "success": True,
            "scores": scores,
            "metadata": {
                "locations_count": len(request.locations),
                "properties_scored": request.property_names,
                "semantic_properties_info": get_overlay_manager().get_semantic_property_info()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to score locations semantically: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score locations: {str(e)}") from e


@app.get("/api/semantic-scoring/single")
async def score_single_location(
    lat: float,
    lon: float,
    property_names: list[str] = ["forests", "rivers", "lakes"],
    ensure_loaded_radius: float = 2.0
):
    """Score a single location based on proximity to semantic features."""
    try:
        # Ensure features are loaded for the area
        get_overlay_manager().ensure_features_loaded_for_area(lat, lon, ensure_loaded_radius)
        
        # Score the location
        scores = get_overlay_manager().score_location_semantics(lat, lon, property_names)
        
        return {
            "success": True,
            "location": {"lat": lat, "lon": lon},
            "scores": scores,
            "metadata": {
                "properties_scored": property_names,
                "semantic_properties_info": get_overlay_manager().get_semantic_property_info()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to score location semantically: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score location: {str(e)}") from e


@app.get("/api/semantic-properties")
async def get_semantic_properties():
    """Get information about available semantic properties and their configurations."""
    try:
        properties_info = get_overlay_manager().get_semantic_property_info()
        
        return {
            "success": True,
            "properties": properties_info,
            "available_properties": list(properties_info.keys()),
            "total_properties": len(properties_info)
        }
    
    except Exception as e:
        logger.error(f"Failed to get semantic properties info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get properties info: {str(e)}") from e


@app.post("/api/semantic-properties/{property_name}/update")
async def update_semantic_property(property_name: str, updates: dict):
    """Update configuration for a semantic property."""
    try:
        get_overlay_manager().update_semantic_property(property_name, **updates)
        
        return {
            "success": True,
            "message": f"Updated property {property_name}",
            "updated_property": get_overlay_manager().get_semantic_property_info().get(property_name)
        }
    
    except Exception as e:
        logger.error(f"Failed to update semantic property {property_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update property: {str(e)}") from e


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cache_stats = get_route_builder().get_cache_statistics()

    return {
        "status": "healthy",
        "service": "Perfect10k Interactive Route Builder v2.0",
        "approach": "User-driven step-by-step route construction",
        "features": [
            "Interactive route building",
            "Real-time candidate generation",
            "Conflict avoidance",
            "Distance-based heuristics", 
            "Semantic preference matching",
            "Disjunct path planning",
            "Route completion estimation",
            "Persistent graph caching",
            "Semantic overlays (forests, rivers, lakes)"
        ],
        "workflow": [
            "Start session with location and preferences",
            "Get 3 candidates at suitable distance",
            "Add waypoints interactively",
            "Complete circular route",
            "Export final route"
        ],
        "performance": {
            "active_client_sessions": len(get_route_builder().client_sessions),
            "legacy_cache_graphs": len(get_route_builder().graph_cache),
            "spatial_tiles": cache_stats.get("persistent_cache", {}).get("spatial_tile_storage", {}).get("tile_count", 0),
            "total_nodes": cache_stats.get("persistent_cache", {}).get("spatial_tile_storage", {}).get("total_nodes", 0),
            "total_edges": cache_stats.get("persistent_cache", {}).get("spatial_tile_storage", {}).get("total_edges", 0),
            "memory_cached_graphs": cache_stats.get("persistent_cache", {}).get("memory_cache", {}).get("memory_cached_graphs", 0),
            "cache_size_mb": (cache_stats.get("persistent_cache", {}).get("spatial_tile_storage", {}).get("total_size", 0) or 0) / (1024 * 1024),
            "coverage_km2": cache_stats.get("persistent_cache", {}).get("total_coverage_estimate", {}).get("estimated_area_km2", 0),
            "semantic_overlay_cache": get_overlay_manager().get_cache_stats()
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
