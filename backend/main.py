"""
Perfect10k Backend - Interactive Route Builder
New simplified approach for user-driven route construction.
"""

import os
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
        # Use environment variable or default to logs directory relative to project root
        log_dir = os.environ.get("LOG_DIR", str(Path(__file__).parent.parent / "logs"))
        os.makedirs(log_dir, exist_ok=True)
        log_path = Path(log_dir) / "perfect10k.log"

        logger.add(
            str(log_path),
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )
        _file_logging_configured = True


app = FastAPI(
    title="Perfect10k Interactive Route Builder",
    description="User-driven route planning with step-by-step construction",
    version="2.0",
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
        from clean_router import CleanRouter

        route_builder = CleanRouter(get_overlay_manager())
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
    logger.info(
        f"Starting route for client {client_id} at ({request_data.lat:.6f}, {request_data.lon:.6f})"
    )

    try:
        result = get_route_builder().start_route(
            client_id=client_id,
            lat=request_data.lat,
            lon=request_data.lon,
            preference=request_data.preference,
            target_distance=request_data.target_distance,
        )

        # Add loading state information to help frontend show appropriate feedback
        result["loading_info"] = {
            "was_cached": result.get("semantic_precomputation", {}).get("was_cached", False),
            "computation_time_ms": result.get("semantic_precomputation", {}).get(
                "computation_time_ms", 0
            ),
            "nodes_processed": result.get("semantic_precomputation", {}).get("nodes_processed", 0),
            "cache_key": result.get("semantic_precomputation", {}).get("cache_key", ""),
            "loading_phases": result.get("loading_phases", []),
        }

        logger.info(
            f"Route started for client {client_id} with {len(result['candidates'])} candidates"
        )
        return result

    except ValueError as e:
        if "No graph data available" in str(e):
            logger.warning(f"No graph data for location: {str(e)}")
            raise HTTPException(
                status_code=404, 
                detail={
                    "error": "No graph data available for this location",
                    "message": str(e),
                    "suggestion": "This area needs OSM data to be loaded first. Please contact the administrator to add coverage for this geographic region.",
                    "location": {"lat": request_data.lat, "lon": request_data.lon}
                }
            ) from e
        else:
            logger.error(f"Failed to start route: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start route: {str(e)}") from e
    except Exception as e:
        logger.error(f"Failed to start route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start route: {str(e)}") from e


@app.post("/api/start-session-async")
async def start_session_async(request_data: StartRouteRequest, request: Request):
    """Async session start - returns immediate result since clean system is fast."""
    logger.info("Async endpoint called - using fast sync processing")
    
    # Clean system is fast enough - get sync result and wrap in async format
    try:
        sync_result = await start_session(request_data, request)
        
        # Return format that frontend expects - session_id at top level
        return {
            "job_id": sync_result.get("session_id", "immediate"),
            "status": "completed", 
            "result": {
                # Frontend expects session_id at top level of result object
                "session_id": sync_result.get("session_id"),
                "client_id": sync_result.get("session_id"),  # Fallback field
                "candidates": sync_result.get("candidates", []),
                "route_stats": sync_result.get("route_stats", {}),
                "start_location": sync_result.get("start_location", {}),
                "generation_info": sync_result.get("generation_info", {})
            },
            "message": "Route generation completed immediately",
            "completed_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to start async route: {str(e)}")
        return {
            "job_id": "failed",
            "status": "failed",
            "error": str(e),
            "message": "Route generation failed"
        }


@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a job - for immediate completion jobs, return completed status."""
    try:
        # For the clean system, jobs complete immediately
        # Check if this is a valid session ID
        route_builder = get_route_builder()
        
        if job_id == "failed":
            return {
                "job_id": job_id,
                "status": "failed",
                "error": "Route generation failed",
                "message": "Job failed during execution"
            }
        elif job_id == "immediate":
            return {
                "job_id": job_id,
                "status": "completed",
                "message": "Job completed immediately - no polling needed"
            }
        elif job_id in route_builder.client_sessions:
            # Valid session exists - job is completed
            session = route_builder.client_sessions[job_id]
            return {
                "job_id": job_id,
                "status": "completed",
                "message": "Route generation completed",
                "session_id": job_id,
                "completed_at": session.created_at if session else time.time()
            }
        else:
            # Session not found - might be old or invalid
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}") from e


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
            "has_active_route": session.active_route is not None,
        }

        if session.active_route:
            route = session.active_route
            session_info.update(
                {
                    "waypoints_count": len(route.current_waypoints),
                    "current_distance": route.total_distance,
                    "target_distance": route.target_distance,
                    "start_location": route.start_location,
                }
            )

        sessions.append(session_info)

    return {"active_sessions": sessions, "cached_graphs": len(get_route_builder().graph_cache)}


@app.delete("/api/session/{client_id}")
async def delete_session(client_id: str):
    """Delete a client session (cleanup)."""
    if client_id in get_route_builder().client_sessions:
        del get_route_builder().client_sessions[client_id]
        return {"success": True, "message": f"Client session {client_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Client session not found")


# DEPRECATED: Orienteering-Based API Endpoints (removed in clean refactor)

@app.post("/api/explore-alternatives")
async def explore_route_alternatives(request: dict):
    """DEPRECATED: Explore alternative routes using beam search with orienteering optimization."""
    return {
        "error": "This endpoint has been deprecated in the clean refactor",
        "message": "Orienteering-based route exploration is no longer available. Use the new interpretable candidate system.",
        "deprecated": True
    }


@app.get("/api/route-quality/{client_id}")
async def get_route_quality_metrics(client_id: str):
    """DEPRECATED: Get orienteering-based quality metrics for a route."""
    return {
        "error": "This endpoint has been deprecated in the clean refactor",
        "message": "Orienteering-based quality metrics are no longer available. Use route status endpoint for basic metrics.",
        "deprecated": True
    }


@app.get("/api/orienteering-performance")
async def get_orienteering_performance():
    """DEPRECATED: Get performance metrics for the orienteering system."""
    return {
        "error": "This endpoint has been deprecated in the clean refactor",
        "message": "Orienteering performance metrics are no longer available. Use /api/performance-stats for new system metrics.",
        "deprecated": True
    }


@app.post("/api/configure-orienteering")
async def configure_orienteering_system(request: dict):
    """DEPRECATED: Configure orienteering system parameters."""
    return {
        "error": "This endpoint has been deprecated in the clean refactor",
        "message": "Orienteering system configuration is no longer available. The new system auto-configures based on preferences.",
        "deprecated": True
    }


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
        target_distance=request.get("target_distance", 8000),
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
                [result["start_location"]["lat"], result["start_location"]["lon"]],  # Back to start
            ],
            "distance": candidate["estimated_completion"],
            "value_score": candidate["value_score"],
            "message": "Interactive routing started - use new endpoints for full functionality",
            "session_id": result["session_id"],
            "interactive_mode": True,
            "candidates": result["candidates"],
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
        # For CleanRouter, return basic statistics
        route_builder = get_route_builder()
        if hasattr(route_builder, 'get_statistics'):
            return route_builder.get_statistics()
        else:
            return {
                "system": "clean_router",
                "active_sessions": len(route_builder.client_sessions) if hasattr(route_builder, 'client_sessions') else 0,
                "message": "Clean system - minimal caching for optimal performance"
            }
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache statistics: {str(e)}"
        ) from e


# Removed duplicate endpoint - kept the updated version below


@app.get("/api/performance-stats")
async def get_performance_statistics():
    """Get comprehensive performance statistics for the clean candidate generation system."""
    try:
        route_builder = get_route_builder()
        stats = {
            "session_count": len(route_builder.client_sessions),
            "system_type": "clean_candidate_generator",
            "active_generators": len(route_builder.candidate_generators) if hasattr(route_builder, "candidate_generators") else 0
        }

        # Add statistics from our clean candidate generators
        if hasattr(route_builder, "candidate_generators"):
            generator_stats = {}
            for area_key, generator in route_builder.candidate_generators.items():
                generator_stats[area_key] = generator.get_statistics()
            stats["generator_statistics"] = generator_stats

        # Add router statistics
        if hasattr(route_builder, "get_statistics"):
            stats["router_statistics"] = route_builder.get_statistics()

        return stats

    except Exception as e:
        logger.error(f"Failed to get performance statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance statistics: {str(e)}"
        ) from e


@app.post("/api/toggle-fast-generation")
async def toggle_fast_generation(enable_fast: bool = True):
    """DEPRECATED: Toggle between fast and regular candidate generation."""
    return {
        "success": True,
        "message": "Fast generation toggle is deprecated. The clean system automatically optimizes performance.",
        "deprecated": True,
        "fast_generation_enabled": True  # Clean system is always fast
    }


@app.post("/api/cleanup-sessions")
async def cleanup_old_sessions(max_age_hours: float = 24.0):
    """Clean up old client sessions."""
    try:
        get_route_builder().cleanup_old_sessions(max_age_hours)
        return {
            "success": True,
            "message": f"Cleaned up sessions older than {max_age_hours} hours",
            "remaining_sessions": len(get_route_builder().client_sessions),
        }
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup sessions: {str(e)}") from e


@app.post("/api/clear-semantic-cache")
async def clear_semantic_candidate_cache():
    """Clear candidate generation cache to get fresh results."""
    try:
        route_builder = get_route_builder()

        # Clear cache in our new clean candidate generators
        generators_cleared = 0
        if hasattr(route_builder, "candidate_generators"):
            for generator in route_builder.candidate_generators.values():
                generator.clear_cache()
                generators_cleared += 1

        # Reset active sessions
        cleared_sessions = len(route_builder.client_sessions)
        route_builder.client_sessions.clear()

        return {
            "success": True,
            "message": "Cleared all candidate generation caches - next requests will get fresh results",
            "generators_cleared": generators_cleared,
            "active_sessions_reset": cleared_sessions,
            "system": "clean_candidate_generator"
        }
    except Exception as e:
        logger.error(f"Failed to clear candidate cache: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear candidate cache: {str(e)}"
        ) from e


@app.post("/api/semantic-overlays")
async def get_semantic_overlays(request: SemanticOverlayRequest):
    """Get semantic overlay data for specified area and feature types."""
    try:
        # Calculate bounding box from center point and radius
        bbox = get_overlay_manager().calculate_bbox_from_center(
            request.lat, request.lon, request.radius_km
        )

        logger.info(
            f"Getting semantic overlays for {request.feature_types} around ({request.lat:.6f}, {request.lon:.6f}) with radius {request.radius_km}km"
        )

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
                "east": bbox.east,
            },
            "overlays": overlays,
            "metadata": {
                "center": {"lat": request.lat, "lon": request.lon},
                "radius_km": request.radius_km,
                "requested_features": request.feature_types,
                "returned_features": list(overlays.keys()),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get semantic overlays: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get semantic overlays: {str(e)}"
        ) from e


@app.get("/api/semantic-overlays/{feature_type}")
async def get_single_semantic_overlay(
    feature_type: str, lat: float, lon: float, radius_km: float = 2.0, use_cache: bool = True
):
    """Get semantic overlay data for a single feature type."""
    try:
        # Calculate bounding box from center point and radius
        bbox = get_overlay_manager().calculate_bbox_from_center(lat, lon, radius_km)

        logger.info(
            f"Getting {feature_type} overlay around ({lat:.6f}, {lon:.6f}) with radius {radius_km}km"
        )

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
                "east": bbox.east,
            },
            "data": overlay_data,
            "metadata": {"center": {"lat": lat, "lon": lon}, "radius_km": radius_km},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get {feature_type} overlay: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get {feature_type} overlay: {str(e)}"
        ) from e


@app.get("/api/semantic-overlays-info")
async def get_semantic_overlays_info():
    """Get information about available semantic overlay types."""
    feature_info = {}

    for feature_type, config in get_overlay_manager().feature_configs.items():
        feature_info[feature_type] = {
            "style": config["style"],
            "description": f"OSM {feature_type} features",
        }

    cache_stats = get_overlay_manager().get_cache_stats()

    return {
        "available_features": feature_info,
        "cache_stats": cache_stats,
        "api_info": {
            "endpoints": [
                "POST /api/semantic-overlays - Get multiple overlay types",
                "GET /api/semantic-overlays/{feature_type} - Get single overlay type",
                "GET /api/semantic-overlays-info - Get this info",
            ],
            "supported_features": list(get_overlay_manager().feature_configs.keys()),
        },
    }


@app.post("/api/semantic-overlays/clear-cache")
async def clear_semantic_overlay_cache(older_than_hours: int = None):
    """Clear semantic overlay cache."""
    try:
        cleared_count = get_overlay_manager().clear_cache(older_than_hours)
        return {
            "success": True,
            "cleared_files": cleared_count,
            "message": f"Cleared {cleared_count} cache files"
            + (f" older than {older_than_hours} hours" if older_than_hours else ""),
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
                "semantic_properties_info": get_overlay_manager().get_semantic_property_info(),
            },
        }

    except Exception as e:
        logger.error(f"Failed to score locations semantically: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score locations: {str(e)}") from e


@app.get("/api/semantic-scoring/single")
async def score_single_location(
    lat: float,
    lon: float,
    property_names: list[str] = ["forests", "rivers", "lakes"],
    ensure_loaded_radius: float = 2.0,
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
                "semantic_properties_info": get_overlay_manager().get_semantic_property_info(),
            },
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
            "total_properties": len(properties_info),
        }

    except Exception as e:
        logger.error(f"Failed to get semantic properties info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get properties info: {str(e)}"
        ) from e


@app.post("/api/semantic-properties/{property_name}/update")
async def update_semantic_property(property_name: str, updates: dict):
    """Update configuration for a semantic property."""
    try:
        get_overlay_manager().update_semantic_property(property_name, **updates)

        return {
            "success": True,
            "message": f"Updated property {property_name}",
            "updated_property": get_overlay_manager()
            .get_semantic_property_info()
            .get(property_name),
        }

    except Exception as e:
        logger.error(f"Failed to update semantic property {property_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update property: {str(e)}") from e


@app.get("/api/spatial-grid/{client_id}")
async def get_spatial_grid_data(client_id: str):
    """Get spatial grid scoring data for visualization overlays."""
    try:
        route_builder = get_route_builder()
        
        if client_id not in route_builder.client_sessions:
            raise HTTPException(status_code=404, detail="Client session not found")
        
        session = route_builder.client_sessions[client_id]
        lat, lon = session.graph_center
        area_key = f"{lat:.3f}_{lon:.3f}"
        
        if area_key not in route_builder.candidate_generators:
            raise HTTPException(status_code=404, detail="No candidate generator available for this area")
        
        generator = route_builder.candidate_generators[area_key]
        
        # Get grid cells and their feature scores
        grid_data = []
        for grid_coords, cell in generator.spatial_grid.grid.items():
            cell_features = generator.feature_database.get_cell_features(cell.center_lat, cell.center_lon)
            if cell_features:
                grid_data.append({
                    "lat": cell.center_lat,
                    "lon": cell.center_lon,
                    "bounds": {
                        "south": cell.center_lat - generator.spatial_grid.cell_size_degrees / 2,
                        "west": cell.center_lon - generator.spatial_grid.cell_size_degrees / 2,
                        "north": cell.center_lat + generator.spatial_grid.cell_size_degrees / 2,
                        "east": cell.center_lon + generator.spatial_grid.cell_size_degrees / 2
                    },
                    "feature_scores": {
                        feature_type.value: cell_features.get_feature(feature_type)
                        for feature_type in cell_features.features.keys()
                    },
                    "node_count": len(cell.nodes)
                })
        
        return {
            "success": True,
            "grid_data": grid_data,
            "metadata": {
                "total_cells": len(grid_data),
                "cell_size_meters": generator.spatial_grid.cell_size_meters,
                "area_center": {"lat": lat, "lon": lon}
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get spatial grid data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get spatial grid data: {str(e)}")


@app.get("/api/coverage")
async def get_coverage_info():
    """Get information about geographic areas with available data."""
    try:
        # Check spatial tile storage for available areas
        spatial_storage = get_route_builder().spatial_storage
        
        # Query database for available tiles
        cursor = spatial_storage.conn.cursor()
        cursor.execute("""
            SELECT geohash, min_lat, min_lon, max_lat, max_lon, node_count, created_at
            FROM tiles 
            ORDER BY created_at DESC 
            LIMIT 20
        """)
        tiles = cursor.fetchall()
        
        coverage_areas = []
        for tile in tiles:
            geohash, min_lat, min_lon, max_lat, max_lon, node_count, created_at = tile
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            
            coverage_areas.append({
                "center": {"lat": center_lat, "lon": center_lon},
                "bbox": {"min_lat": min_lat, "min_lon": min_lon, "max_lat": max_lat, "max_lon": max_lon},
                "node_count": node_count,
                "created_at": created_at
            })
        
        return {
            "available_areas": coverage_areas,
            "total_areas": len(coverage_areas),
            "message": "These are the geographic areas with available routing data" if coverage_areas else "No areas with routing data available. OSM data needs to be loaded first."
        }
        
    except Exception as e:
        logger.error(f"Failed to get coverage info: {e}")
        return {
            "available_areas": [],
            "total_areas": 0,
            "message": "Unable to determine coverage areas",
            "error": str(e)
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cache_stats = get_route_builder().get_statistics()

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
            "Semantic overlays (forests, rivers, lakes)",
        ],
        "workflow": [
            "Start session with location and preferences",
            "Get 3 candidates at suitable distance",
            "Add waypoints interactively",
            "Complete circular route",
            "Export final route",
        ],
        "performance": {
            "active_client_sessions": cache_stats.get("active_sessions", 0),
            "candidate_generators": cache_stats.get("candidate_generators", 0),
            "generator_stats": cache_stats.get("generator_stats", {}),
            "semantic_overlay_cache": get_overlay_manager().get_cache_stats(),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
