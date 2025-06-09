from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid

from core.database import get_db
from models.schemas import RouteRequest, RouteResponse, RouteHistoryResponse, APIResponse
from services.route_planner import RoutePlannerService
from services.auth import get_current_user
from models.models import User

router = APIRouter()


@router.post("/plan", response_model=RouteResponse)
async def plan_route(
    route_request: RouteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Plan a new route based on user preferences and location.
    """
    try:
        route_planner = RoutePlannerService(db)
        route = await route_planner.plan_route(
            user_id=current_user.id,
            latitude=route_request.latitude,
            longitude=route_request.longitude,
            target_distance=route_request.target_distance,
            tolerance=route_request.tolerance,
            preference_query=route_request.preference_query,
            min_elevation_gain=route_request.min_elevation_gain,
            avoid_roads=route_request.avoid_roads
        )
        return route
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route planning failed: {str(e)}"
        )


@router.get("/history", response_model=RouteHistoryResponse)
async def get_route_history(
    page: int = 1,
    size: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's route planning history.
    """
    try:
        route_planner = RoutePlannerService(db)
        history = await route_planner.get_user_routes(
            user_id=current_user.id,
            page=page,
            size=size
        )
        return history
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve route history: {str(e)}"
        )


@router.get("/{route_id}", response_model=RouteResponse)
async def get_route(
    route_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific route by ID.
    """
    try:
        route_planner = RoutePlannerService(db)
        route = await route_planner.get_route_by_id(route_id, current_user.id)
        if not route:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Route not found"
            )
        return route
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve route: {str(e)}"
        )


@router.delete("/{route_id}", response_model=APIResponse)
async def delete_route(
    route_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a route.
    """
    try:
        route_planner = RoutePlannerService(db)
        success = await route_planner.delete_route(route_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Route not found"
            )
        return APIResponse(success=True, message="Route deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete route: {str(e)}"
        )


@router.post("/{route_id}/export/gpx")
async def export_route_gpx(
    route_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Export route as GPX file.
    """
    try:
        route_planner = RoutePlannerService(db)
        gpx_content = await route_planner.export_route_gpx(route_id, current_user.id)
        if not gpx_content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Route not found"
            )
        
        from fastapi.responses import Response
        return Response(
            content=gpx_content,
            media_type="application/gpx+xml",
            headers={"Content-Disposition": f"attachment; filename=route_{route_id}.gpx"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export route: {str(e)}"
        )