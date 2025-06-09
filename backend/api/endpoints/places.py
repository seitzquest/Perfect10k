from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from core.database import get_db
from models.schemas import PlaceSearchRequest, PlaceResponse, APIResponse
from services.place_service import PlaceService
from services.auth import get_current_user
from models.models import User

router = APIRouter()


@router.post("/search", response_model=List[PlaceResponse])
async def search_places(
    search_request: PlaceSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Search for places using semantic matching.
    """
    try:
        place_service = PlaceService(db)
        places = await place_service.search_places(
            latitude=search_request.latitude,
            longitude=search_request.longitude,
            radius=search_request.radius,
            query=search_request.query,
            place_types=search_request.place_types,
            limit=search_request.limit
        )
        return places
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Place search failed: {str(e)}"
        )


@router.get("/nearby", response_model=List[PlaceResponse])
async def get_nearby_places(
    latitude: float,
    longitude: float,
    radius: float = 5000,
    place_types: str = None,  # Comma-separated list
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get nearby places within a radius.
    """
    try:
        place_service = PlaceService(db)
        
        # Parse place_types if provided
        types_list = None
        if place_types:
            types_list = [t.strip() for t in place_types.split(',')]
        
        places = await place_service.get_nearby_places(
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            place_types=types_list,
            limit=limit
        )
        return places
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get nearby places: {str(e)}"
        )


@router.post("/refresh", response_model=APIResponse)
async def refresh_places_cache(
    latitude: float,
    longitude: float,
    radius: float = 10000,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Refresh the places cache for a given area.
    """
    try:
        place_service = PlaceService(db)
        count = await place_service.refresh_places_cache(
            latitude=latitude,
            longitude=longitude,
            radius=radius
        )
        return APIResponse(
            success=True,
            message=f"Refreshed {count} places in cache",
            data={"places_updated": count}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh places cache: {str(e)}"
        )


@router.get("/types", response_model=List[str])
async def get_place_types(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all available place types.
    """
    try:
        place_service = PlaceService(db)
        types = await place_service.get_available_place_types()
        return types
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get place types: {str(e)}"
        )