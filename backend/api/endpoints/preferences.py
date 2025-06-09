from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid

from core.database import get_db
from models.schemas import PreferenceCreate, PreferenceResponse, APIResponse
from services.preference_service import PreferenceService
from services.auth import get_current_user
from models.models import User

router = APIRouter()


@router.post("/", response_model=PreferenceResponse)
async def create_preference(
    preference: PreferenceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new user preference.
    """
    try:
        preference_service = PreferenceService(db)
        new_preference = await preference_service.create_preference(
            user_id=current_user.id,
            description=preference.description,
            weight=preference.weight
        )
        return new_preference
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create preference: {str(e)}"
        )


@router.get("/", response_model=List[PreferenceResponse])
async def get_user_preferences(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all user preferences.
    """
    try:
        preference_service = PreferenceService(db)
        preferences = await preference_service.get_user_preferences(current_user.id)
        return preferences
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get preferences: {str(e)}"
        )


@router.put("/{preference_id}", response_model=PreferenceResponse)
async def update_preference(
    preference_id: uuid.UUID,
    preference_update: PreferenceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a user preference.
    """
    try:
        preference_service = PreferenceService(db)
        updated_preference = await preference_service.update_preference(
            preference_id=preference_id,
            user_id=current_user.id,
            description=preference_update.description,
            weight=preference_update.weight
        )
        if not updated_preference:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preference not found"
            )
        return updated_preference
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preference: {str(e)}"
        )


@router.delete("/{preference_id}", response_model=APIResponse)
async def delete_preference(
    preference_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a user preference.
    """
    try:
        preference_service = PreferenceService(db)
        success = await preference_service.delete_preference(
            preference_id=preference_id,
            user_id=current_user.id
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preference not found"
            )
        return APIResponse(success=True, message="Preference deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preference: {str(e)}"
        )


@router.post("/{preference_id}/toggle", response_model=PreferenceResponse)
async def toggle_preference(
    preference_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Toggle a preference active/inactive.
    """
    try:
        preference_service = PreferenceService(db)
        toggled_preference = await preference_service.toggle_preference(
            preference_id=preference_id,
            user_id=current_user.id
        )
        if not toggled_preference:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preference not found"
            )
        return toggled_preference
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle preference: {str(e)}"
        )