from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


# User schemas
class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Route planning schemas
class RouteRequest(BaseModel):
    latitude: float
    longitude: float
    target_distance: Optional[int] = 8000  # Default 8km for 10k steps
    tolerance: Optional[int] = 1000        # Default 1km tolerance
    preference_query: Optional[str] = None  # "I like parks and lakes"
    min_elevation_gain: Optional[int] = None
    avoid_roads: Optional[bool] = True
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v


class RouteResponse(BaseModel):
    id: uuid.UUID
    actual_distance: float
    path_coordinates: List[List[float]]  # [[lat, lon], ...]
    elevation_gain: Optional[float]
    matched_places: Optional[List[Dict[str, Any]]]
    gpx_data: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class RouteHistoryResponse(BaseModel):
    routes: List[RouteResponse]
    total: int
    page: int
    size: int


# Place schemas
class PlaceResponse(BaseModel):
    id: uuid.UUID
    name: str
    place_type: str
    latitude: float
    longitude: float
    description: Optional[str]
    tags: Optional[Dict[str, Any]]
    similarity_score: Optional[float]  # For semantic search results

    class Config:
        from_attributes = True


class PlaceSearchRequest(BaseModel):
    latitude: float
    longitude: float
    radius: Optional[float] = 5000  # 5km radius
    query: Optional[str] = None     # Semantic search query
    place_types: Optional[List[str]] = None  # Filter by place types
    limit: Optional[int] = 50


# User preference schemas
class PreferenceCreate(BaseModel):
    description: str
    weight: Optional[float] = 1.0


class PreferenceResponse(BaseModel):
    id: uuid.UUID
    description: str
    weight: float
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


# API response wrappers
class APIResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None


# Health check schema
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str