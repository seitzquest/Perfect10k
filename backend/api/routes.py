from fastapi import APIRouter
from .endpoints import routes, places, preferences, auth

router = APIRouter()

# Include all endpoint routers
router.include_router(auth.router, prefix="/auth", tags=["authentication"])
router.include_router(routes.router, prefix="/routes", tags=["route planning"])
router.include_router(places.router, prefix="/places", tags=["place search"])
router.include_router(preferences.router, prefix="/preferences", tags=["user preferences"])