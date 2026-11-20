from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.analysis import router as analysis_router
from app.api.predictions import router as predictions_router

api_router = APIRouter(prefix="/api")

# Mount sub-routers
api_router.include_router(health_router)
api_router.include_router(analysis_router)
api_router.include_router(predictions_router)

__all__ = ["api_router"]
