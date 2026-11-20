import os
from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse
from app.services.model_prediction import predictor_service

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Model Readiness",
    description="Returns the operational status of the FreshTrack API, environment, and ML model availability."
)
def get_health() -> HealthResponse:
    dataset_exists = os.path.exists(settings.PROCESSED_DATA_PATH)
    model_ready = predictor_service.model_pipeline is not None
    
    return HealthResponse(
        status="healthy" if (dataset_exists and model_ready) else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_availability=model_ready,
        model_name=predictor_service.model_name if model_ready else None,
        dataset_available=dataset_exists
    )
