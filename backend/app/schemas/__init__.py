"""Pydantic Request and Response Schemas for FreshTrack API"""
from app.schemas.health import HealthResponse
from app.schemas.analysis import (
    DatasetSummaryResponse,
    FruitAnalysisResponse,
    EnvironmentalAnalysisResponse,
    FruitQualityItem,
    EnvironmentalMetric
)
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    SupportedFruitsResponse,
    SUPPORTED_FRUITS
)

__all__ = [
    "HealthResponse",
    "DatasetSummaryResponse",
    "FruitAnalysisResponse",
    "EnvironmentalAnalysisResponse",
    "FruitQualityItem",
    "EnvironmentalMetric",
    "PredictionRequest",
    "PredictionResponse",
    "SupportedFruitsResponse",
    "SUPPORTED_FRUITS"
]
