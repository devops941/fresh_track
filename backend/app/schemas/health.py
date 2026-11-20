from typing import Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy", description="Operational health status of the API")
    app_name: str = Field(..., example="FreshTrack API", description="Application service name")
    version: str = Field(..., example="1.0.0", description="API version")
    model_availability: bool = Field(..., example=True, description="Whether the trained ML model artifact is loaded and ready")
    model_name: Optional[str] = Field(None, example="Random Forest Classifier", description="Active champion model identifier")
    dataset_available: bool = Field(..., example=True, description="Whether the cleaned empirical dataset is present")
