from typing import Dict, Any, List
from pydantic import BaseModel, Field

class DatasetSummaryResponse(BaseModel):
    total_records: int = Field(..., example=8738, description="Total cleaned empirical telemetry records")
    fruit_counts: Dict[str, int] = Field(..., example={"Tomato": 3741, "Orange": 3330, "Banana": 1667}, description="Observation counts broken down by fruit commodity")
    good_quality_count: int = Field(..., example=4553, description="Total Good quality observations")
    bad_quality_count: int = Field(..., example=4185, description="Total Bad quality observations")
    good_quality_percentage: float = Field(..., example=52.11, description="Percentage of Good quality observations")
    bad_quality_percentage: float = Field(..., example=47.89, description="Percentage of Bad quality observations")
    good_ratio: float = Field(52.11, description="Alias for good_quality_percentage")
    bad_ratio: float = Field(47.89, description="Alias for bad_quality_percentage")
    good_count: int = Field(4553, description="Alias for good_quality_count")
    bad_count: int = Field(4185, description="Alias for bad_quality_count")

class FruitQualityItem(BaseModel):
    total: int = Field(..., description="Total observations for this commodity")
    good_count: int = Field(..., description="Good quality observations")
    bad_count: int = Field(..., description="Bad quality observations")
    good_percentage: float = Field(..., description="Good quality percentage for this commodity")
    bad_percentage: float = Field(..., description="Bad quality percentage for this commodity")

class FruitAnalysisResponse(BaseModel):
    fruits: Dict[str, FruitQualityItem] = Field(..., description="Quality distribution cross-tabulated per fruit commodity")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Alias for fruits breakdown")
    summary: str = Field(..., description="High-level analytical summary of commodity distributions")

class EnvironmentalMetric(BaseModel):
    mean: float = Field(..., description="Sample mean")
    median: float = Field(..., description="Sample median (50th percentile)")
    std: float = Field(0.0, description="Standard deviation")
    min: float = Field(..., description="Minimum recorded value")
    max: float = Field(..., description="Maximum recorded value")

class EnvironmentalAnalysisResponse(BaseModel):
    overall_by_quality: Dict[str, Dict[str, EnvironmentalMetric]] = Field(
        ...,
        description="Environmental parameters (temperature_c, humidity_percent, light_lux, co2_ppm) grouped by Good and Bad quality"
    )
    grouped_by_fruit_and_quality: Dict[str, Dict[str, Dict[str, Any]]] = Field(
        ...,
        description="Granular parameter statistics grouped by commodity type and quality state"
    )
    insights: List[str] = Field(
        ...,
        description="Key domain observations derived from empirical sensor distributions"
    )

class ScatterPoint(BaseModel):
    fruit: str
    temperature_c: float
    humidity_percent: float
    light_lux: float
    co2_ppm: float
    quality: str

class ScatterDataResponse(BaseModel):
    total_samples: int
    points: List[ScatterPoint]
    descriptive_statistics: Dict[str, Dict[str, float]]
    correlations: Dict[str, Dict[str, float]]

