from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_FRUITS = ["Tomato", "Orange", "Banana"]

class PredictionRequest(BaseModel):
    fruit_type: str = Field(
        ...,
        example="Tomato",
        description="Fruit commodity to evaluate. Supported: Tomato, Orange, Banana"
    )
    temperature: float = Field(
        ...,
        example=23.5,
        description="Chamber ambient temperature in degrees Celsius (°C). Valid range: -10°C to 60°C"
    )
    humidity: float = Field(
        ...,
        example=90.0,
        description="Relative humidity percentage (%). Valid range: 0% to 100%"
    )
    light_intensity: float = Field(
        ...,
        example=12.5,
        description="Light illuminance intensity in Lux. Must be non-negative (>= 0)"
    )
    co2: float = Field(
        ...,
        example=340.0,
        description="Carbon dioxide concentration in ppm. Must be non-negative (>= 0)"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # fruit aliases
            if "fruit_type" not in data and "fruit" in data:
                data["fruit_type"] = data["fruit"]

            # temperature aliases
            if "temperature" not in data:
                if "temperature_c" in data:
                    data["temperature"] = data["temperature_c"]
                elif "temp" in data:
                    data["temperature"] = data["temp"]

            # humidity aliases
            if "humidity" not in data:
                if "humidity_percent" in data:
                    data["humidity"] = data["humidity_percent"]
                elif "humid" in data:
                    data["humidity"] = data["humid"]

            # light aliases
            if "light_intensity" not in data:
                if "light_lux" in data:
                    data["light_intensity"] = data["light_lux"]
                elif "light" in data:
                    data["light_intensity"] = data["light"]

            # co2 aliases
            if "co2" not in data:
                if "co2_ppm" in data:
                    data["co2"] = data["co2_ppm"]
        return data

    @field_validator("fruit_type")
    @classmethod
    def validate_fruit(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("Fruit type must be a non-empty string.")
        cleaned = v.strip().capitalize()
        if cleaned not in SUPPORTED_FRUITS:
            raise ValueError(
                f"Unsupported fruit '{v}'. Supported commodities are: {', '.join(SUPPORTED_FRUITS)}."
            )
        return cleaned

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < -10.0 or v > 60.0:
            raise ValueError(f"Temperature {v}°C is outside realistic storage range (-10°C to 60°C).")
        return v

    @field_validator("humidity")
    @classmethod
    def validate_humidity(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError(f"Humidity {v}% must be between 0% and 100%.")
        return v

    @field_validator("light_intensity")
    @classmethod
    def validate_light(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Light intensity cannot be negative.")
        if v > 2000.0:
            raise ValueError(f"Light intensity {v} Lux exceeds plausible cold storage chamber conditions.")
        return v

    @field_validator("co2")
    @classmethod
    def validate_co2(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("CO2 concentration cannot be negative.")
        if v > 5000.0:
            raise ValueError(f"CO2 concentration {v} ppm exceeds realistic storage chamber limits.")
        return v

class PredictionResponse(BaseModel):
    predicted_quality: str = Field(..., example="Good", description="Classification result: Good or Bad")
    quality_prediction: Optional[str] = Field(None, example="Good", description="Alias for predicted_quality")
    good_probability: float = Field(..., example=0.998, description="Estimated class probability of Good quality condition")
    bad_probability: float = Field(..., example=0.002, description="Estimated class probability of Bad quality condition")
    confidence_score: Optional[float] = Field(None, example=0.998, description="Winning class confidence probability")
    diagnosis: Optional[str] = Field(None, description="Detailed biological quality diagnosis insight")
    model_name: str = Field(..., example="Random Forest Classifier", description="Machine learning algorithm executing inference")
    prediction_status: str = Field("success", example="success", description="Status indicator of the prediction service")
    input_parameters: Dict[str, Any] = Field(..., description="Echo of received and normalized input telemetry")
    disclaimer: str = Field(
        ...,
        description="Disclaimer on model boundaries (statistical confidence vs real-world physical guarantee)"
    )

class SupportedFruitsResponse(BaseModel):
    supported_fruits: List[str] = Field(..., example=["Tomato", "Orange", "Banana"], description="List of fruit commodities supported by the model")
    total_supported: int = Field(..., example=3, description="Count of supported fruits")
    description: str = Field(..., description="Information regarding supported commodities")

class SampleRecordItem(BaseModel):
    record_id: int = Field(..., example=4, description="1-based original index in cleaned dataset")
    fruit: str = Field(..., example="Tomato")
    fruit_type: Optional[str] = Field("Tomato", example="Tomato", description="Alias for fruit")
    temperature_c: float = Field(..., example=23.0)
    humidity_percent: float = Field(..., example=90.0)
    light_lux: float = Field(..., example=12.62)
    co2_ppm: float = Field(..., example=316.0)
    actual_quality: str = Field(..., example="Good", description="Ground truth quality recorded in dataset")

class SampleRecordsResponse(BaseModel):
    fruit_type: str = Field(..., example="Tomato")
    total_available: int = Field(..., example=3741)
    returned_count: int = Field(..., example=20)
    source: str = Field("backend/data/processed/freshtrack_cleaned.csv")
    disclaimer: str = Field("Dataset Sample – For Demonstration and Testing Only")
    records: List[SampleRecordItem]
