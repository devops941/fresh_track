from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.core.config import settings
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    SupportedFruitsResponse,
    SampleRecordItem,
    SampleRecordsResponse,
    SUPPORTED_FRUITS
)
from app.services.eda import load_data
from app.services.model_prediction import predictor_service

router = APIRouter(prefix="/predictions", tags=["Quality Predictions"])

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Fruit Quality State",
    description=(
        "Executes machine learning inference to classify fruit quality as Good or Bad "
        "based on user-provided environmental telemetry (Temperature, Humidity, Light, and CO2). "
        "Uses the pre-trained, serialized Random Forest model."
    )
)
def predict_quality(request: PredictionRequest) -> PredictionResponse:
    # Check model service readiness
    if predictor_service.model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained machine learning model is not loaded on server."
        )

    # Execute prediction via the service
    result = predictor_service.predict(
        fruit=request.fruit_type,
        temperature_c=request.temperature,
        humidity_percent=request.humidity,
        light_lux=request.light_intensity,
        co2_ppm=request.co2
    )

    if not result.get("success"):
        errors = result.get("validation_errors", ["Prediction failed."])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors)
        )

    probs = result.get("class_probabilities", {})
    good_prob = float(probs.get("Good", 0.0))
    bad_prob = float(probs.get("Bad", 0.0))
    pred_qual = result["predicted_quality"]
    conf = good_prob if pred_qual == "Good" else bad_prob

    # Generate college-grade biological diagnosis insight
    f_lower = request.fruit_type.lower()
    if pred_qual == "Good":
        diagnosis_text = f"Storage conditions are ideal for preserving cellular firmness and quality in {f_lower}s."
    else:
        reasons = []
        if request.temperature > 28.0 or request.temperature < 8.0:
            reasons.append(f"Temperature ({request.temperature}°C) outside safe threshold")
        if request.humidity < 70.0 or request.humidity > 95.0:
            reasons.append(f"Humidity ({request.humidity}%) triggers desiccation/mold")
        if request.light_intensity > 300.0:
            reasons.append(f"Excess lighting ({request.light_intensity} Lux) accelerates decay")
        if request.co2 > 400.0:
            reasons.append(f"CO₂ concentration ({request.co2} ppm) accelerates senescence")
        diag_reason = "; ".join(reasons) if reasons else "Environmental telemetry indicates accelerated spoilage risk."
        diagnosis_text = f"High Spoilage Risk: {diag_reason}"

    return PredictionResponse(
        predicted_quality=pred_qual,
        quality_prediction=pred_qual,
        good_probability=good_prob,
        bad_probability=bad_prob,
        confidence_score=conf,
        diagnosis=diagnosis_text,
        model_name=result.get("model_name", "Random Forest Classifier"),
        prediction_status="success",
        input_parameters={
            "fruit_type": request.fruit_type,
            "temperature_c": request.temperature,
            "humidity_percent": request.humidity,
            "light_lux": request.light_intensity,
            "co2_ppm": request.co2
        },
        disclaimer=result.get("disclaimer", "Statistical prediction based on chamber telemetry.")
    )

@router.get(
    "/fruits",
    response_model=SupportedFruitsResponse,
    summary="Supported Fruit Commodities",
    description="Returns the list of fruit types supported by the pre-trained machine learning model."
)
def get_supported_fruits() -> SupportedFruitsResponse:
    return SupportedFruitsResponse(
        supported_fruits=SUPPORTED_FRUITS,
        total_supported=len(SUPPORTED_FRUITS),
        description="Fruit commodities currently supported for quality classification by the trained model."
    )

@router.get(
    "/sample-records",
    response_model=SampleRecordsResponse,
    summary="Get Actual Dataset Sample Records",
    description=(
        "Retrieves genuine environmental records from the cleaned dataset (backend/data/processed/freshtrack_cleaned.csv) "
        "for a given fruit type. Used to allow users to explore and test the ML prediction engine "
        "with real historical data without needing physical sensing equipment."
    )
)
def get_sample_records(
    fruit_type: str = Query("Tomato", description="Fruit commodity: Tomato, Orange, or Banana"),
    quality: Optional[str] = Query(default=None, description="Optional quality filter: Good, Bad, or All"),
    limit: int = Query(default=20, ge=1, le=100, description="Max number of records to return"),
    offset: int = Query(default=0, ge=0, description="Offset index for pagination")
) -> SampleRecordsResponse:
    clean_fruit = fruit_type if isinstance(fruit_type, str) else "Tomato"
    cleaned_fruit = clean_fruit.strip().capitalize()
    if cleaned_fruit not in SUPPORTED_FRUITS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported fruit '{fruit_type}'. Supported: {', '.join(SUPPORTED_FRUITS)}"
        )
    
    num_limit = limit if isinstance(limit, int) else 20
    num_offset = offset if isinstance(offset, int) else 0

    try:
        df = load_data(settings.PROCESSED_DATA_PATH)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cleaned dataset freshtrack_cleaned.csv not found on server."
        )

    # Filter by fruit
    fruit_df = df[df["fruit"].str.lower() == cleaned_fruit.lower()]
    
    # Optional quality filter
    if quality is not None and isinstance(quality, str) and quality.strip().lower() in ("good", "bad"):
        fruit_df = fruit_df[fruit_df["quality"].str.lower() == quality.strip().lower()]
    
    total_avail = len(fruit_df)
    sliced_df = fruit_df.iloc[num_offset : num_offset + num_limit]

    records = []
    for idx, row in sliced_df.iterrows():
        records.append(
            SampleRecordItem(
                record_id=int(idx) + 1,
                fruit=str(row["fruit"]),
                fruit_type=str(row["fruit"]),
                temperature_c=float(row["temperature_c"]),
                humidity_percent=float(row["humidity_percent"]),
                light_lux=round(float(row["light_lux"]), 2),
                co2_ppm=round(float(row["co2_ppm"]), 1),
                actual_quality=str(row["quality"])
            )
        )

    return SampleRecordsResponse(
        fruit_type=cleaned_fruit,
        total_available=total_avail,
        returned_count=len(records),
        source="backend/data/processed/freshtrack_cleaned.csv",
        disclaimer="Dataset Sample – For Demonstration and Testing Only",
        records=records
    )
