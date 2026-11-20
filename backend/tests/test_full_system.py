"""
FreshTrack End-to-End System Integration Test Suite
Validates Phases 1 through 7:
1. Data loading, cleaning and immutability.
2. EDA summaries and Plotly chart artifacts.
3. Machine learning model loading and predictions.
4. Input validation and guardrail checks.
5. FastAPI route handlers and responses.
6. Frontend static asset integrity.
"""

import os
import sys
import hashlib
from pathlib import Path

# Configure paths
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
import joblib

from app.core.config import settings
from app.services.data_cleaning import validate_cleaned_dataset
from app.services.eda import load_data, get_overall_summary, get_fruit_wise_summary, get_environmental_summary
from app.services.model_prediction import predictor_service, predict_quality
from app.api.health import get_health
from app.api.analysis import get_dataset_summary, get_fruits_analysis, get_environmental_analysis
from app.api.predictions import predict_quality as api_predict, get_supported_fruits
from app.schemas.prediction import PredictionRequest

def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()

def run_e2e_tests():
    print("=================================================================")
    print("FRESH TRACK PHASE 7: COMPLETE SYSTEM INTEGRATION AUDIT")
    print("=================================================================\n")

    passed = 0
    failed = 0

    def assert_check(condition, test_name, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"[PASS] {test_name}")
            if detail:
                print(f"       -> {detail}")
            passed += 1
        else:
            print(f"[FAIL] {test_name}")
            if detail:
                print(f"       -> {detail}")
            failed += 1

    # -------------------------------------------------------------
    # 1. Dataset Integrity & Immutability
    # -------------------------------------------------------------
    orig_path = settings.ORIGINAL_DATA_PATH
    orig_hash = get_file_sha256(orig_path)
    expected_orig_hash = "218F76F061A12B0A38C55107E5BE3B377E9EECB33F1A1840EF2B4E4618A5E5AA"
    assert_check(
        orig_hash == expected_orig_hash,
        "1. Original Dataset Immutability",
        f"SHA256: {orig_hash[:16]}... Match verified"
    )

    cleaned_val = validate_cleaned_dataset(settings.PROCESSED_DATA_PATH)
    assert_check(
        cleaned_val["is_valid"] and cleaned_val["rows"] == 8738 and cleaned_val["total_missing_values"] == 0,
        "2. Cleaned Dataset Validation",
        f"Rows: {cleaned_val['rows']}, Missing: {cleaned_val['total_missing_values']}, Fruits: {list(cleaned_val['fruit_counts'].keys())}"
    )

    # -------------------------------------------------------------
    # 2. EDA Computations & Plotly Artifacts
    # -------------------------------------------------------------
    df = load_data(settings.PROCESSED_DATA_PATH)
    eda_sum = get_overall_summary(df)
    assert_check(
        eda_sum["total_records"] == 8738 and eda_sum["good_percentage"] == 52.11 and eda_sum["bad_percentage"] == 47.89,
        "3. Phase 3 EDA Metric Computations",
        f"Good: {eda_sum['good_percentage']}%, Bad: {eda_sum['bad_percentage']}%"
    )

    expected_eda_charts = [
        "good_vs_bad_donut.html",
        "fruit_wise_quality_bar.html",
        "environmental_conditions_comparison.html",
        "correlation_heatmap.html",
        "fruit_comparison_chart.html"
    ]
    charts_dir = BACKEND_DIR / "reports" / "charts"
    eda_charts_exist = all((charts_dir / c).exists() and (charts_dir / c).stat().st_size > 5000 for c in expected_eda_charts)
    assert_check(
        eda_charts_exist,
        "4. Phase 3 Interactive Plotly Charts Verification",
        f"All 5 EDA charts verified in {charts_dir.relative_to(PROJECT_ROOT)}"
    )

    # -------------------------------------------------------------
    # 3. Model Loading & Machine Learning Inference
    # -------------------------------------------------------------
    model_path = settings.MODEL_PATH
    assert_check(
        model_path.exists() and model_path.stat().st_size > 50000,
        "5. Serialized Random Forest Model File Verification",
        f"Path: {model_path.relative_to(PROJECT_ROOT)}, Size: {model_path.stat().st_size / 1024:.1f} KB"
    )

    # Fresh tomato test
    pred_fresh = predict_quality(fruit="Tomato", temperature_c=23.0, humidity_percent=90.0, light_lux=9.8, co2_ppm=350.0)
    assert_check(
        pred_fresh["success"] and pred_fresh["predicted_quality"] == "Good" and pred_fresh["confidence"] >= 0.95,
        "6. ML Prediction: Optimal Fresh Tomato",
        f"Quality: {pred_fresh['predicted_quality']} (Confidence: {pred_fresh['confidence']:.2%})"
    )

    # Degraded high-light tomato test
    pred_bad = predict_quality(fruit="Tomato", temperature_c=25.0, humidity_percent=95.0, light_lux=150.0, co2_ppm=320.0)
    assert_check(
        pred_bad["success"] and pred_bad["predicted_quality"] == "Bad" and pred_bad["confidence"] >= 0.95,
        "7. ML Prediction: High-Light Degraded Tomato",
        f"Quality: {pred_bad['predicted_quality']} (Confidence: {pred_bad['confidence']:.2%})"
    )

    expected_ml_charts = [
        "model_comparison.html",
        "confusion_matrices.html",
        "feature_importance.html",
        "classification_metrics.html"
    ]
    ml_charts_dir = BACKEND_DIR / "reports" / "ml_charts"
    ml_charts_exist = all((ml_charts_dir / c).exists() and (ml_charts_dir / c).stat().st_size > 5000 for c in expected_ml_charts)
    assert_check(
        ml_charts_exist,
        "8. Phase 4 ML Diagnostic Visualizations Verification",
        f"All 4 ML diagnostic charts verified in {ml_charts_dir.relative_to(PROJECT_ROOT)}"
    )

    # -------------------------------------------------------------
    # 4. Input Validation & Error Guardrails
    # -------------------------------------------------------------
    inv_fruit = predictor_service.predict("Grape", 25.0, 90.0, 10.0, 300.0)
    assert_check(
        inv_fruit["success"] is False and "Unsupported fruit" in inv_fruit["validation_errors"][0],
        "9. Input Guardrail: Unsupported Fruit Rejection",
        f"Error captured: {inv_fruit['validation_errors'][0]}"
    )

    inv_humid = predictor_service.predict("Orange", 22.0, -10.0, 10.0, 300.0)
    assert_check(
        inv_humid["success"] is False and "Humidity" in inv_humid["validation_errors"][0],
        "10. Input Guardrail: Negative Humidity Rejection",
        f"Error captured: {inv_humid['validation_errors'][0]}"
    )

    # -------------------------------------------------------------
    # 5. REST API Handlers & Contract Verification
    # -------------------------------------------------------------
    h_res = get_health()
    assert_check(
        h_res.status == "healthy" and h_res.model_availability is True,
        "11. REST API: GET /api/health Handler",
        f"Status: {h_res.status}, Active Model: {h_res.model_name}"
    )

    sum_res = get_dataset_summary()
    assert_check(
        sum_res.total_records == 8738 and sum_res.good_quality_count == 4553,
        "12. REST API: GET /api/analysis/summary Handler",
        f"Total: {sum_res.total_records}, Good: {sum_res.good_quality_count}, Bad: {sum_res.bad_quality_count}"
    )

    fruits_res = get_fruits_analysis()
    assert_check(
        len(fruits_res.fruits) == 3 and "Banana" in fruits_res.fruits,
        "13. REST API: GET /api/analysis/fruits Handler",
        f"Fruits: {list(fruits_res.fruits.keys())}"
    )

    env_res = get_environmental_analysis()
    assert_check(
        "Good" in env_res.overall_by_quality and "Bad" in env_res.overall_by_quality,
        "14. REST API: GET /api/analysis/environment Handler",
        f"Overall Keys: {list(env_res.overall_by_quality.keys())}"
    )

    supp_res = get_supported_fruits()
    assert_check(
        supp_res.total_supported == 3 and supp_res.supported_fruits == ["Tomato", "Orange", "Banana"],
        "15. REST API: GET /api/predictions/fruits Handler",
        f"Supported Commodities: {supp_res.supported_fruits}"
    )

    api_p_req = PredictionRequest(
        fruit_type="Orange",
        temperature=22.0,
        humidity=94.0,
        light_intensity=7.0,
        co2=290.0
    )
    api_p_res = api_predict(api_p_req)
    assert_check(
        api_p_res.prediction_status == "success" and api_p_res.predicted_quality == "Good",
        "16. REST API: POST /api/predictions/predict Handler",
        f"Quality: {api_p_res.predicted_quality}, Good Prob: {api_p_res.good_probability:.2%}"
    )

    # -------------------------------------------------------------
    # 6. Frontend Artifacts & Architecture Separation
    # -------------------------------------------------------------
    frontend_dir = PROJECT_ROOT / "frontend"
    index_html = frontend_dir / "index.html"
    api_js = frontend_dir / "js" / "api.js"
    predict_html = frontend_dir / "pages" / "predict.html"
    analytics_html = frontend_dir / "pages" / "analytics.html"

    fe_files_exist = index_html.exists() and api_js.exists() and predict_html.exists() and analytics_html.exists()
    assert_check(
        fe_files_exist,
        "17. Frontend Interface & Client Files Verification",
        "index.html, api.js, pages/predict.html, pages/analytics.html verified"
    )

    # Verify zero Python code in frontend
    fe_python_files = list(frontend_dir.rglob("*.py"))
    assert_check(
        len(fe_python_files) == 0,
        "18. Architecture Boundary: Zero Python Code in Frontend",
        f"Found {len(fe_python_files)} python files in frontend"
    )

    # Verify zero JS/HTML in backend
    be_fe_files = list((BACKEND_DIR / "app").rglob("*.html")) + list((BACKEND_DIR / "app").rglob("*.js"))
    assert_check(
        len(be_fe_files) == 0,
        "19. Architecture Boundary: Zero HTML/JS Code in Backend app/",
        f"Found {len(be_fe_files)} HTML/JS files in backend app/"
    )

    print("\n-----------------------------------------------------------------")
    print(f"AUDIT SUMMARY: {passed} PASSED, {failed} FAILED (TOTAL: {passed + failed})")
    print("=================================================================")

    return failed == 0

if __name__ == "__main__":
    success = run_e2e_tests()
    if not success:
        sys.exit(1)
