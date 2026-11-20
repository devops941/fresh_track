"""
FreshTrack Reusable Inference & Quality Prediction Service
Phase 4: Machine Learning Real-Time Quality Classification

Service Responsibilities:
1. Loads the saved Joblib pipeline from backend/models/freshtrack_model.joblib.
2. Performs strict validation on incoming environmental telemetry inputs:
   - Fruit commodity (Tomato, Orange, Banana)
   - Temperature (°C)
   - Humidity (% RH)
   - Light intensity (Lux)
   - CO2 concentration (ppm)
3. Generates quality predictions (Good / Bad) with class probabilities and model confidence.
4. Returns structured results including error diagnostics and empirical disclaimers.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import joblib

# Paths
SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent.parent
PRIMARY_MODEL_PATH = BACKEND_DIR / "models" / "freshtrack_model.joblib"
BACKUP_MODEL_PATH = BACKEND_DIR / "ml" / "saved_model" / "freshtrack_model.joblib"
METADATA_PATH = BACKEND_DIR / "models" / "model_metadata.json"

SUPPORTED_FRUITS = ["Tomato", "Orange", "Banana"]

class FreshnessPredictorService:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or PRIMARY_MODEL_PATH
        if not os.path.exists(self.model_path) and os.path.exists(BACKUP_MODEL_PATH):
            self.model_path = BACKUP_MODEL_PATH
        self.model_pipeline = None
        self.model_name = "Random Forest Classifier"
        self._load_model()

    def _load_model(self):
        """Loads serialized model pipeline and metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                f"Run model training first via app.services.model_training."
            )
        self.model_pipeline = joblib.load(self.model_path)

        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.model_name = meta.get("model_name", "Random Forest Classifier")
            except Exception:
                pass

    def validate_inputs(
        self,
        fruit: Any,
        temperature_c: Any,
        humidity_percent: Any,
        light_lux: Any,
        co2_ppm: Any
    ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        """
        Validates telemetry inputs against biological and domain boundaries.
        Returns: (is_valid, error_list, cleaned_input_dict)
        """
        errors = []
        cleaned = {}

        # 1. Validate fruit type
        if not fruit or not isinstance(fruit, str):
            errors.append("Fruit must be a non-empty string.")
        else:
            fruit_clean = fruit.strip().capitalize()
            if fruit_clean not in SUPPORTED_FRUITS:
                errors.append(
                    f"Unsupported fruit '{fruit}'. Supported commodities are: {', '.join(SUPPORTED_FRUITS)}."
                )
            else:
                cleaned["fruit"] = fruit_clean

        # 2. Validate Temperature
        try:
            temp = float(temperature_c)
            if np.isnan(temp) or np.isinf(temp):
                errors.append("Temperature must be a finite numerical value.")
            elif temp < -10.0 or temp > 60.0:
                errors.append(f"Temperature {temp}°C is outside plausible produce storage range (-10°C to 60°C).")
            else:
                cleaned["temperature_c"] = temp
        except (ValueError, TypeError):
            errors.append(f"Invalid temperature '{temperature_c}'. Must be numeric.")

        # 3. Validate Humidity
        try:
            humid = float(humidity_percent)
            if np.isnan(humid) or np.isinf(humid):
                errors.append("Humidity must be a finite numerical value.")
            elif humid < 0.0 or humid > 100.0:
                errors.append(f"Humidity {humid}% must be between 0% and 100%.")
            else:
                cleaned["humidity_percent"] = humid
        except (ValueError, TypeError):
            errors.append(f"Invalid humidity '{humidity_percent}'. Must be numeric.")

        # 4. Validate Light Intensity
        try:
            light = float(light_lux)
            if np.isnan(light) or np.isinf(light):
                errors.append("Light intensity must be a finite numerical value.")
            elif light < 0.0:
                errors.append(f"Light intensity {light} Lux cannot be negative.")
            elif light > 2000.0:
                errors.append(f"Light intensity {light} Lux exceeds expected chamber lighting conditions.")
            else:
                cleaned["light_lux"] = light
        except (ValueError, TypeError):
            errors.append(f"Invalid light intensity '{light_lux}'. Must be numeric.")

        # 5. Validate CO2 Level
        try:
            co2 = float(co2_ppm)
            if np.isnan(co2) or np.isinf(co2):
                errors.append("CO2 concentration must be a finite numerical value.")
            elif co2 < 0.0:
                errors.append(f"CO2 concentration {co2} ppm cannot be negative.")
            elif co2 > 5000.0:
                errors.append(f"CO2 concentration {co2} ppm exceeds plausible storage chamber bounds.")
            else:
                cleaned["co2_ppm"] = co2
        except (ValueError, TypeError):
            errors.append(f"Invalid CO2 concentration '{co2_ppm}'. Must be numeric.")

        is_valid = len(errors) == 0
        return is_valid, errors, cleaned if is_valid else None

    def predict(
        self,
        fruit: Any,
        temperature_c: Any,
        humidity_percent: Any,
        light_lux: Any,
        co2_ppm: Any
    ) -> Dict[str, Any]:
        """
        Executes end-to-end validated inference for a single environmental telemetry reading.
        """
        # Validate inputs
        is_valid, errors, cleaned = self.validate_inputs(
            fruit, temperature_c, humidity_percent, light_lux, co2_ppm
        )

        if not is_valid:
            return {
                "success": False,
                "predicted_quality": None,
                "confidence": None,
                "class_probabilities": None,
                "model_name": self.model_name,
                "input_echo": {
                    "fruit": fruit,
                    "temperature_c": temperature_c,
                    "humidity_percent": humidity_percent,
                    "light_lux": light_lux,
                    "co2_ppm": co2_ppm
                },
                "validation_errors": errors,
                "disclaimer": "Inputs failed validation checks."
            }

        # Construct single-row DataFrame for pipeline input
        input_df = pd.DataFrame([cleaned])

        try:
            # Predict quality class
            pred_class = self.model_pipeline.predict(input_df)[0]
            
            # Predict class probabilities
            probs_dict = {}
            confidence = 0.0
            if hasattr(self.model_pipeline, "predict_proba"):
                classes = self.model_pipeline.classes_.tolist()
                probs = self.model_pipeline.predict_proba(input_df)[0]
                probs_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
                confidence = probs_dict.get(str(pred_class), round(float(np.max(probs)), 4))

            return {
                "success": True,
                "predicted_quality": str(pred_class),
                "confidence": confidence,
                "class_probabilities": probs_dict,
                "model_name": self.model_name,
                "input_echo": cleaned,
                "validation_errors": [],
                "disclaimer": (
                    "Model confidence reflects statistical probability under experimental chamber telemetry "
                    "and does not represent guaranteed real-world produce preservation or shelf-life."
                )
            }
        except Exception as e:
            return {
                "success": False,
                "predicted_quality": None,
                "confidence": None,
                "class_probabilities": None,
                "model_name": self.model_name,
                "input_echo": cleaned,
                "validation_errors": [f"Inference exception: {str(e)}"],
                "disclaimer": "An internal error occurred during model prediction execution."
            }

# Global singleton predictor instance
predictor_service = FreshnessPredictorService()

def predict_quality(
    fruit: str,
    temperature_c: float,
    humidity_percent: float,
    light_lux: float,
    co2_ppm: float
) -> Dict[str, Any]:
    """Convenience function calling global predictor service."""
    return predictor_service.predict(fruit, temperature_c, humidity_percent, light_lux, co2_ppm)
