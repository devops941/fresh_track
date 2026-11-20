"""
FreshTrack ML Prediction Engine
Unified inference interface wrapping app.services.model_prediction
"""

import os
from pathlib import Path
from typing import Dict, Any

from app.services.model_prediction import predictor_service, predict_quality

class FreshnessPredictor:
    def __init__(self):
        self.service = predictor_service

    def predict(self, fruit: str, temp: float, humid: float, light: float, co2: float) -> Dict[str, Any]:
        """
        Executes inference for a single input record.
        """
        return self.service.predict(
            fruit=fruit,
            temperature_c=temp,
            humidity_percent=humid,
            light_lux=light,
            co2_ppm=co2
        )

predictor = FreshnessPredictor()

if __name__ == "__main__":
    # Test sample prediction
    res = predictor.predict(fruit="Tomato", temp=23.0, humid=90.0, light=9.5, co2=340.0)
    print("Sample prediction result:")
    print(res)
