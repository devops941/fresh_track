"""
FreshTrack ML Training Pipeline
Unified training interface wrapping app.services.model_training
"""

from app.services.model_training import run_training_pipeline

def train():
    """Runs the full model training, evaluation, chart generation, and model storage pipeline."""
    return run_training_pipeline()

if __name__ == "__main__":
    res = train()
    print("Training finished successfully:", res["champion_model"])
