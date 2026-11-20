import os
from pathlib import Path
from typing import List

# Read .env file from backend directory if present
def load_env_file():
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key not in os.environ:
                        os.environ[key] = val

load_env_file()

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "FreshTrack API")
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    # Bind to 0.0.0.0 so localhost, 127.0.0.1, and all local IP aliases work
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # SQLite Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app/database/freshtrack.db")
    
    # CORS Origins
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500,http://localhost:5555,http://127.0.0.1:5555,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080,http://127.0.0.1:8080"
    )
    
    # Base paths relative to backend directory
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    ORIGINAL_DATA_PATH: Path = BACKEND_DIR / os.getenv("ORIGINAL_DATA_PATH", "data/original/Dataset.csv")
    PROCESSED_DATA_PATH: Path = BACKEND_DIR / os.getenv("PROCESSED_DATA_PATH", "data/processed/freshtrack_cleaned.csv")
    MODEL_PATH: Path = BACKEND_DIR / "models" / "freshtrack_model.joblib"
    MODEL_METADATA_PATH: Path = BACKEND_DIR / "models" / "model_metadata.json"
    BACKUP_MODEL_PATH: Path = BACKEND_DIR / "ml" / "saved_model" / "freshtrack_model.joblib"
    REPORTS_DIR: Path = BACKEND_DIR / os.getenv("REPORTS_DIR", "reports")

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS:
            return ["*"]
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        # Add wildcard support in development
        if self.DEBUG and "*" not in origins:
            origins.append("*")
        return origins

settings = Settings()
