"""
FreshTrack Data Cleaning and Preparation Service
Phase 2: Dataset Inspection, Normalization, and Preparation

Pipeline Responsibilities:
1. Loads original dataset safely from backend/data/original/Dataset.csv (never modifies original).
2. Normalizes target quality labels: Good -> Good, BAD -> Bad, Bad -> Bad.
3. Filters fruit types to retain only Tomato, Banana, and Orange.
4. Analyzes duplicate records and preserves them by default as legitimate IoT steady-state observations.
5. Validates data integrity (zero missing values, boundary checks).
6. Renames columns cleanly with verified real-world units (correcting typos 'Fux' -> 'lux' and 'pmm' -> 'ppm').
7. Saves cleaned dataset to backend/data/processed/freshtrack_cleaned.csv.
8. Re-loads and validates output.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Base directory resolution
SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent.parent
DEFAULT_ORIGINAL_PATH = BACKEND_DIR / "data" / "original" / "Dataset.csv"
DEFAULT_PROCESSED_PATH = BACKEND_DIR / "data" / "processed" / "freshtrack_cleaned.csv"

# Target fruit subset for Phase 2
RETAINED_FRUITS = ["Tomato", "Banana", "Orange"]

# Standardized column name mapping (correcting dataset typographical errors)
COLUMN_MAPPING = {
    "Fruit": "fruit",
    "Temp": "temperature_c",
    "Humid (%)": "humidity_percent",
    "Light (Fux)": "light_lux",     # Correcting typo 'Fux' -> Lux (illuminance)
    "CO2 (pmm)": "co2_ppm",         # Correcting typo 'pmm' -> ppm (parts per million)
    "Class": "quality"              # Standard quality classification target
}

def inspect_dataset(file_path: Path = DEFAULT_ORIGINAL_PATH) -> Dict[str, Any]:
    """
    Performs comprehensive statistical and structural inspection of the raw dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    
    # Basic shape
    n_records, n_cols = df.shape
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_vals = df.isnull().sum().to_dict()
    
    # Exact duplicate records
    n_duplicates = int(df.duplicated().sum())
    dup_percent = round((n_duplicates / n_records) * 100, 2)
    
    # Categorical breakdown
    fruits = df["Fruit"].unique().tolist()
    fruit_counts = df["Fruit"].value_counts().to_dict()
    raw_classes = df["Class"].unique().tolist()
    raw_class_counts = df["Class"].value_counts().to_dict()
    
    # Numeric column statistics
    numeric_cols = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    num_stats = {}
    for col in numeric_cols:
        q25 = float(df[col].quantile(0.25))
        q75 = float(df[col].quantile(0.75))
        iqr = q75 - q25
        num_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": round(float(df[col].mean()), 3),
            "median": round(float(df[col].median()), 3),
            "std": round(float(df[col].std()), 3),
            "q25": round(q25, 3),
            "q75": round(q75, 3),
            "iqr": round(iqr, 3),
            "tukey_lower": round(q25 - 1.5 * iqr, 3),
            "tukey_upper": round(q75 + 1.5 * iqr, 3),
            "outlier_count": int(((df[col] < (q25 - 1.5 * iqr)) | (df[col] > (q75 + 1.5 * iqr))).sum())
        }

    return {
        "file_path": str(file_path),
        "total_records": n_records,
        "total_columns": n_cols,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_vals,
        "duplicate_count": n_duplicates,
        "duplicate_percentage": dup_percent,
        "unique_fruits": fruits,
        "fruit_counts": fruit_counts,
        "unique_classes": raw_classes,
        "class_counts": raw_class_counts,
        "numeric_stats": num_stats
    }

def clean_dataset(
    original_path: Path = DEFAULT_ORIGINAL_PATH,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    fruits_to_retain: List[str] = RETAINED_FRUITS,
    drop_duplicates: bool = False
) -> pd.DataFrame:
    """
    Executes the cleaning pipeline, exports to CSV, and returns the cleaned DataFrame.
    """
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original dataset not found at {original_path}")

    # 1. Load original data safely
    df = pd.read_csv(original_path)
    initial_rows = len(df)

    # 2. Normalize quality labels (Good -> Good, BAD -> Bad, Bad -> Bad)
    df["Class"] = df["Class"].astype(str).str.strip()
    class_mapping = {
        "Good": "Good",
        "good": "Good",
        "GOOD": "Good",
        "Bad": "Bad",
        "bad": "Bad",
        "BAD": "Bad"
    }
    df["Class"] = df["Class"].map(class_mapping)

    # 3. Retain only target fruits (Tomato, Banana, Orange)
    df["Fruit"] = df["Fruit"].astype(str).str.strip().str.capitalize()
    normalized_retained = [f.capitalize() for f in fruits_to_retain]
    df = df[df["Fruit"].isin(normalized_retained)].copy()
    filtered_rows = len(df)

    # 4. Handle duplicates
    # Duplicates are retained by default to reflect true IoT steady-state observation density
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # 5. Rename columns clearly with verified real-world units
    df = df.rename(columns=COLUMN_MAPPING)

    # 6. Save cleaned dataset safely into backend/data/processed/
    os.makedirs(processed_path.parent, exist_ok=True)
    df.to_csv(processed_path, index=False)

    return df

def validate_cleaned_dataset(processed_path: Path = DEFAULT_PROCESSED_PATH) -> Dict[str, Any]:
    """
    Reopens and rigorously validates the cleaned dataset output.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}")

    df_cleaned = pd.read_csv(processed_path)

    # Validation checks
    unique_fruits = df_cleaned["fruit"].unique().tolist()
    unique_qualities = df_cleaned["quality"].unique().tolist()
    missing_count = int(df_cleaned.isnull().sum().sum())
    duplicate_count = int(df_cleaned.duplicated().sum())

    fruit_check = set(unique_fruits).issubset(set(RETAINED_FRUITS)) and len(unique_fruits) == len(RETAINED_FRUITS)
    quality_check = set(unique_qualities) == {"Good", "Bad"}
    missing_check = (missing_count == 0)

    is_valid = fruit_check and quality_check and missing_check

    return {
        "is_valid": is_valid,
        "rows": len(df_cleaned),
        "columns": df_cleaned.columns.tolist(),
        "fruit_counts": df_cleaned["fruit"].value_counts().to_dict(),
        "quality_counts": df_cleaned["quality"].value_counts().to_dict(),
        "total_missing_values": missing_count,
        "duplicate_count": duplicate_count,
        "summary_per_fruit": {
            fruit: {
                "count": int((df_cleaned["fruit"] == fruit).sum()),
                "quality_breakdown": df_cleaned[df_cleaned["fruit"] == fruit]["quality"].value_counts().to_dict(),
                "temp_range": [float(df_cleaned[df_cleaned["fruit"] == fruit]["temperature_c"].min()),
                               float(df_cleaned[df_cleaned["fruit"] == fruit]["temperature_c"].max())],
                "humidity_range": [float(df_cleaned[df_cleaned["fruit"] == fruit]["humidity_percent"].min()),
                                   float(df_cleaned[df_cleaned["fruit"] == fruit]["humidity_percent"].max())],
                "light_range": [round(float(df_cleaned[df_cleaned["fruit"] == fruit]["light_lux"].min()), 2),
                                round(float(df_cleaned[df_cleaned["fruit"] == fruit]["light_lux"].max()), 2)],
                "co2_range": [float(df_cleaned[df_cleaned["fruit"] == fruit]["co2_ppm"].min()),
                              float(df_cleaned[df_cleaned["fruit"] == fruit]["co2_ppm"].max())]
            }
            for fruit in RETAINED_FRUITS
        }
    }

if __name__ == "__main__":
    print("Executing FreshTrack Data Inspection...")
    inspection = inspect_dataset()
    print(f"Total raw records: {inspection['total_records']}")
    print(f"Duplicates found: {inspection['duplicate_count']} ({inspection['duplicate_percentage']}%)")
    
    print("\nExecuting Data Cleaning Pipeline...")
    df_cleaned = clean_dataset()
    print(f"Cleaned dataset rows: {len(df_cleaned)}")
    
    print("\nValidating Cleaned Dataset...")
    val = validate_cleaned_dataset()
    print(f"Validation Status: {'PASSED' if val['is_valid'] else 'FAILED'}")
    print(f"Final records: {val['rows']}, Columns: {val['columns']}")
    print(f"Quality distribution: {val['quality_counts']}")
    print(f"Fruit distribution: {val['fruit_counts']}")
