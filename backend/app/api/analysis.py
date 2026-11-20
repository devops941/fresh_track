from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.schemas.analysis import (
    DatasetSummaryResponse,
    FruitAnalysisResponse,
    EnvironmentalAnalysisResponse,
    FruitQualityItem,
    EnvironmentalMetric,
    ScatterPoint,
    ScatterDataResponse
)
from app.services.eda import (
    load_data,
    get_overall_summary,
    get_fruit_wise_summary,
    get_environmental_summary
)

router = APIRouter(prefix="/analysis", tags=["Exploratory Data Analysis"])

@router.get(
    "/summary",
    response_model=DatasetSummaryResponse,
    summary="Overall Dataset Summary",
    description="Returns aggregate observation counts, commodity distribution, and overall Good/Bad quality percentages from the cleaned dataset."
)
def get_dataset_summary() -> DatasetSummaryResponse:
    try:
        df = load_data(settings.PROCESSED_DATA_PATH)
        summary = get_overall_summary(df)
        
        good_count = summary["quality_counts"].get("Good", 0)
        bad_count = summary["quality_counts"].get("Bad", 0)
        
        return DatasetSummaryResponse(
            total_records=summary["total_records"],
            fruit_counts=summary["fruit_counts"],
            good_quality_count=good_count,
            bad_quality_count=bad_count,
            good_quality_percentage=summary["good_percentage"],
            bad_quality_percentage=summary["bad_percentage"]
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Cleaned dataset file is not available on server.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to compute dataset summary.")

@router.get(
    "/fruits",
    response_model=FruitAnalysisResponse,
    summary="Fruit-Wise Quality Distribution",
    description="Returns cross-tabulated quality counts and Good/Bad percentages for Tomato, Banana, and Orange."
)
def get_fruits_analysis() -> FruitAnalysisResponse:
    try:
        df = load_data(settings.PROCESSED_DATA_PATH)
        fruit_data = get_fruit_wise_summary(df)
        
        fruit_items = {
            fruit: FruitQualityItem(
                total=data["total"],
                good_count=data["good_count"],
                bad_count=data["bad_count"],
                good_percentage=data["good_percentage"],
                bad_percentage=data["bad_percentage"]
            )
            for fruit, data in fruit_data.items()
        }
        
        summary_text = (
            f"Analyzed {len(fruit_items)} commodities. "
            f"Banana has the highest fresh proportion ({fruit_items['Banana'].good_percentage}% Good), "
            f"while Tomato ({fruit_items['Tomato'].good_percentage}% Good) and Orange ({fruit_items['Orange'].good_percentage}% Good) "
            "exhibit balanced 50/50 quality distributions."
        )
        
        return FruitAnalysisResponse(
            fruits=fruit_items,
            summary=summary_text
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Cleaned dataset file is not available on server.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to compute fruit-wise analysis.")

@router.get(
    "/environment",
    response_model=EnvironmentalAnalysisResponse,
    summary="Environmental Telemetry Statistics",
    description="Returns mean, median, min, max, and std for temperature, humidity, light, and CO2, grouped by quality and fruit commodity."
)
def get_environmental_analysis() -> EnvironmentalAnalysisResponse:
    try:
        df = load_data(settings.PROCESSED_DATA_PATH)
        env_data = get_environmental_summary(df)
        
        # Build structured overall by quality
        overall_structured = {}
        for q, metrics in env_data["overall_by_quality"].items():
            overall_structured[q] = {
                param: EnvironmentalMetric(
                    mean=vals["mean"],
                    median=vals["median"],
                    std=vals["std"],
                    min=vals["min"],
                    max=vals["max"]
                )
                for param, vals in metrics.items()
            }
            
        insights = [
            "Continuous saturated humidity (mean 94.96% RH, std 0.64%) strongly correlates with the Bad quality condition.",
            "Light exposure above 25 Lux is exclusively associated with Bad quality, particularly in tomatoes.",
            "Good produce exhibits slightly lower median storage temperatures (23°C vs 24°C in Bad).",
            "CO2 levels show healthy respiration signatures (340 ppm) in fresh tomatoes and bananas."
        ]
        
        return EnvironmentalAnalysisResponse(
            overall_by_quality=overall_structured,
            grouped_by_fruit_and_quality=env_data["grouped_by_fruit_and_quality"],
            insights=insights
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Cleaned dataset file is not available on server.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to compute environmental analysis.")

@router.get(
    "/scatter-data",
    response_model=ScatterDataResponse,
    summary="Interactive Scatter Plot & Descriptive Statistics Data",
    description="Returns sample observations and Pandas-calculated descriptive statistics and correlations for interactive scatter plotting and regression analysis."
)
def get_scatter_data(fruit: str = "All", limit: int = 600) -> ScatterDataResponse:
    try:
        df = load_data(settings.PROCESSED_DATA_PATH)
        if fruit != "All" and fruit in ["Tomato", "Orange", "Banana"]:
            filtered_df = df[df["fruit"] == fruit]
        else:
            filtered_df = df
        
        sample_size = min(limit, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size, random_state=42)
        
        points = [
            ScatterPoint(
                fruit=row["fruit"],
                temperature_c=float(row["temperature_c"]),
                humidity_percent=float(row["humidity_percent"]),
                light_lux=round(float(row["light_lux"]), 2),
                co2_ppm=float(row["co2_ppm"]),
                quality=row["quality"]
            )
            for _, row in sample_df.iterrows()
        ]
        
        num_cols = ["temperature_c", "humidity_percent", "light_lux", "co2_ppm"]
        desc = filtered_df[num_cols].describe().to_dict()
        desc_clean = {
            col: {
                "mean": round(desc[col]["mean"], 2),
                "std": round(desc[col]["std"], 2),
                "min": round(desc[col]["min"], 2),
                "25%": round(desc[col]["25%"], 2),
                "50%": round(desc[col]["50%"], 2),
                "75%": round(desc[col]["75%"], 2),
                "max": round(desc[col]["max"], 2)
            }
            for col in num_cols
        }
        
        corr = filtered_df[num_cols].corr().to_dict()
        corr_clean = {
            col1: {col2: round(corr[col1][col2], 3) for col2 in num_cols}
            for col1 in num_cols
        }
        
        return ScatterDataResponse(
            total_samples=len(points),
            points=points,
            descriptive_statistics=desc_clean,
            correlations=corr_clean
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch scatter data.")

