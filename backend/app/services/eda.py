"""
FreshTrack Exploratory Data Analysis (EDA) Service
Phase 3: Data Analysis & Visualization Pipeline

Pipeline Responsibilities:
1. Loads cleaned dataset from backend/data/processed/freshtrack_cleaned.csv.
2. Computes statistical summaries (overall, fruit-wise, environmental, correlation).
3. Generates publication-grade, interactive Plotly visualizations:
   - Overall Quality Donut Chart (Good vs. Bad)
   - Fruit-Wise Quality Breakdown (Stacked / Grouped Bar)
   - Environmental Conditions Distribution & Box Plots
   - Pearson Correlation Heatmap
   - Commodity-Specific Environmental Comparison
4. Exports all interactive charts as standalone HTML files into backend/reports/charts/.
5. Provides programmatic analysis APIs for FastAPI routers.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Base directories
SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent.parent
PROCESSED_DATA_PATH = BACKEND_DIR / "data" / "processed" / "freshtrack_cleaned.csv"
CHARTS_DIR = BACKEND_DIR / "reports" / "charts"

# Theme colors
COLOR_GOOD = "#198754"   # Forest Green
COLOR_BAD = "#dc3545"    # Crimson Red
COLOR_PRIMARY = "#0d6efd" # Royal Blue
COLOR_ACCENT = "#ffc107"  # Amber

FONT_FAMILY = "Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, sans-serif"

def load_data(file_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Loads and validates the cleaned dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {file_path}")
    return pd.read_csv(file_path)

def get_overall_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes high-level dataset metrics."""
    total_records = len(df)
    quality_counts = df["quality"].value_counts().to_dict()
    fruit_counts = df["fruit"].value_counts().to_dict()
    
    good_count = quality_counts.get("Good", 0)
    bad_count = quality_counts.get("Bad", 0)
    
    return {
        "total_records": total_records,
        "fruit_counts": fruit_counts,
        "quality_counts": quality_counts,
        "good_percentage": round((good_count / total_records) * 100, 2),
        "bad_percentage": round((bad_count / total_records) * 100, 2)
    }

def get_fruit_wise_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes cross-tabulated fruit vs quality distributions."""
    ct = pd.crosstab(df["fruit"], df["quality"])
    ct_pct = pd.crosstab(df["fruit"], df["quality"], normalize="index") * 100
    
    results = {}
    for fruit in ct.index:
        good = int(ct.loc[fruit, "Good"]) if "Good" in ct.columns else 0
        bad = int(ct.loc[fruit, "Bad"]) if "Bad" in ct.columns else 0
        total = good + bad
        good_pct = round((good / total) * 100, 2) if total > 0 else 0
        bad_pct = round((bad / total) * 100, 2) if total > 0 else 0
        results[fruit] = {
            "total": total,
            "good_count": good,
            "bad_count": bad,
            "good_percentage": good_pct,
            "bad_percentage": bad_pct
        }
    return results

def get_environmental_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes descriptive statistics for environmental parameters by quality and fruit."""
    env_cols = ["temperature_c", "humidity_percent", "light_lux", "co2_ppm"]
    
    overall_by_quality = {}
    for q in ["Good", "Bad"]:
        sub = df[df["quality"] == q]
        overall_by_quality[q] = {
            col: {
                "mean": round(float(sub[col].mean()), 2),
                "median": round(float(sub[col].median()), 2),
                "std": round(float(sub[col].std()), 2),
                "min": round(float(sub[col].min()), 2),
                "max": round(float(sub[col].max()), 2)
            }
            for col in env_cols
        }
        
    grouped_fruit_quality = {}
    for fruit in df["fruit"].unique():
        grouped_fruit_quality[fruit] = {}
        for q in ["Good", "Bad"]:
            sub = df[(df["fruit"] == fruit) & (df["quality"] == q)]
            grouped_fruit_quality[fruit][q] = {
                col: {
                    "mean": round(float(sub[col].mean()), 2),
                    "median": round(float(sub[col].median()), 2),
                    "min": round(float(sub[col].min()), 2),
                    "max": round(float(sub[col].max()), 2)
                }
                for col in env_cols
            }
            
    return {
        "overall_by_quality": overall_by_quality,
        "grouped_by_fruit_and_quality": grouped_fruit_quality
    }

def get_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Pearson correlation coefficients including binary quality."""
    env_cols = ["temperature_c", "humidity_percent", "light_lux", "co2_ppm"]
    df_corr = df[env_cols].copy()
    df_corr["quality_binary"] = (df["quality"] == "Good").astype(int)
    corr_matrix = df_corr.corr().round(3)
    
    return {
        "columns": corr_matrix.columns.tolist(),
        "matrix": corr_matrix.to_dict()
    }

# ==========================================
# VISUALIZATION GENERATORS (PLOTLY INTERACTIVE)
# ==========================================

def create_donut_chart(df: pd.DataFrame, output_dir: Path = CHARTS_DIR) -> str:
    """Generates Good vs. Bad Quality Donut Chart."""
    counts = df["quality"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=[COLOR_GOOD if l == "Good" else COLOR_BAD for l in labels],
                    line=dict(color="#ffffff", width=2)),
        textinfo="label+percent+value",
        texttemplate="<b>%{label}</b><br>%{value:,} (%{percent:.1%})",
        hoverinfo="label+value+percent",
        textfont=dict(family=FONT_FAMILY, size=14)
    )])
    
    total = len(df)
    fig.update_layout(
        title=dict(text="<b>Overall Fruit Quality Distribution</b><br><span style='font-size:13px; color:#64748b;'>FreshTrack IoT Monitoring (8,738 records)</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        annotations=[dict(text=f"<b>Total</b><br>{total:,}", x=0.5, y=0.5, font_size=18, font_family=FONT_FAMILY, showarrow=False)],
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(t=80, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    out_path = output_dir / "good_vs_bad_donut.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def create_fruit_wise_bar_chart(df: pd.DataFrame, output_dir: Path = CHARTS_DIR) -> str:
    """Generates Grouped Bar Chart of Fruit-Wise Quality Breakdown."""
    ct = pd.crosstab(df["fruit"], df["quality"])
    fruits = ct.index.tolist()
    good_counts = ct["Good"].tolist()
    bad_counts = ct["Bad"].tolist()
    
    fig = go.Figure(data=[
        go.Bar(name="Good Quality", x=fruits, y=good_counts,
               marker_color=COLOR_GOOD, text=[f"{c:,}" for c in good_counts], textposition="auto"),
        go.Bar(name="Bad Quality", x=fruits, y=bad_counts,
               marker_color=COLOR_BAD, text=[f"{c:,}" for c in bad_counts], textposition="auto")
    ])
    
    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Fruit-Wise Quality Distribution (Counts)</b><br><span style='font-size:13px; color:#64748b;'>Comparing Tomato, Orange, and Banana Quality States</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        xaxis=dict(title="<b>Fruit Commodity</b>", tickfont=dict(size=14)),
        yaxis=dict(title="<b>Observation Count</b>", gridcolor="#e2e8f0"),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        margin=dict(t=80, b=50, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    out_path = output_dir / "fruit_wise_quality_bar.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def create_environmental_comparison_chart(df: pd.DataFrame, output_dir: Path = CHARTS_DIR) -> str:
    """Generates 4-Panel Subplot Comparison of Environmental Conditions vs Quality."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Temperature (°C) by Quality</b>",
            "<b>Relative Humidity (%) by Quality</b>",
            "<b>Light Intensity (Lux) by Quality</b>",
            "<b>CO₂ Concentration (ppm) by Quality</b>"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Temperature
    for q, color in [("Good", COLOR_GOOD), ("Bad", COLOR_BAD)]:
        sub = df[df["quality"] == q]
        fig.add_trace(go.Box(y=sub["temperature_c"], name=q, marker_color=color, showlegend=(q=="Good")), row=1, col=1)
        fig.add_trace(go.Box(y=sub["humidity_percent"], name=q, marker_color=color, showlegend=False), row=1, col=2)
        fig.add_trace(go.Box(y=sub["light_lux"], name=q, marker_color=color, showlegend=False), row=2, col=1)
        fig.add_trace(go.Box(y=sub["co2_ppm"], name=q, marker_color=color, showlegend=False), row=2, col=2)
        
    fig.update_layout(
        title=dict(text="<b>Environmental Sensor Parameters vs Fruit Quality</b><br><span style='font-size:13px; color:#64748b;'>Distribution comparison across Temperature, Humidity, Light, and CO₂</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=700,
        margin=dict(t=100, b=50, l=60, r=40)
    )
    fig.update_yaxes(gridcolor="#e2e8f0")
    
    out_path = output_dir / "environmental_conditions_comparison.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path = CHARTS_DIR) -> str:
    """Generates Annotated Heatmap of Correlation Matrix."""
    env_cols = ["temperature_c", "humidity_percent", "light_lux", "co2_ppm"]
    df_corr = df[env_cols].copy()
    df_corr["quality_good"] = (df["quality"] == "Good").astype(int)
    corr = df_corr.corr().round(3)
    
    labels = ["Temperature (°C)", "Humidity (%)", "Light (Lux)", "CO₂ (ppm)", "Quality (Good=1)"]
    z = corr.values
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=z,
        texttemplate="<b>%{text:.2f}</b>",
        textfont=dict(family=FONT_FAMILY, size=13),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text="<b>Sensor Telemetry & Quality Correlation Heatmap</b><br><span style='font-size:13px; color:#64748b;'>Pearson correlation coefficients across features and quality state</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        margin=dict(t=90, b=50, l=120, r=50),
        width=700,
        height=600
    )
    
    out_path = output_dir / "correlation_heatmap.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def create_fruit_comparison_chart(df: pd.DataFrame, output_dir: Path = CHARTS_DIR) -> str:
    """Generates Grouped Metric Comparison across Fruit Commodities."""
    summary = df.groupby(["fruit", "quality"]).agg({
        "temperature_c": "mean",
        "humidity_percent": "mean",
        "light_lux": "mean",
        "co2_ppm": "mean"
    }).reset_index()
    
    fruits = sorted(df["fruit"].unique().tolist())
    
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("<b>Mean Temp (°C)</b>", "<b>Mean Humid (%)</b>", "<b>Mean Light (Lux)</b>", "<b>Mean CO₂ (ppm)</b>")
    )
    
    for q, color in [("Good", COLOR_GOOD), ("Bad", COLOR_BAD)]:
        sub = summary[summary["quality"] == q]
        fig.add_trace(go.Bar(name=f"{q} Quality", x=sub["fruit"], y=sub["temperature_c"], marker_color=color, showlegend=(q=="Good" or q=="Bad")), row=1, col=1)
        fig.add_trace(go.Bar(name=f"{q} Quality", x=sub["fruit"], y=sub["humidity_percent"], marker_color=color, showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name=f"{q} Quality", x=sub["fruit"], y=sub["light_lux"], marker_color=color, showlegend=False), row=1, col=3)
        fig.add_trace(go.Bar(name=f"{q} Quality", x=sub["fruit"], y=sub["co2_ppm"], marker_color=color, showlegend=False), row=1, col=4)
        
    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Commodity Environmental Comparison by Quality State</b><br><span style='font-size:13px; color:#64748b;'>Comparing mean environmental requirements across Tomato, Orange, and Banana</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=480,
        margin=dict(t=90, b=50, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )
    fig.update_yaxes(gridcolor="#e2e8f0")
    
    out_path = output_dir / "fruit_comparison_chart.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def generate_all_charts(df: pd.DataFrame, output_dir: Path = CHARTS_DIR) -> List[str]:
    """Generates all 5 required interactive Plotly charts and returns their filepaths."""
    os.makedirs(output_dir, exist_ok=True)
    charts = [
        create_donut_chart(df, output_dir),
        create_fruit_wise_bar_chart(df, output_dir),
        create_environmental_comparison_chart(df, output_dir),
        create_correlation_heatmap(df, output_dir),
        create_fruit_comparison_chart(df, output_dir)
    ]
    return charts

def run_eda_pipeline() -> Dict[str, Any]:
    """Executes full EDA computation and chart generation pipeline."""
    df = load_data()
    overall = get_overall_summary(df)
    fruit_wise = get_fruit_wise_summary(df)
    environmental = get_environmental_summary(df)
    correlation = get_correlation_matrix(df)
    generated_charts = generate_all_charts(df)
    
    return {
        "status": "success",
        "overall_summary": overall,
        "fruit_wise_summary": fruit_wise,
        "environmental_summary": environmental,
        "correlation_matrix": correlation,
        "charts_generated": generated_charts
    }

if __name__ == "__main__":
    print("Executing FreshTrack EDA Pipeline...")
    res = run_eda_pipeline()
    print("EDA Complete!")
    print(f"Total Records Analyzed: {res['overall_summary']['total_records']}")
    print(f"Charts Generated in: {CHARTS_DIR}")
    for c in res["charts_generated"]:
        print(f" - {Path(c).name}")
