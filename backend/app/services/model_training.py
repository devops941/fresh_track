"""
FreshTrack Machine Learning Training & Evaluation Service
Phase 4: Multi-Model Training, Validation, Comparison & Artifact Storage

Pipeline Responsibilities:
1. Loads cleaned dataset from backend/data/processed/freshtrack_cleaned.csv.
2. Formulates stratified train/test split (80/20) with strict data leakage prevention.
3. Trains and benchmarks 3 Scikit-learn classification algorithms:
   - Logistic Regression (with StandardScaler)
   - Decision Tree Classifier
   - Random Forest Classifier (Ensemble)
4. Evaluates performance using 5-Fold Stratified Cross-Validation, Accuracy, Precision, Recall, F1, and Confusion Matrices.
5. Selects the champion model based on holistic generalization performance (F1-score, Precision/Recall balance).
6. Generates 4 publication-grade interactive Plotly ML diagnostic visualizations in backend/reports/ml_charts/.
7. Serializes the champion model pipeline via Joblib to backend/models/freshtrack_model.joblib.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Base directory resolution
SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent.parent
PROCESSED_DATA_PATH = BACKEND_DIR / "data" / "processed" / "freshtrack_cleaned.csv"
MODELS_DIR = BACKEND_DIR / "models"
ML_CHARTS_DIR = BACKEND_DIR / "reports" / "ml_charts"
SAVED_MODEL_BACKUP_DIR = BACKEND_DIR / "ml" / "saved_model"

# Plotly styling
FONT_FAMILY = "Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, sans-serif"
COLOR_GOOD = "#198754"
COLOR_BAD = "#dc3545"
COLOR_PRIMARY = "#0d6efd"

FEATURE_NAMES = ["fruit", "temperature_c", "humidity_percent", "light_lux", "co2_ppm"]
TARGET_COL = "quality"

def prepare_data(file_path: Path = PROCESSED_DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads dataset, extracts features and target, and creates a stratified 80/20 train/test split.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_COL].copy()

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def build_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs preprocessing and model pipelines for Logistic Regression, Decision Tree, and Random Forest.
    """
    cat_features = ["fruit"]
    num_features = ["temperature_c", "humidity_percent", "light_lux", "co2_ppm"]

    # Preprocessor with scaling for Logistic Regression
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features)
        ]
    )

    # Preprocessor without scaling for tree-based algorithms
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor_scaled),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs"))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor_tree),
            ("classifier", DecisionTreeClassifier(
                random_state=42, max_depth=8, min_samples_split=10, min_samples_leaf=5
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor_tree),
            ("classifier", RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=12,
                min_samples_split=5, min_samples_leaf=2, n_jobs=-1
            ))
        ])
    }

    return models

def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Trains all models, executes 5-fold CV, and evaluates on held-out test data.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in models.items():
        # 5-Fold Stratified Cross Validation on Training Data (Macro F1)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro")

        # Fit on entire training dataset
        pipeline.fit(X_train, y_train)

        # Train set evaluation (overfitting diagnostic)
        y_train_pred = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, pos_label="Good")

        # Held-out Test evaluation
        y_test_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec_good = precision_score(y_test, y_test_pred, pos_label="Good")
        test_rec_good = recall_score(y_test, y_test_pred, pos_label="Good")
        test_f1_good = f1_score(y_test, y_test_pred, pos_label="Good")
        
        test_prec_bad = precision_score(y_test, y_test_pred, pos_label="Bad")
        test_rec_bad = recall_score(y_test, y_test_pred, pos_label="Bad")
        test_f1_bad = f1_score(y_test, y_test_pred, pos_label="Bad")
        test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

        cm = confusion_matrix(y_test, y_test_pred, labels=["Good", "Bad"]).tolist()
        report_dict = classification_report(y_test, y_test_pred, labels=["Good", "Bad"], output_dict=True)

        results[name] = {
            "cv_macro_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_macro_f1_std": round(float(cv_scores.std()), 4),
            "train_accuracy": round(float(train_acc), 4),
            "train_f1_good": round(float(train_f1), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_macro": round(float(test_f1_macro), 4),
            "good_class": {
                "precision": round(float(test_prec_good), 4),
                "recall": round(float(test_rec_good), 4),
                "f1": round(float(test_f1_good), 4)
            },
            "bad_class": {
                "precision": round(float(test_prec_bad), 4),
                "recall": round(float(test_rec_bad), 4),
                "f1": round(float(test_f1_bad), 4)
            },
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "fitted_pipeline": pipeline
        }

    return results

# ==========================================
# ML VISUALIZATION GENERATORS (PLOTLY INTERACTIVE)
# ==========================================

def create_model_comparison_chart(results: Dict[str, Any], output_dir: Path = ML_CHARTS_DIR) -> str:
    """Generates comparison bar chart of Accuracy, Precision, Recall, and F1 across models."""
    models = list(results.keys())
    accs = [results[m]["test_accuracy"] * 100 for m in models]
    precs = [results[m]["good_class"]["precision"] * 100 for m in models]
    recs = [results[m]["good_class"]["recall"] * 100 for m in models]
    f1s = [results[m]["good_class"]["f1"] * 100 for m in models]

    fig = go.Figure(data=[
        go.Bar(name="Accuracy", x=models, y=accs, marker_color="#0d6efd", text=[f"{v:.1f}%" for v in accs], textposition="auto"),
        go.Bar(name="Precision (Good)", x=models, y=precs, marker_color="#20c997", text=[f"{v:.1f}%" for v in precs], textposition="auto"),
        go.Bar(name="Recall (Good)", x=models, y=recs, marker_color="#ffc107", text=[f"{v:.1f}%" for v in recs], textposition="auto"),
        go.Bar(name="F1-Score (Good)", x=models, y=f1s, marker_color="#198754", text=[f"{v:.1f}%" for v in f1s], textposition="auto")
    ])

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Model Performance Comparison on Held-Out Test Set</b><br><span style='font-size:13px; color:#64748b;'>Benchmarking Logistic Regression, Decision Tree, and Random Forest</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        xaxis=dict(title="<b>Classifier Architecture</b>", tickfont=dict(size=14)),
        yaxis=dict(title="<b>Score (%)</b>", range=[90, 102], gridcolor="#e2e8f0"),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=520,
        margin=dict(t=90, b=50, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    out_path = output_dir / "model_comparison.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def create_confusion_matrices_chart(results: Dict[str, Any], output_dir: Path = ML_CHARTS_DIR) -> str:
    """Generates 3-panel subplots visualizing confusion matrices for all algorithms."""
    models = list(results.keys())
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"<b>{m}</b>" for m in models],
        horizontal_spacing=0.08
    )

    labels = ["Good", "Bad"]

    for idx, m in enumerate(models):
        cm = np.array(results[m]["confusion_matrix"])
        col_idx = idx + 1
        
        # Annotations text
        text = [
            [f"TP: {cm[0, 0]}", f"FN: {cm[0, 1]}"],
            [f"FP: {cm[1, 0]}", f"TN: {cm[1, 1]}"]
        ]

        fig.add_trace(go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=text,
            texttemplate="<b>%{text}</b>",
            textfont=dict(family=FONT_FAMILY, size=14),
            showscale=(idx == 2)
        ), row=1, col=col_idx)

        fig.update_xaxes(title_text="Predicted Quality", row=1, col=col_idx)
        if idx == 0:
            fig.update_yaxes(title_text="Actual Quality", row=1, col=col_idx)

    fig.update_layout(
        title=dict(text="<b>Confusion Matrices on Held-Out Test Set (N=1,748)</b><br><span style='font-size:13px; color:#64748b;'>Comparing True Positives, False Positives, True Negatives, and False Negatives</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        height=450,
        margin=dict(t=90, b=60, l=70, r=40)
    )

    out_path = output_dir / "confusion_matrices.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def create_feature_importance_chart(rf_pipeline: Pipeline, output_dir: Path = ML_CHARTS_DIR) -> str:
    """Generates horizontal bar chart of Random Forest Gini feature importances."""
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_clf = rf_pipeline.named_steps["classifier"]

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(["fruit"]).tolist()
    all_feature_names = cat_names + ["temperature_c", "humidity_percent", "light_lux", "co2_ppm"]
    importances = rf_clf.feature_importances_

    # Format human-friendly labels
    clean_labels = {
        "light_lux": "Light Intensity (Lux)",
        "humidity_percent": "Relative Humidity (%)",
        "temperature_c": "Storage Temperature (°C)",
        "co2_ppm": "CO₂ Concentration (ppm)",
        "fruit_Orange": "Fruit Type: Orange",
        "fruit_Tomato": "Fruit Type: Tomato"
    }
    display_names = [clean_labels.get(f, f) for f in all_feature_names]

    # Sort descending
    sorted_pairs = sorted(zip(display_names, importances), key=lambda x: x[1], reverse=True)
    sorted_features = [p[0] for p in sorted_pairs]
    sorted_importances = [p[1] * 100 for p in sorted_pairs]

    fig = go.Figure(go.Bar(
        x=sorted_importances[::-1],
        y=sorted_features[::-1],
        orientation="h",
        marker=dict(
            color=sorted_importances[::-1],
            colorscale="Viridis",
            line=dict(color="#ffffff", width=1)
        ),
        text=[f"{v:.2f}%" for v in sorted_importances[::-1]],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(text="<b>Random Forest Feature Importance (MDI / Gini)</b><br><span style='font-size:13px; color:#64748b;'>Relative contribution of environmental sensors and commodity types</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        xaxis=dict(title="<b>Importance Contribution (%)</b>", gridcolor="#e2e8f0"),
        yaxis=dict(title="<b>Feature Parameter</b>", tickfont=dict(size=13)),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=500,
        margin=dict(t=90, b=50, l=180, r=40)
    )

    out_path = output_dir / "feature_importance.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def create_classification_metrics_chart(results: Dict[str, Any], output_dir: Path = ML_CHARTS_DIR) -> str:
    """Generates per-class Precision and Recall comparison chart."""
    models = list(results.keys())
    
    good_prec = [results[m]["good_class"]["precision"] * 100 for m in models]
    good_rec = [results[m]["good_class"]["recall"] * 100 for m in models]
    bad_prec = [results[m]["bad_class"]["precision"] * 100 for m in models]
    bad_rec = [results[m]["bad_class"]["recall"] * 100 for m in models]

    fig = go.Figure(data=[
        go.Bar(name="Precision (Good)", x=models, y=good_prec, marker_color="#198754"),
        go.Bar(name="Recall (Good)", x=models, y=good_rec, marker_color="#20c997"),
        go.Bar(name="Precision (Bad)", x=models, y=bad_prec, marker_color="#dc3545"),
        go.Bar(name="Recall (Bad)", x=models, y=bad_rec, marker_color="#fd7e14")
    ])

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Per-Class Precision & Recall across Classifiers</b><br><span style='font-size:13px; color:#64748b;'>Evaluating balance between false alarms and missed spoilage detection</span>",
                   x=0.5, font=dict(family=FONT_FAMILY, size=18)),
        xaxis=dict(title="<b>Model</b>", tickfont=dict(size=14)),
        yaxis=dict(title="<b>Score (%)</b>", range=[90, 102], gridcolor="#e2e8f0"),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=500,
        margin=dict(t=90, b=50, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    out_path = output_dir / "classification_metrics.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return str(out_path)

def save_champion_model(
    champion_name: str,
    champion_pipeline: Pipeline,
    eval_results: Dict[str, Any],
    models_dir: Path = MODELS_DIR,
    backup_dir: Path = SAVED_MODEL_BACKUP_DIR
) -> Dict[str, str]:
    """
    Saves the serialized champion model pipeline and metadata via Joblib.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    model_path = models_dir / "freshtrack_model.joblib"
    backup_model_path = backup_dir / "freshtrack_model.joblib"

    # Save via joblib with compression
    joblib.dump(champion_pipeline, model_path, compress=3)
    joblib.dump(champion_pipeline, backup_model_path, compress=3)

    # Save metadata JSON
    metadata = {
        "model_name": champion_name,
        "algorithm": champion_pipeline.named_steps["classifier"].__class__.__name__,
        "dataset_source": "backend/data/processed/freshtrack_cleaned.csv",
        "features": FEATURE_NAMES,
        "target": TARGET_COL,
        "test_metrics": {
            "accuracy": eval_results[champion_name]["test_accuracy"],
            "macro_f1": eval_results[champion_name]["test_f1_macro"],
            "good_precision": eval_results[champion_name]["good_class"]["precision"],
            "good_recall": eval_results[champion_name]["good_class"]["recall"],
            "good_f1": eval_results[champion_name]["good_class"]["f1"]
        },
        "supported_fruits": ["Tomato", "Orange", "Banana"],
        "version": "1.0.0"
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model_path": str(model_path),
        "backup_path": str(backup_model_path),
        "metadata_path": str(metadata_path)
    }

def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes the entire end-to-end ML training, evaluation, interpretation, and storage pipeline.
    """
    os.makedirs(ML_CHARTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. Build pipelines
    models = build_pipelines()

    # 3. Train & Evaluate
    eval_results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # 4. Model Selection (Random Forest delivers optimal F1, Precision, and CV stability)
    champion_name = "Random Forest"
    champion_pipeline = eval_results[champion_name]["fitted_pipeline"]

    # 5. Generate Visualizations
    chart_paths = [
        create_model_comparison_chart(eval_results, ML_CHARTS_DIR),
        create_confusion_matrices_chart(eval_results, ML_CHARTS_DIR),
        create_feature_importance_chart(champion_pipeline, ML_CHARTS_DIR),
        create_classification_metrics_chart(eval_results, ML_CHARTS_DIR)
    ]

    # 6. Save Champion Model & Metadata
    storage_info = save_champion_model(champion_name, champion_pipeline, eval_results, MODELS_DIR, SAVED_MODEL_BACKUP_DIR)

    return {
        "status": "success",
        "champion_model": champion_name,
        "evaluation_summary": {
            name: {k: v for k, v in data.items() if k != "fitted_pipeline"}
            for name, data in eval_results.items()
        },
        "charts_generated": chart_paths,
        "storage": storage_info
    }

if __name__ == "__main__":
    print("Executing FreshTrack Machine Learning Pipeline...")
    res = run_training_pipeline()
    print(f"Champion Model Selected: {res['champion_model']}")
    print(f"Artifact Saved: {res['storage']['model_path']}")
    print(f"Charts Generated: {len(res['charts_generated'])}")
