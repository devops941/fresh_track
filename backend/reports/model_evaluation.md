# FreshTrack – Phase 4: Machine Learning & Quality Prediction Report

**Project:** FreshTrack – Fruit Freshness & Quality Intelligence  
**Phase:** Phase 4: Machine Learning & Quality Prediction  
**Dataset Source:** [backend/data/processed/freshtrack_cleaned.csv](file:///d:/collegeprojects/fresh_track/backend/data/processed/freshtrack_cleaned.csv) (8,738 rows)  
**Training Service:** [backend/app/services/model_training.py](file:///d:/collegeprojects/fresh_track/backend/app/services/model_training.py)  
**Inference Service:** [backend/app/services/model_prediction.py](file:///d:/collegeprojects/fresh_track/backend/app/services/model_prediction.py)  
**Trained Model Artifact:** [backend/models/freshtrack_model.joblib](file:///d:/collegeprojects/fresh_track/backend/models/freshtrack_model.joblib)  
**Interactive Visualizations:** [backend/reports/ml_charts/](file:///d:/collegeprojects/fresh_track/backend/reports/ml_charts/)  

---

## 1. Executive Summary & Objective

The objective of Phase 4 is to build, benchmark, interpret, and store an empirical machine learning classification system capable of predicting fruit quality condition (**Good** vs. **Bad**) based on real-time environmental telemetry:
- **Fruit Type:** Tomato, Orange, Banana
- **Temperature:** Storage chamber ambient temperature (°C)
- **Relative Humidity:** Storage relative humidity (% RH)
- **Light Intensity:** Ambient illuminance (Lux)
- **CO₂ Concentration:** Carbon dioxide accumulation (ppm)

The machine learning pipeline was trained strictly on the preprocessed 8,738 observations with an **80/20 stratified train/test split** and evaluated using **5-Fold Stratified Cross-Validation**. 

---

## 2. Data Preparation & Leakage Prevention

| Step | Technical Specification | Validation Status |
| :--- | :--- | :--- |
| **Input Matrix ($X$)** | `['fruit', 'temperature_c', 'humidity_percent', 'light_lux', 'co2_ppm']` | 5 features (1 categorical, 4 numerical) |
| **Target Vector ($y$)** | `quality` (`Good` vs. `Bad`) | Binary categorical |
| **Class Distribution** | 4,553 Good (52.11%) vs. 4,185 Bad (47.89%) | Balanced (1.09 : 1 ratio) |
| **Splitting Strategy** | Stratified 80/20 train/test split (`random_state=42`) | **Train:** 6,990 rows (80%)<br>**Test:** 1,748 rows (20%) |
| **Categorical Encoding** | `OneHotEncoder(drop='first', handle_unknown='ignore')` | Fitted strictly on training set |
| **Feature Scaling** | `StandardScaler()` (applied within Pipeline for Logistic Regression) | Fitted strictly on training set; zero data leakage |

---

## 3. Model Benchmark & Evaluation Results

Three distinct classification architectures were implemented and benchmarked on identical training and testing partitions:
1. **Logistic Regression** (L2 Regularized with Standardized Features)
2. **Decision Tree Classifier** (Cost-Complexity / Pruned: `max_depth=8`, `min_samples_leaf=5`)
3. **Random Forest Classifier** (Ensemble of 100 Trees: `max_depth=12`, `min_samples_leaf=2`)

### 3.1 Comprehensive Metric Comparison (Held-Out Test Set: N = 1,748)

| Architecture | 5-Fold CV F1 (Macro) | Train Acc | Test Acc | Macro F1 | Good Precision | Good Recall | Good F1 | Bad Precision | Bad Recall | Bad F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $0.9502 \pm 0.0086$ | 94.61% | 95.19% | 0.9519 | 95.29% | 95.50% | 0.9539 | 95.09% | 94.86% | 0.9498 |
| **Decision Tree** | $0.9934 \pm 0.0012$ | 99.56% | 99.43% | 0.9943 | 99.78% | 99.12% | 0.9945 | 99.05% | 99.76% | 0.9940 |
| **Random Forest** ⭐ | $\mathbf{0.9976 \pm 0.0004}$ | **99.94%** | **99.94%** | **0.9994** | **100.00%** | **99.89%** | **0.9995** | **99.88%** | **100.00%** | **0.9994** |

---

### 3.2 Confusion Matrix Audit (Test Set: 911 Actual Good, 837 Actual Bad)

```text
========================================================================
1. Logistic Regression Confusion Matrix (84 Total Errors / 1,748)
------------------------------------------------------------------------
                    Predicted Good        Predicted Bad
Actual Good:             870 (TP)              41 (FN)
Actual Bad:               43 (FP)             794 (TN)

========================================================================
2. Decision Tree Confusion Matrix (10 Total Errors / 1,748)
------------------------------------------------------------------------
                    Predicted Good        Predicted Bad
Actual Good:             903 (TP)               8 (FN)
Actual Bad:                2 (FP)             835 (TN)

========================================================================
3. Random Forest Confusion Matrix (Only 1 Error / 1,748) ⭐ CHAMPION
------------------------------------------------------------------------
                    Predicted Good        Predicted Bad
Actual Good:             910 (TP)               1 (FN)
Actual Bad:                0 (FP)             837 (TN)
========================================================================
```

---

## 4. Model Selection Justification

**Champion Model:** **Random Forest Classifier**

The Random Forest architecture was selected based on a holistic evaluation of critical metrics rather than accuracy alone:
1. **Flawless Spoilage Identification (Recall on Bad = 100.00%):** In produce quality monitoring, failing to identify spoiled fruit (**False Positive / Type I error**, shipping rotten produce) is catastrophic. Random Forest achieved **0 False Positives** out of 837 actual bad test cases.
2. **Zero False Alarm Rate (Precision on Good = 100.00%):** Every single produce batch predicted as `Good` was verified to be authentic Good quality.
3. **Cross-Validation Stability:** Evaluated over 5 stratified folds on the training partition, Random Forest delivered a macro F1 of **$0.9976 \pm 0.0004$**, showing virtually zero variance and complete stability across subsamples.
4. **Generalization & Overfitting Check:** 
   - Training Accuracy: **99.94%**
   - Testing Accuracy: **99.94%**
   - The exact parity between training and testing performance confirms the model is neither underfitting nor memorizing noise.
5. **Why Tree Ensembles Outperform Linear Models:** As observed in Phase 3 EDA, fruit spoilage in cold storage is governed by non-linear interactions (e.g., tomatoes degrade sharply when light exceeds 25 Lux, while continuous 95% humidity triggers surface mold regardless of minor temperature variations). Logistic Regression is constrained by linear hyperplanes and misclassified 84 samples, whereas Random Forest captures non-linear decision splits cleanly.

---

## 5. Feature Importance & Model Interpretation

### 5.1 Random Forest Feature Importances (Mean Decrease in Impurity / Gini)

| Feature Parameter | Importance Score | Percentage Contribution | Physical & Domain Role |
| :--- | :--- | :--- | :--- |
| **`light_lux`** | **0.3926** | **39.26%** | **Primary Driver:** Sharp threshold effect; high illuminance accelerates chlorophyll decay and solanine ripening, particularly in tomatoes. |
| **`humidity_percent`** | **0.3544** | **35.44%** | **Secondary Driver:** Saturated ~95% RH fosters surface condensation and fungal germination, separating fresh from decayed fruit. |
| **`temperature_c`** | **0.1218** | **12.18%** | **Thermodynamic Factor:** Governs respiration rate and enzymatic kinetics across tested 21°C - 27°C storage ranges. |
| **`co2_ppm`** | **0.0787** | **7.87%** | **Metabolic Indicator:** Distinguishes active aerobic respiration from stagnant or anaerobic chamber conditions. |
| **`fruit_Tomato`** | **0.0300** | **3.00%** | Commodity-specific baseline adjustment for tomato respiration and light sensitivity. |
| **`fruit_Orange`** | **0.0224** | **2.24%** | Commodity-specific baseline adjustment for citrus chill resistance. |

### 5.2 Logistic Regression Coefficient Interpretation
For Logistic Regression with standardized numerical features predicting log-odds of `Good` quality:
- $\beta_{\text{light\_lux}} = \mathbf{-22.1087}$: Severe negative impact on probability of Good quality as light intensity rises.
- $\beta_{\text{humidity\_percent}} = \mathbf{-4.9383}$: Strong negative penalty on Good quality as humidity reaches near-saturation.
- $\beta_{\text{fruit\_Tomato}} = -1.4579$ and $\beta_{\text{fruit\_Orange}} = -0.5801$: Lower baseline odds relative to Banana under identical conditions.
- $\beta_{\text{co2\_ppm}} = +0.0210$: Mild positive association with active respiration.

---

## 6. Generated ML Diagnostic Visualizations

Four interactive Plotly HTML diagnostic charts were generated and saved in [backend/reports/ml_charts/](file:///d:/collegeprojects/fresh_track/backend/reports/ml_charts/):

| Visual Deliverable | File Path | Focus |
| :--- | :--- | :--- |
| **Model Comparison Chart** | [model_comparison.html](file:///d:/collegeprojects/fresh_track/backend/reports/ml_charts/model_comparison.html) | Side-by-side grouped bar chart comparing Accuracy, Precision, Recall, and F1 across all 3 classifiers. |
| **Confusion Matrices Subplots** | [confusion_matrices.html](file:///d:/collegeprojects/fresh_track/backend/reports/ml_charts/confusion_matrices.html) | 3-panel annotated heatmap displaying True Positives, False Positives, True Negatives, and False Negatives. |
| **Feature Importance Chart** | [feature_importance.html](file:///d:/collegeprojects/fresh_track/backend/reports/ml_charts/feature_importance.html) | Horizontal bar chart of MDI feature contributions highlighting Light and Humidity dominance. |
| **Classification Performance** | [classification_metrics.html](file:///d:/collegeprojects/fresh_track/backend/reports/ml_charts/classification_metrics.html) | Detailed Precision vs. Recall balance for both Good and Bad quality states. |

---

## 7. Model Storage & Artifacts

The champion model and preprocessing pipeline were serialized and saved using Joblib:
- **Primary Model Artifact:** [backend/models/freshtrack_model.joblib](file:///d:/collegeprojects/fresh_track/backend/models/freshtrack_model.joblib) (170.8 KB, zlib compressed)
- **Model Metadata File:** [backend/models/model_metadata.json](file:///d:/collegeprojects/fresh_track/backend/models/model_metadata.json)
- **Backup Mirror:** [backend/ml/saved_model/freshtrack_model.joblib](file:///d:/collegeprojects/fresh_track/backend/ml/saved_model/freshtrack_model.joblib)

The saved Joblib object is an end-to-end `Pipeline` encapsulating both the `ColumnTransformer` (categorical one-hot encoder and numerical passthrough) and the trained `RandomForestClassifier`. This allows direct inference on raw JSON dictionaries or DataFrames without external manual feature transformation.

---

## 8. Prediction Service Verification

The reusable inference service was implemented in [backend/app/services/model_prediction.py](file:///d:/collegeprojects/fresh_track/backend/app/services/model_prediction.py) and verified using independent test scripts:

### 8.1 Valid Inference Telemetry Tests

| Test Case Name | Input Telemetry | Predicted Quality | Model Confidence | Probabilities (Bad / Good) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Optimal Fresh Tomato** | Tomato, 23°C, 90% RH, 9.8 Lux, 350 ppm | **Good** | **100.00%** | `{Bad: 0.0%, Good: 100.0%}` | **PASSED** |
| **High Light Spoiled Tomato** | Tomato, 25°C, 95% RH, 150.0 Lux, 320 ppm | **Bad** | **100.00%** | `{Bad: 100.0%, Good: 0.0%}` | **PASSED** |
| **Optimal Fresh Orange** | Orange, 22°C, 94% RH, 7.0 Lux, 290 ppm | **Good** | **100.00%** | `{Bad: 0.0%, Good: 100.0%}` | **PASSED** |
| **Saturated Bad Orange** | Orange, 24°C, 95% RH, 16.0 Lux, 370 ppm | **Bad** | **100.00%** | `{Bad: 100.0%, Good: 0.0%}` | **PASSED** |
| **Fresh Banana** | Banana, 26°C, 90% RH, 16.5 Lux, 340 ppm | **Good** | **99.58%** | `{Bad: 0.42%, Good: 99.58%}` | **PASSED** |
| **Chilled Damp Bad Banana** | Banana, 24°C, 95% RH, 15.0 Lux, 250 ppm | **Bad** | **99.00%** | `{Bad: 99.0%, Good: 1.0%}` | **PASSED** |

### 8.2 Input Validation & Error Handling Tests

| Error Condition Tested | Input Vector | Service Response | Error Message Returned |
| :--- | :--- | :--- | :--- |
| **Unsupported Commodity** | Fruit: `"Mango"`, Temp: 23°C | `success: False` | *"Unsupported fruit 'Mango'. Supported commodities are: Tomato, Orange, Banana."* |
| **Negative Humidity** | Humidity: `-10.0%` | `success: False` | *"Humidity -10.0% must be between 0% and 100%."* |
| **Negative Light Intensity**| Light: `-5.0 Lux` | `success: False` | *"Light intensity -5.0 Lux cannot be negative."* |
| **Extreme Temperature** | Temp: `95.0°C` | `success: False` | *"Temperature 95.0°C is outside plausible produce storage range (-10°C to 60°C)."* |
| **Non-Numeric Value** | Temp: `"invalid_temp"` | `success: False` | *"Invalid temperature 'invalid_temp'. Must be numeric."* |

---

## 9. Important Model Limitations & Boundaries

> [!WARNING]
> **Strict Operational Boundaries of FreshTrack ML:**
> 
> 1. **Snapshot Classification vs. Longitudinal Degradation:**
>    The model predicts quality state associated with discrete environmental sensor readings. It **cannot and does not estimate shelf-life countdowns** (e.g. "fruit will spoil in 3 days") because the dataset contains zero temporal exposure timestamps or duration metrics.
> 2. **Controlled IoT Chamber Domain:**
>    The model was trained on data collected inside experimental cold storage chambers with automated environmental control. Its predictions apply to storage environments within calibrated sensor ranges (21°C - 27°C, 71% - 95% RH).
> 3. **Statistical Confidence vs. Real-World Guarantee:**
>    High model confidence (e.g. 99.8%) reflects mathematical certainty within the learned training distribution. It does not guarantee physical produce immunity against microbial contamination, mechanical bruising, or unmonitored ethylene exposure.
