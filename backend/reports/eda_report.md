# FreshTrack – Phase 3: Exploratory Data Analysis (EDA) & Visualization Report

**Project:** FreshTrack – Fruit Freshness & Quality Intelligence  
**Phase:** Phase 3: Data Analysis & Visualization  
**Dataset Analyzed:** [backend/data/processed/freshtrack_cleaned.csv](file:///d:/collegeprojects/fresh_track/backend/data/processed/freshtrack_cleaned.csv)  
**EDA Service Script:** [backend/app/services/eda.py](file:///d:/collegeprojects/fresh_track/backend/app/services/eda.py)  
**Interactive Charts Directory:** [backend/reports/charts/](file:///d:/collegeprojects/fresh_track/backend/reports/charts/)  

---

## 1. Overall Dataset Summary

The cleaned FreshTrack dataset comprises **8,738 ground-truth observations** collected from an IoT-enabled cold storage monitoring chamber across three target fruit commodities:

| Metric | Empirical Value |
| :--- | :--- |
| **Total Cleaned Observations** | **8,738** |
| **Good Quality Observations** | **4,553** |
| **Bad Quality Observations** | **4,185** |
| **Overall Good Quality Percentage** | **52.11%** |
| **Overall Bad Quality Percentage** | **47.89%** |
| **Target Label Balance Ratio** | **1.09 : 1** (Practically balanced, no class imbalance risk) |

### Fruit-Wise Record Distribution
- **🍅 Tomato:** 3,741 records (42.81% of dataset)
- **🍊 Orange:** 3,330 records (38.11% of dataset)
- **🍌 Banana:** 1,667 records (19.08% of dataset)

---

## 2. Fruit-Wise Quality Analysis

Cross-tabulation of fruit commodities against normalized quality condition:

| Fruit Commodity | Total Records | Good Quality | Bad Quality | Good % | Bad % | Dominant State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tomato** | 3,741 | 1,916 | 1,825 | 51.22% | 48.78% | Balanced (51.2% Good) |
| **Orange** | 3,330 | 1,681 | 1,649 | 50.48% | 49.52% | Balanced (50.5% Good) |
| **Banana** | 1,667 | 956 | 711 | 57.35% | 42.65% | Skewed Good (57.4% Good) |
| **All Fruits** | **8,738** | **4,553** | **4,185** | **52.11%** | **47.89%** | **Near-Symmetric** |

### Key Observations:
1. **High Consistency across Citrus and Solanaceous Produce:** Both Tomato and Orange exhibit near parity (~50.5% - 51.2% Good), indicating identical experimental exposure durations between fresh and degraded conditions during data collection.
2. **Banana Resilience in Storage Trials:** Bananas show a higher proportion of Good state (57.35%), likely reflecting the narrower temperature band tested (24°C - 26°C) compared to tomatoes (22°C - 27°C).

---

## 3. Environmental Condition Analysis vs. Quality

### 3.1 Aggregate Parameter Comparison (Good vs. Bad)

| Environmental Feature | Quality | Mean | Median | Std Dev | Min | Max | Statistical Signal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Temperature (°C)** | **Good** | 23.49 | 23.00 | 1.42 | 21.00 | 27.00 | Slightly cooler median (23°C) |
| | **Bad** | 24.16 | 24.00 | 0.93 | 23.00 | 26.00 | Tightly bounded higher mean (24.16°C) |
| **Relative Humidity (%)** | **Good** | 91.93 | 93.00 | 3.53 | 71.00 | 95.00 | Wider spread (71% - 95%) |
| | **Bad** | **94.96** | **95.00** | **0.64** | 80.00 | 95.00 | **Extreme saturation cluster (~95% RH)** |
| **Light Intensity (Lux)** | **Good** | 10.33 | 8.52 | 4.30 | 4.22 | 23.53 | Low illuminance (< 24 Lux) |
| | **Bad** | **47.83** | 15.77 | 72.05 | 10.22 | **268.45** | **High light exposure threshold** |
| **CO₂ Concentration (ppm)**| **Good** | 319.31 | 332.00 | 63.64 | 37.00 | 478.00 | Higher respiration baseline |
| | **Bad** | 309.23 | 304.00 | 58.84 | 20.00 | 477.00 | Lower median (304 ppm) |

---

### 3.2 Granular Analysis: Parameter Breakdown by Fruit and Quality

| Fruit Commodity | Quality | Mean Temp (°C) | Median Temp | Mean Humid (%) | Median Humid | Mean Light (Lux) | Median Light | Mean CO₂ (ppm) | Median CO₂ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Banana** | **Good** | 25.40 | 26.00 | 90.24 | 90.00 | 16.89 | 16.89 | 340.22 | 335.00 |
| | **Bad** | 24.08 | 24.00 | **94.97** | 95.00 | 15.36 | 16.29 | 254.00 | 253.00 |
| **Orange** | **Good** | 22.33 | 22.00 | 94.77 | 95.00 | 6.98 | 7.00 | 282.92 | 297.00 |
| | **Bad** | 23.35 | 23.00 | **94.95** | 95.00 | 16.16 | 15.34 | 318.52 | 347.00 |
| **Tomato** | **Good** | 23.57 | 23.00 | 90.28 | 90.00 | 10.00 | 9.81 | 340.80 | 350.00 |
| | **Bad** | 24.94 | 25.00 | **94.96** | 95.00 | **89.10** | 16.03 | 322.35 | 312.00 |

---

## 4. Statistical Correlation & Empirical Pattern Analysis

### 4.1 Pearson Correlation Matrix

For numerical correlation analysis, `quality` was mapped to binary representation ($1 = \text{Good}, 0 = \text{Bad}$):

| Feature | `temperature_c` | `humidity_percent` | `light_lux` | `co2_ppm` | `quality_binary` (Good=1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`temperature_c`** | 1.000 | -0.129 | 0.423 | 0.206 | **-0.268** |
| **`humidity_percent`** | -0.129 | 1.000 | 0.157 | -0.266 | **-0.505** |
| **`light_lux`** | 0.423 | 0.157 | 1.000 | -0.076 | **-0.351** |
| **`co2_ppm`** | 0.206 | -0.266 | -0.076 | 1.000 | **+0.082** |
| **`quality_binary`** | **-0.268** | **-0.505** | **-0.351** | **+0.082** | 1.000 |

### 4.2 Interpreting Identified Patterns (Without Claiming Causation)

1. **Negative Association between Saturated Humidity and Quality ($r = -0.505$):**
   - Relative humidity displays the strongest correlation with spoilage. 
   - Across all three commodities, the `Bad` class has a standard deviation of only **0.64%** centered tightly on **94.96% RH**.
   - In horticultural storage physics, saturated humidity (> 94% RH) promotes water condensation on fruit skins, creating a micro-environment conducive to fungal hyphae growth and surface rot. Conversely, the `Good` quality class operates across a broader, less saturated range (mean 91.93%, standard deviation 3.53%).
2. **Illuminance Threshold Effect ($r = -0.351$):**
   - All `Good` quality produce is confined to low light intensity ($\le 23.53\text{ Lux}$, mean $10.33\text{ Lux}$).
   - In contrast, elevated light readings ($\ge 25\text{ Lux}$ up to $268.45\text{ Lux}$) occur exclusively in the `Bad` category, particularly in tomatoes. Exposure to light stimulates metabolic ripening enzymes and chlorophyll breakdown in post-harvest tomatoes.
3. **Mild Negative Association with Storage Temperature ($r = -0.268$):**
   - Higher ambient temperatures generally correlate with increased respiration and accelerated senescence. For tomatoes and oranges, average temperatures in the `Bad` class were 1.0°C to 1.4°C higher than in the `Good` class.
4. **Weak Direct Linear Correlation with CO₂ ($r = +0.082$):**
   - CO₂ alone does not exhibit a strong monotonic linear relationship with quality. Instead, it displays fruit-specific dynamics: Good tomatoes and bananas had higher CO₂ levels (indicating healthy active respiration without anaerobic decay), while Good oranges had slightly lower CO₂.

---

## 5. Generated Interactive Plotly Visualizations

The EDA service [backend/app/services/eda.py](file:///d:/collegeprojects/fresh_track/backend/app/services/eda.py) generated five interactive Plotly HTML charts in [backend/reports/charts/](file:///d:/collegeprojects/fresh_track/backend/reports/charts/):

| Chart File | Visualization Type | Focus & Purpose |
| :--- | :--- | :--- |
| [good_vs_bad_donut.html](file:///d:/collegeprojects/fresh_track/backend/reports/charts/good_vs_bad_donut.html) | Interactive Donut Chart | Proportional breakdown of overall quality: **52.1% Good (4,553)** vs. **47.9% Bad (4,185)** with hover tooltips. |
| [fruit_wise_quality_bar.html](file:///d:/collegeprojects/fresh_track/backend/reports/charts/fruit_wise_quality_bar.html) | Grouped Bar Chart | Direct comparison of Good vs. Bad counts across Tomato, Orange, and Banana. |
| [environmental_conditions_comparison.html](file:///d:/collegeprojects/fresh_track/backend/reports/charts/environmental_conditions_comparison.html) | 4-Panel Box Plot Subplots | Distribution, medians, and outlier ranges for Temperature, Humidity, Light, and CO₂ stratified by Quality. |
| [correlation_heatmap.html](file:///d:/collegeprojects/fresh_track/backend/reports/charts/correlation_heatmap.html) | Annotated Heatmap | Pearson correlation matrix visualizing linear relationships between sensors and quality state. |
| [fruit_comparison_chart.html](file:///d:/collegeprojects/fresh_track/backend/reports/charts/fruit_comparison_chart.html) | Multi-Metric Grouped Bar | Subplot comparisons of mean environmental fingerprints for each fruit commodity under Good and Bad states. |

---

## 6. Important Dataset Limitations & Scope Clarification

> [!WARNING]
> **Strict Boundaries of the FreshTrack Dataset:**
> 
> The dataset represents discrete **environmental snapshot telemetry** matched with binary quality states (`Good` vs. `Bad`).
> 
> **The dataset explicitly lacks:**
> - Chronological timestamps or observation calendar dates.
> - Duration of storage exposure (e.g., hours or days fruit spent in the chamber).
> - Daily progression or rate of degradation.
> - Actual spoilage transition timestamps.
> - Shelf-life countdown metrics (e.g., "days until decay").
> 
> **Architectural Adherence:**
> - The observed associations must be interpreted as **correlations in stationary environmental states**, not causal biological progression.
> - FreshTrack will not fabricate synthetic shelf-life durations or claim predictive time-to-decay capabilities that cannot be empirically grounded.
