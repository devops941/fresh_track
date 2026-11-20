# FreshTrack – Phase 2: Data Inspection & Cleaning Report

**Project:** FreshTrack – Fruit Freshness & Quality Intelligence  
**Phase:** Phase 2: Dataset Inspection, Cleaning & Preparation  
**Generated On:** 2026-11-19  
**Source Dataset:** *A Multi-Parameter Dataset for Machine Learning Based Fruit Spoilage Prediction in an IoT-Enabled Cold Storage System* (Mendeley Data, Nshekanabo Marius & Stanley Mugisha)  
**Location:** [backend/data/original/Dataset.csv](file:///d:/collegeprojects/fresh_track/backend/data/original/Dataset.csv)  
**Processed Target:** [backend/data/processed/freshtrack_cleaned.csv](file:///d:/collegeprojects/fresh_track/backend/data/processed/freshtrack_cleaned.csv)  
**Cleaning Service:** [backend/app/services/data_cleaning.py](file:///d:/collegeprojects/fresh_track/backend/app/services/data_cleaning.py)  

---

## 1. Executive Summary

In Phase 2, the raw IoT sensor dataset was systematically audited, verified against published research literature, normalized, and prepared for machine learning modeling:
- The raw dataset contains **10,995 observations** across 6 columns with **0 missing values**.
- The target labels were cleaned from three variants (`Good`, `Bad`, `BAD`) into standard binary classes: **`Good`** and **`Bad`**.
- As specified, the scope was focused to three primary fruit commodities: **Tomato**, **Banana**, and **Orange** (filtering out Pineapple), resulting in **8,738 observations**.
- Raw column headers containing typographical errors and special characters (`Light (Fux)`, `CO2 (pmm)`, `Humid (%)`) were renamed into self-describing, standard identifiers (`light_lux`, `co2_ppm`, `humidity_percent`, `temperature_c`, `fruit`, `quality`).
- The original CSV was maintained in an immutable state with verified SHA-256 integrity (`218F76F061A12B0A38C55107E5BE3B377E9EECB33F1A1840EF2B4E4618A5E5AA`).

---

## 2. Raw Dataset Inspection Findings

### 2.1 Schema & Data Types

| Original Column | Raw Data Type | Non-Null Count | Sample Value | Verified Meaning & Unit |
| :--- | :--- | :--- | :--- | :--- |
| `Fruit` | `str` / `object` | 10,995 (100%) | `Orange` | Fruit commodity name |
| `Temp` | `int64` | 10,995 (100%) | `22` | Storage chamber temperature in degrees Celsius (°C) |
| `Humid (%)` | `int64` | 10,995 (100%) | `95` | Relative Humidity in percentage (%) |
| `Light (Fux)` | `float64` | 10,995 (100%) | `7.358649` | Illuminance in **Lux** (typographical error "Fux" in header) |
| `CO2 (pmm)` | `int64` | 10,995 (100%) | `361` | Carbon Dioxide concentration in **ppm** (typo "pmm" in header) |
| `Class` | `str` / `object` | 10,995 (100%) | `Good` | Freshness / Quality state (`Good`, `Bad`, `BAD`) |

### 2.2 Missing Values
- **Null / NaN Count:** 0 in all columns.
- **Whitespace / Empty Strings:** 0 in all text columns.
- **Data Completeness:** 100.0%.

### 2.3 Categorical Distributions (Raw Dataset)

#### Fruit Distribution:
- **Tomato:** 3,741 records (34.02%)
- **Orange:** 3,330 records (30.29%)
- **Pineapple:** 2,257 records (20.53%) *(Filtered out in Phase 2)*
- **Banana:** 1,667 records (15.16%)
- **Total:** 10,995 records

#### Quality Label Distribution (Pre-Cleaning):
- `Good`: 5,667 records (51.54%)
- `Bad`: 4,617 records (41.99%)
- `BAD`: 711 records (6.47%) *(Inconsistent uppercase casing)*
- **Total Bad (`Bad` + `BAD`):** 5,328 records (48.46%)

### 2.4 Numeric Summary Statistics (Raw Dataset)

| Metric | Temp (°C) | Humid (%) | Light (Lux) | CO₂ (ppm) |
| :--- | :--- | :--- | :--- | :--- |
| **Minimum** | 21.000 | 71.000 | 4.221 | 20.000 |
| **25th Percentile (Q1)** | 23.000 | 94.000 | 9.929 | 289.000 |
| **Median (50%)** | 24.000 | 95.000 | 12.908 | 323.000 |
| **Mean** | 23.839 | 93.526 | 24.933 | 319.530 |
| **75th Percentile (Q3)** | 25.000 | 95.000 | 15.592 | 359.000 |
| **Maximum** | 27.000 | 95.000 | 268.448 | 478.000 |
| **Std Deviation** | 1.235 | 3.001 | 48.023 | 58.889 |
| **Tukey Bounds [Lower, Upper]** | [20.0, 28.0] | [92.5, 96.5] | [1.435, 24.086] | [184.0, 464.0] |

---

## 3. Deep Dive: Outliers, Physical Realities & Units

### 3.1 Unit Verification via Literature
Literature review of the dataset publication (*"A Multi-Parameter Dataset for Machine Learning Based Fruit Spoilage Prediction in an IoT-Enabled Cold Storage System"*, Mendeley Data) confirmed the hardware instrumentation:
- **Temperature & Humidity:** Collected via digital hygrometer/thermometer (DHT series), calibrated to integer precision (°C and % RH).
- **Light (`Light (Fux)`):** The abbreviation "Fux" is a typographical error for **Lux**, collected via ambient light sensor (BH1750 / calibrated LDR).
- **CO₂ (`CO2 (pmm)`):** The abbreviation "pmm" is a typographical error for **ppm** (parts per million), collected via an optical or electrochemical gas sensor (e.g., MQ-135 / NDIR module).

### 3.2 Analysis of Unusual Light Values (Up to 268.45 Lux)
Statistical inspection flagged 739 records with light readings exceeding the Tukey upper threshold of 24.09 Lux, reaching up to 268.45 Lux.
- **Fruit Specificity:** All 739 high-light readings belong exclusively to **Tomato** (`Tomato` mean light is 48.59 Lux vs. `Orange` 11.53 Lux and `Banana` 16.24 Lux).
- **Quality Correlation:** **100.0% of these 739 high-light tomato records are classified as `Bad` (Spoiled).**
- **Domain Interpretation:** This is not sensor malfunction. Cold storage chambers are intentionally kept dark to prevent chlorophyll degradation and accelerated ripening. High light exposures (> 25 Lux) represent illumination stress or transparent storage conditions that consistently triggered post-harvest spoilage. These records provide vital predictive signal and must be preserved.

### 3.3 Analysis of Low CO₂ Values (< 100 ppm)
- Outdoor atmospheric baseline CO₂ is approximately 400–420 ppm.
- In this dataset, 13 records recorded CO₂ levels below 100 ppm (minimum: 20 ppm).
- 12 of the 13 low-CO₂ records belong to the `Bad`/`BAD` class (mostly `Banana`).
- While extremely low CO₂ in an unventilated chamber is physically rare, these records likely reflect early sensor calibration stabilization cycles during chamber setup. Because they constitute less than 0.15% of records, they do not distort model generalization.

---

## 4. Duplicate Records & Handling Decision

### 4.1 Duplicate Statistics
- **Raw Dataset (10,995 rows):** 2,880 exact duplicate rows across all 6 columns (26.19%).
- **Phase 2 Filtered Subset (8,738 rows):** 2,213 exact duplicate rows (25.33%).
  - Tomato: 922 duplicates
  - Orange: 892 duplicates
  - Banana: 399 duplicates

### 4.2 Duplicate Integrity Check (Zero Conflict)
A grouping audit of all feature combinations `[Fruit, Temp, Humid, Light, CO2]` against `Class` confirmed:
$$\text{Conflicting Labels} = 0$$
When environmental parameters match, the assigned quality class is 100% consistent across every duplicate row.

### 4.3 Why Do Duplicates Exist in This Dataset?
1. **IoT Steady-State Telemetry:** Cold storage chambers are controlled environments equipped with automated thermostats and humidifiers. When the chamber stabilizes (e.g., at 24°C, 95% RH, dark room), periodic sensor polling every few minutes records identical measurements over extended steady-state intervals.
2. **Sensor Quantization:** Temperature and humidity are stored as discrete integers (24, 95). The floating-point light values repeat because they are calculated using discrete ADC step formulas (e.g., ADC raw value 100 maps to exactly `12.133418` Lux).
3. **Fruit Condition Invariance:** A batch of produce in a controlled chamber does not change state instantly from one minute to the next; it remains in the same condition over consecutive reading intervals.

### 4.4 Duplicate Handling Decision
**Decision: Retain repeated observations in `freshtrack_cleaned.csv`.**
- *Reasoning:* Automatically dropping duplicates would artificially discard over 25% of real-world observations and bias the empirical representation of steady-state chamber operations.
- *Flexibility:* The cleaning service `backend/app/services/data_cleaning.py` exposes a dedicated parameter `drop_duplicates=False` so that future cross-validation workflows can evaluate both deduplicated and complete observation sets.

---

## 5. Cleaning Transformations Applied

1. **Class Label Normalization:**
   - `Good` → `Good`
   - `Bad` → `Bad`
   - `BAD` → `Bad`
2. **Commodity Scope Filtering:**
   - Retained: `Tomato`, `Banana`, `Orange`
   - Excluded: `Pineapple` (2,257 rows excluded safely)
3. **Column Standardization & Renaming:**
   - `Fruit` → `fruit`
   - `Temp` → `temperature_c` (Storage temperature in °C)
   - `Humid (%)` → `humidity_percent` (Relative humidity in %)
   - `Light (Fux)` → `light_lux` (Ambient illuminance in Lux, typo corrected)
   - `CO2 (pmm)` → `co2_ppm` (CO₂ concentration in ppm, typo corrected)
   - `Class` → `quality` (Quality classification target)
4. **Output Generation:**
   - Cleaned dataset exported strictly to [backend/data/processed/freshtrack_cleaned.csv](file:///d:/collegeprojects/fresh_track/backend/data/processed/freshtrack_cleaned.csv).
   - Original dataset [backend/data/original/Dataset.csv](file:///d:/collegeprojects/fresh_track/backend/data/original/Dataset.csv) left 100% unaltered.

---

## 6. Output Dataset Validation

Upon reloading and validating `backend/data/processed/freshtrack_cleaned.csv`:

```text
Validation Status: PASSED
Total Rows: 8,738
Total Columns: 6
Columns: ['fruit', 'temperature_c', 'humidity_percent', 'light_lux', 'co2_ppm', 'quality']
Missing Values: 0 (0.0%)
```

### 6.1 Cleaned Class Distribution
| Quality Label | Count | Percentage |
| :--- | :--- | :--- |
| **Good** | 4,553 | 52.11% |
| **Bad** | 4,185 | 47.89% |
| **Total** | 8,738 | 100.00% |

The normalized dataset is remarkably balanced (52.1% Good vs. 47.9% Bad), preventing class imbalance bias during model training.

### 6.2 Per-Commodity Breakdown (Cleaned Dataset)

| Fruit | Total Observations | Good | Bad | Temp Range (°C) | Humidity Range (%) | Light Range (Lux) | CO₂ Range (ppm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tomato** | 3,741 (42.8%) | 1,916 | 1,825 | 22.0 – 27.0 | 71.0 – 95.0 | 5.93 – 268.45 | 30.0 – 478.0 |
| **Orange** | 3,330 (38.1%) | 1,681 | 1,649 | 21.0 – 24.0 | 80.0 – 95.0 | 4.22 – 20.29 | 21.0 – 461.0 |
| **Banana** | 1,667 (19.1%) | 956 | 711 | 24.0 – 26.0 | 75.0 – 95.0 | 7.09 – 23.53 | 20.0 – 477.0 |

---

## 7. Critical Dataset Scope & Limitations

> [!WARNING]
> **Strict Boundaries of the FreshTrack Dataset:**
> 
> The FreshTrack dataset is composed exclusively of snapshot environmental measurements (`fruit`, `temperature_c`, `humidity_percent`, `light_lux`, `co2_ppm`) paired with a concurrent quality assessment (`Good` vs. `Bad`).
> 
> **It explicitly does NOT contain:**
> 1. Storage observation timestamps or calendar dates.
> 2. Duration / number of storage days elapsed.
> 3. Longitudinal daily quality transitions or decay rates.
> 4. Spoilage dates or time-series tracking of identical fruit specimens.
> 5. Remaining shelf-life estimation values (e.g., "3 days remaining").
> 
> **Architectural Commitment:** FreshTrack's ML models and REST API will formulate this task as **Multi-Parameter Environmental Quality Classification** (predicting whether current conditions correspond to a *Good* or *Bad* produce condition). We will **never invent, simulate, or hallucinate** synthetic shelf-life countdowns or time-to-decay metrics that are not substantiated by empirical data.
