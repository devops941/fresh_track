# FreshTrack 🍏📊
### Multi-Parameter Machine Learning Platform for Fruit Spoilage Prediction & Quality Intelligence

FreshTrack is an end-to-end, decoupled full-stack intelligence system designed to classify produce freshness condition (**Good** vs. **Bad**) using environmental telemetry (Storage Temperature, Relative Humidity, Light Illuminance, and CO₂ Concentration).

---

## 🏛️ Project Architecture

FreshTrack strictly enforces decoupling between the static client layer and the Python API service layer:

```text
FreshTrack/
├── frontend/                       # Static Presentation Layer (Vanilla JS + Bootstrap 5)
│   ├── index.html                  # Master Dashboard (Overview, Live Predictor, EDA Analytics)
│   ├── pages/
│   │   ├── predict.html            # Standalone Real-Time Freshness Predictor
│   │   └── analytics.html          # Standalone Telemetry & Quality Analytics Dashboard
│   ├── css/
│   │   └── style.css               # Clean modern theme & responsive layout
│   ├── js/
│   │   └── api.js                  # Centralized REST API client (single source of truth for BASE_URL)
│   └── assets/                     # Branding, icons, graphics
├── backend/                        # High-Performance Python Backend Layer
│   ├── app/
│   │   ├── main.py                 # FastAPI application, CORS middleware, safe error handlers
│   │   ├── core/
│   │   │   └── config.py           # Centralized configuration (.env, CORS, filepaths)
│   │   ├── api/                    # REST API Endpoint Routers
│   │   │   ├── router.py           # Master /api router
│   │   │   ├── health.py           # GET /api/health
│   │   │   ├── analysis.py         # GET /api/analysis/summary, /fruits, /environment
│   │   │   └── predictions.py      # POST /api/predictions/predict, GET /api/predictions/fruits
│   │   ├── schemas/                # Pydantic v2 Request & Response Validation Schemas
│   │   │   ├── health.py
│   │   │   ├── analysis.py
│   │   │   └── prediction.py
│   │   ├── services/               # Reusable Business & ML Logic
│   │   │   ├── data_cleaning.py    # Phase 2 cleaning & validation pipeline
│   │   │   ├── eda.py              # Phase 3 exploratory analytics & Plotly generation
│   │   │   ├── model_training.py   # Phase 4 model training, benchmarking & serialization
│   │   │   └── model_prediction.py # Phase 4 real-time inference service & guardrails
│   │   ├── database/               # SQLite connection & session engine
│   │   └── models/                 # Database ORM models
│   ├── models/
│   │   ├── freshtrack_model.joblib # Serialized Champion Random Forest Pipeline (99.94% accuracy)
│   │   └── model_metadata.json     # Model training metadata & test metrics
│   ├── data/
│   │   ├── original/
│   │   │   └── Dataset.csv         # Immutable Ground-Truth CSV (10,995 rows, SHA-256 verified)
│   │   └── processed/
│   │       └── freshtrack_cleaned.csv # Cleaned & normalized CSV (8,738 rows)
│   ├── reports/
│   │   ├── data_cleaning_report.md # Phase 2 cleaning report
│   │   ├── eda_report.md           # Phase 3 exploratory data analysis report
│   │   ├── model_evaluation.md     # Phase 4 machine learning evaluation report
│   │   ├── charts/                 # Phase 3 Interactive Plotly HTML Visualizations (5 charts)
│   │   └── ml_charts/              # Phase 4 ML Diagnostic Visualizations (4 charts)
│   ├── tests/
│   │   └── test_full_system.py     # 19-point end-to-end integration test suite
│   ├── requirements.txt            # Python dependencies (FastAPI, scikit-learn, pandas, etc.)
│   ├── .env.example                # Environment variable configuration template
│   └── .env                        # Active local environment configuration
├── run_backend.bat                 # One-click Windows launcher for FastAPI Uvicorn server
├── run_frontend.bat                # One-click Windows launcher for static HTTP frontend
├── README.md                       # Comprehensive system documentation
└── .gitignore                      # Standard ignores
```

---

## 🛠️ Technology Stack

| Layer | Component | Implementation |
| :--- | :--- | :--- |
| **Frontend** | UI & Components | HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons |
| **Data Viz** | Interactive Charts | Plotly.js (loaded via CDN, zero build tool overhead) |
| **API Client** | REST Communication | Vanilla JavaScript (`fetch` wrapper in [frontend/js/api.js](file:///d:/collegeprojects/fresh_track/frontend/js/api.js)) |
| **Backend API** | REST Framework | Python **FastAPI** + **Uvicorn** (Asynchronous ASGI) |
| **Data Validation** | Schema & Guardrails | **Pydantic v2** |
| **Data Processing** | Analytics & EDA | **Pandas 3.1**, **NumPy 2.5** |
| **Machine Learning** | Classifier Engine | **Scikit-learn 1.9** (Random Forest, Decision Tree, Logistic Regression) |
| **Model Storage** | Serialization | **Joblib 1.6** (Compressed end-to-end pipeline) |
| **Database** | Persistence | **SQLite** + **SQLAlchemy** |

---

## 🔒 Strict Architecture & Security Rules

1. **Complete Isolation:** `frontend/` contains only static client assets (HTML/CSS/JS). No Python code exists in the frontend, and no frontend code resides in `backend/app/`.
2. **REST-Only Communication:** All data flows through standard HTTP REST endpoints (`/api/*`).
3. **Single API Configuration Point:** All routing originates from `API_CONFIG.BASE_URL` in [frontend/js/api.js](file:///d:/collegeprojects/fresh_track/frontend/js/api.js).
4. **Data Immutability:** The raw dataset [backend/data/original/Dataset.csv](file:///d:/collegeprojects/fresh_track/backend/data/original/Dataset.csv) is immutable (verified SHA-256: `218F76F061A12B0A38C55107E5BE3B377E9EECB33F1A1840EF2B4E4618A5E5AA`).
5. **Zero Secrets in Frontend:** All environment variables reside strictly in `backend/.env`.
6. **No Node.js / React Runtime:** The frontend is 100% lightweight Vanilla JavaScript with zero npm dependencies.

---

## 🚀 How to Run Frontend and Backend Separately

### Option 1: One-Click Windows Launchers (Recommended)

1. **Launch the Backend API Server:**
   - Double-click `run_backend.bat` (or run in terminal: `.\run_backend.bat`).
   - Starts FastAPI on [http://127.0.0.1:8000](http://127.0.0.1:8000).
   - Interactive Swagger API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
2. **Launch the Frontend Client:**
   - Double-click `run_frontend.bat` (or run in terminal: `.\run_frontend.bat`).
   - Serves the frontend on [http://127.0.0.1:5500](http://127.0.0.1:5500).
   - Open browser at [http://127.0.0.1:5500/index.html](http://127.0.0.1:5500/index.html).

---

### Option 2: Manual Terminal Commands

#### Terminal 1: Backend
```powershell
cd d:\collegeprojects\fresh_track\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Terminal 2: Frontend
```powershell
cd d:\collegeprojects\fresh_track\frontend
python -m http.server 5500
```
Open browser at: [http://127.0.0.1:5500/index.html](http://127.0.0.1:5500/index.html).

---

## 📡 REST API Reference

| Method | Endpoint | Description | Sample Response / Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Health check & model readiness | `{"status": "healthy", "model_availability": true}` |
| `GET` | `/api/analysis/summary` | Cleaned dataset summary metrics | `{"total_records": 8738, "good_quality_percentage": 52.11}` |
| `GET` | `/api/analysis/fruits` | Fruit-wise quality cross-tabulations | `{"fruits": {"Tomato": {...}, "Orange": {...}, "Banana": {...}}}` |
| `GET` | `/api/analysis/environment`| Mean/median environmental statistics | `{"overall_by_quality": {"Good": {...}, "Bad": {...}}}` |
| `GET` | `/api/predictions/fruits` | Supported fruit commodities | `{"supported_fruits": ["Tomato", "Orange", "Banana"]}` |
| `POST`| `/api/predictions/predict` | Real-time ML freshness inference | `{"predicted_quality": "Good", "good_probability": 0.998}` |

---

## 🧪 Comprehensive Integration Test Suite

To run the complete 19-point system audit covering dataset integrity, EDA services, ML model inference, input validation, and API contracts:

```powershell
python backend/tests/test_full_system.py
```

Expected output:
```text
=================================================================
FRESH TRACK PHASE 7: COMPLETE SYSTEM INTEGRATION AUDIT
=================================================================
[PASS] 1. Original Dataset Immutability
[PASS] 2. Cleaned Dataset Validation
[PASS] 3. Phase 3 EDA Metric Computations
[PASS] 4. Phase 3 Interactive Plotly Charts Verification
[PASS] 5. Serialized Random Forest Model File Verification
[PASS] 6. ML Prediction: Optimal Fresh Tomato
[PASS] 7. ML Prediction: High-Light Degraded Tomato
[PASS] 8. Phase 4 ML Diagnostic Visualizations Verification
[PASS] 9. Input Guardrail: Unsupported Fruit Rejection
[PASS] 10. Input Guardrail: Negative Humidity Rejection
[PASS] 11. REST API: GET /api/health Handler
[PASS] 12. REST API: GET /api/analysis/summary Handler
[PASS] 13. REST API: GET /api/analysis/fruits Handler
[PASS] 14. REST API: GET /api/analysis/environment Handler
[PASS] 15. REST API: GET /api/predictions/fruits Handler
[PASS] 16. REST API: POST /api/predictions/predict Handler
[PASS] 17. Frontend Interface & Client Files Verification
[PASS] 18. Architecture Boundary: Zero Python Code in Frontend
[PASS] 19. Architecture Boundary: Zero HTML/JS Code in Backend app/
-----------------------------------------------------------------
AUDIT SUMMARY: 19 PASSED, 0 FAILED (100% Success)
=================================================================
```

---

## ⚠️ Important Dataset & Model Limitations

> [!IMPORTANT]
> - **Snapshot Sensor Telemetry:** The dataset and model evaluate snapshot environmental storage conditions (`temperature_c`, `humidity_percent`, `light_lux`, `co2_ppm`) paired with a categorical quality label (`Good` vs. `Bad`).
> - **No Temporal Duration in Data:** The dataset contains zero timestamps, storage durations, or elapsed shelf-life days. The model **does not estimate remaining days until expiry** or predict future decay rates.
> - **Statistical Confidence:** Model confidence reflects mathematical likelihood within calibrated chamber ranges (21°C–27°C, 71%–95% RH) and does not represent an absolute biological guarantee against unmonitored microbial contamination or physical bruising.
