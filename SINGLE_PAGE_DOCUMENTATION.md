# FRESHTRACK AI
### *Multi-Parameter Machine Learning Platform for Fruit Spoilage Prediction & Storage Quality Intelligence*

---

### PROJECT OVERVIEW
FreshTrack is an intelligent machine learning web platform designed to evaluate and predict fruit freshness conditions (Good vs. Bad) across agricultural supply chains and cold storage warehouses. Leveraging multi-parameter environmental telemetry (Temperature, Relative Humidity, Light Illuminance, and CO₂ concentration), it provides predictive spoilage alerts, optimal climate guidance, and data-driven quality certification to minimize post-harvest food waste and ensure supply chain safety.

---

### PROBLEM STATEMENT
Traditional post-harvest fruit storage and cold-chain logistics rely heavily on manual sampling and fixed-temperature refrigeration. High post-harvest spoilage rates (up to 30–40%) occur due to undetected micro-climatic fluctuations, improper humidity balance, and gas accumulation without real-time multi-sensor telemetry, leading to substantial financial losses and food degradation.

---

### PROJECT OBJECTIVES
* Real-time automated classification of fruit freshness status (Good / Fresh vs. Bad / Degraded).
* Continuous multi-parameter telemetry monitoring (Temperature, Humidity, Light, and CO₂).
* Multi-commodity risk benchmarking across diverse produce commodities (Tomato, Banana, Orange).
* Actionable climate optimization recommendations and cold-storage corrective checklists.
* Automated digital quality audit certificate generation for traceability and trade compliance.

---

### KEY MODULES & FEATURES
* **Telemetry Analyzer & Live Predictor:** Interactive sensor input interface with real-time ML inference gauge, safety risk index, and actionable corrective checklists.
* **Warehouse Quality Dashboard:** High-level operational overview featuring active batch cards, quality distribution KPIs, and chamber health metrics.
* **Environmental & Exploratory Analytics:** Interactive Pearson correlation matrix, telemetry regression curves, and Pandas parametric summaries across 8,738 clean observations.
* **Multi-Fruit Comparison Engine:** Comparative benchmark visualizer analyzing climate vulnerability, spoilage risk curves, and sample volume across commodities.
* **Audit Trail & Inspection History:** Filterable telemetry archive tracking historical batches with timestamped environmental parameters.
* **Official PDF Audit Certification:** Client-side 1-click official cold storage quality audit certificate generator with QR-coded verification badges via jsPDF.
* **Decoupled RESTful API Engine:** High-performance asynchronous FastAPI backend with strict Pydantic v2 validation and automated OpenAPI/Swagger documentation.

---

### TECHNOLOGY STACK
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+), Bootstrap 5.3, Plotly.js / Canvas Visualizations, jsPDF
* **Backend:** Python 3.12, FastAPI, Uvicorn (Asynchronous ASGI Server)
* **Machine Learning & Data Processing:** Scikit-learn (Random Forest Classifier — 99.9% accuracy, Decision Tree, Logistic Regression), Pandas, NumPy, Joblib
* **Data Validation & Guardrails:** Pydantic v2 (Strict request/response type validation)
* **Persistence & Datasets:** SQLite, SQLAlchemy ORM, Ground-Truth CSV Dataset (10,995 observations)
* **Version Control & Repository:** Git, GitHub

---

### PROJECT LINKS
* **GitHub Repository:** [https://github.com/devops941/fresh_track](https://github.com/devops941/fresh_track)
* **Live Application URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Domain:** AgriTech | Cold-Chain Logistics | Artificial Intelligence | Food Safety
* **Application Type:** Decoupled Full-Stack Machine Learning Web Application
