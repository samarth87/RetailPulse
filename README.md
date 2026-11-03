# RetailPulse: AI-Powered Customer Analytics & Demand Forecasting Platform

**Python:** 3.10 / 3.11 / 3.12 | **Framework:** Streamlit 1.49+ | **ML:** Scikit-Learn, Prophet (optional), XGBoost (optional) | **Container:** Docker | **License:** MIT

RetailPulse is an enterprise-grade AI analytics and decision-intelligence platform tailored for retail, e-commerce, and multi-channel commerce businesses. Built around the industry-standard UCI Machine Learning Repository / Kaggle "Online Retail II" transaction schema, RetailPulse bridges raw transactional data with actionable operations, including customer lifetime value optimization, churn prevention, demand forecasting, dynamic inventory replenishment, and production MLOps drift monitoring.

## Table of Contents

• Executive Overview
• System Architecture (5-Layer MLOps Pipeline)
• How the Platform Works: Deep Dive into Modules
  - 1. Executive Summary & Architecture
  - 2. Data Hub & Automated EDA
  - 3. Sales, Product & Regional Intelligence
  - 4. Customer Segmentation (RFM + Unsupervised ML)
  - 5. Hybrid Demand Forecasting Engine
  - 6. Customer Churn Prediction & Explainability
  - 7. Inventory Optimization & What-If Simulator
  - 8. MLOps Monitoring, Data Drift & Model Registry
• Data Pipeline & Schema Specifications
• Getting Started & Installation
  - Prerequisites
  - Local Installation
  - Running with Docker
• Configuration & Customization
• Project Directory Structure
• Technology Stack
• Troubleshooting & FAQs
• License & Acknowledgements

## Executive Overview

Retail organizations face common hurdles: disconnected spreadsheets, unexpected stock-outs, capital trapped in slow-moving inventory, customer attrition caught too late, and lack of visibility into demand trends. 

RetailPulse unifies these disparate workflows into a single high-performance dashboard:
• Multi-Stream Data Ingestion: Pre-bundled with an authentic curated Kaggle "Online Retail II" dataset (`kaggle_online_retail_sample.csv`), a high-volume synthetic enterprise feed, and drag-and-drop support for custom CSV and Excel files.
• Instant Data Sanitization: Standardizes column aliases, strips 'C' cancellations, filters non-product ledger codes, handles missing customer identifiers, and normalizes financial numbers.
• One-Click Dataset Export: Download the curated Kaggle sample directly from the sidebar to inspect in spreadsheet software or test the custom uploader.
• Unsupervised Segmentation & Actionable Playbooks: Combines RFM scoring with K-Means and DBSCAN clustering, mapping cohorts into defined business personas (Champions, Loyalists, Potential Loyalists, At-Risk, Lost).
• Hybrid Demand Forecasting: Blends additive decomposition (Prophet or Fourier-Ridge fallback) with autoregressive Gradient Boosting to deliver daily/weekly forecasts, scenario testing, and confidence intervals.
• Predictive Churn Engine: Time-split machine learning (Random Forest or XGBoost) to score individual customer churn risks before they lapse, accompanied by SHAP-style permutation feature importance.
• Scientific Inventory Planning: Combines classical inventory formulas (Safety Stock, Reorder Point, EOQ) with dynamic ML forecast adjustments to flag immediate reorders and capital overstock.
• Enterprise MLOps Guardrails: Real-time two-sample Kolmogorov-Smirnov (KS) data drift tests with simulated covariate shift and an MLflow-style Model Registry.

## System Architecture (5-Layer MLOps Pipeline)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RETAILPULSE ARCHITECTURE                        │
└────────────────────────────────────────────────────────────────────────┘

 [1. Data Layer]
   │  • Kaggle / UCI "Online Retail II" CSV/XLSX Upload OR Synthetic Generator
   │  • Canonical cleaning: Strips 'C' cancellations, non-product codes (POST, M, D)
   │  • Type coercion, deduplication, and automated category regex mapping
   ▼
 [2. Feature Engineering]
   │  • Customer RFM metrics: Recency, Frequency, Monetary, Tenure, AOV, Percentiles
   │  • Time Series: Lags (t-1, t-7, t-14), Rolling stats (7d/30d mean & std), Cyclical Fourier
   │  • Churn Features: Inter-purchase gap statistics, 90-day order velocity, basket breadth
   ▼
 [3. Model Layer]
   │  • Unsupervised: K-Means (K=2..8), DBSCAN (eps, min_samples), PCA 2D/3D
   │  • Forecasting: Facebook Prophet + Autoregressive Gradient Boosting Regressor (GBM)
   │  • Classification: Random Forest / XGBoost with class balancing & permutation importance
   ▼
 [4. Serving & Simulation Layer]
   │  • Interactive Streamlit Frontend (Dark Theme: #0B1220 / Accent: #00D1B2)
   │  • What-if promotional demand shock simulator (-50% to +100%)
   │  • Inventory control simulator (Z-score safety stock, ROP, EOQ, holding costs)
   ▼
 [5. MLOps & Governance]
   │  • Two-Sample Kolmogorov-Smirnov (KS) data drift detection (p-value < alpha)
   │  • Longitudinal weekly drift tracking & empirical CDF divergence
   │  • MLflow-style Model Registry (Archived / Staging / Production) & live run logging
```

## How the Platform Works: Deep Dive into Modules

### 1. Executive Summary & Architecture
• Prior-Period Comparative KPIs: Dynamically computes Total Revenue, Estimated Profit, Total Orders, Active Customers, and Average Order Value (AOV). Compares current filter window against an immediately preceding period of identical duration, displaying exact percentage deltas.
• Monthly Revenue & Profit Trajectory: Dual-axis bar and line visualization comparing top-line revenue against estimated net profits.
• Category Revenue Mix: Interactive donut chart breaking down volume and revenue across product sectors.

### 2. Data Hub & Automated EDA
• Data Quality Diagnostics: Quantifies raw input rows, cancelled orders (C prefix), invalid non-positive quantities or unit prices, missing customer identifiers, and duplicate transactions.
• Four Diagnostic Views:
  1. Preview & Download: Inspect paginated records with customizable sample size and export the cleaned dataset as a CSV.
  2. Missing & Outliers: Detailed missingness audit alongside Interquartile Range (IQR) fence calculations (Q1 - 1.5 × IQR to Q3 + 1.5 × IQR) and interactive log-scale box plots.
  3. Statistics & Correlation: Five-number statistical summary across all numerical variables coupled with a dynamic Pearson correlation matrix.
  4. Revenue Intensity Matrix: Heatmap analyzing sales intensity across Day of the Week vs. Month of the Year.

### 3. Sales, Product & Regional Intelligence
• Flexible Time Horizons: Switch between Daily granularity (overlaying 7-day and 30-day moving averages) and Monthly trends (with 3-month rolling averages).
• Product Leaderboards: Interactive horizontal bar charts comparing Top 10 SKUs by total gross revenue against Top 10 SKUs by total unit volume.
• Geographic Performance: Country-level revenue rankings paired with an interactive Plotly Choropleth world map and sortable tabular breakdowns.

### 4. Customer Segmentation (RFM + Unsupervised ML)
• RFM Feature Matrix:
  - Recency (R): Days elapsed since customer's last completed purchase.
  - Frequency (F): Total count of distinct orders placed.
  - Monetary (M): Lifetime gross expenditure.
  - AOV & Tenure: Average order basket size and total customer lifespan (days).
• Clustering Options:
  - K-Means: Configurable cluster count K (between 2 and 8) with silhouette score evaluation.
  - DBSCAN: Density-based spatial clustering with tunable epsilon (neighborhood distance) and min_samples, explicitly identifying noise and anomalies (e.g. bulk wholesale accounts).
• Automated Persona Mapping:
  - Champions: High recency, high frequency, high spend. Strategy: VIP access, early releases, zero aggressive discounts.
  - Loyalists: Consistent repeat buyers. Strategy: Loyalty programs, premium bundles, review requests.
  - Potential Loyalists: Recent buyers showing positive momentum. Strategy: Onboarding nurture streams, category cross-sells.
  - At-Risk: Valuable buyers who have slowed down. Strategy: Immediate win-back sequences and personalized offers.
  - Lost: Long inactive with low spend. Strategy: Low-cost automated reactivation; prune from paid media retargeting.
  - Outliers / Noise: Atypical purchasing behaviors. Strategy: Manual account audits.
• Visualizations: 2D Principal Component Analysis (PCA) projection and 3D logarithmic RFM space scatter plot.

### 5. Hybrid Demand Forecasting Engine
• Ensemble Architecture:
  - Prophet Component: Models overall trend, weekly seasonality, and yearly Fourier patterns. If prophet is not installed, the platform seamlessly defaults to a ridge-regularized Fourier harmonic additive model.
  - Autoregressive Gradient Boosting (GBM): Predicts residual demand using lag features (t-1, t-7, t-14), rolling statistics (7-day and 30-day means and standard deviations), calendar features (day-of-week, month), and continuous sine/cosine cyclical day-of-year embeddings.
  - Tunable Blending Slider: Allows analysts to balance Prophet's global trend capturing with GBM's short-term autoregressive sensitivity:
    Ensemble Forecast = (w × Prophet) + ((1 - w) × GBM)
• Holdout Validation: Evaluated against a 28-day (or 14-day) holdout period tracking Daily MAPE, Weekly WMAPE, and RMSE against a baseline Seasonal Naive model.
• What-If Promotional Simulator: Interactive demand shock slider (-50% to +100%) updating forecasts in real time with dynamic 95% confidence bands.

### 6. Customer Churn Prediction & Explainability
• Strict Time-Split Formulation: Avoids lookahead data leakage by deriving features strictly prior to a historical snapshot date, labeling a customer as churned (y = 1) only if no purchases occur within the designated forward inactivity window (30 to 180 days).
• Classifiers: Random Forest Classifier or XGBoost (with Gradient Boosting fallback), incorporating class weighting to handle cohort imbalances.
• Evaluation & Explainability:
  - Performance metrics: ROC-AUC, F1-Score, Precision, Recall, and ROC Curves.
  - Tree-based Gini/Gain feature importance.
  - Permutation Importance (SHAP-style delta AUC degradation) to show which factors drive attrition.
• Risk Cohort Stratification: Segment customers into Low, Medium, and High Risk groups with exportable watchlist CSVs detailing customer IDs and total historical value at risk.

### 7. Inventory Optimization & What-If Simulator
Connects demand forecasting directly to operations using classical multi-echelon inventory control formulas:

1. Safety Stock (SS): Protects against demand variability during supplier lead time:
   SS = Z × σ_daily × √L
   (where Z is the normal inverse CDF for 90%, 95%, or 99% service levels, σ_daily is daily demand standard deviation, and L is lead time in days).

2. Reorder Point (ROP): The inventory threshold that triggers a purchase order:
   ROP = (d_avg × L) + SS
   (where d_avg is the forecasted average daily demand).

3. Economic Order Quantity (EOQ): Minimizes total inventory holding and ordering costs:
   EOQ = √(2 × D × S / H)
   (where D is annualized demand, S is fixed cost per order, and H is unit holding cost per year).

4. Max Stock Level:
   MaxLevel = ROP + EOQ

5. Actionable Status per SKU:
   • Reorder Immediately: Current On-Hand ≤ ROP
   • Optimal Stock: ROP < On-Hand < MaxLevel
   • Reduce Overstock: On-Hand ≥ MaxLevel (quantifies exact excess capital tied up in £)

• Forecast-Driven Scaling: Integrates with the forecasting engine, automatically adjusting run rates based on projected 30-day forward trends.
• Sensitivity Curves: Plots Safety Stock Value vs. Service Level and Total ROP Capital vs. Supplier Lead Time.

### 8. MLOps Monitoring, Data Drift & Model Registry
• Two-Sample Kolmogorov-Smirnov (KS) Drift Detection:
  - Compares the empirical distribution of recent production features (Quantity, UnitPrice, TotalPrice, OrderValue) against the historical baseline.
  - Flags drift alerts whenever the asymptotic p-value falls below the chosen significance threshold alpha (0.01, 0.05, 0.10).
• Interactive Drift Simulator: Inject synthetic covariate shifts (-60% to +150%) to test detector sensitivity.
• Empirical CDF & Log Histograms: Visual inspection of distribution discrepancies and weekly KS trend lines over time.
• MLflow-Style Model Registry:
  - Manages versioning and stage transitions (Archived, Staging, Production).
  - Tracks model run IDs, validation metrics (MAPE, RMSE, ROC-AUC, F1, Silhouette score), and training timestamps.
  - Automatically captures and registers models trained live in the current dashboard session.

## Data Pipeline & Schema Specifications

RetailPulse accepts datasets structured according to the UCI / Kaggle Online Retail II specification:

| Field | Type | Description | Cleaning Rules Applied |
|---|---|---|---|
| InvoiceNo | String / Object | 6-digit invoice number | Strips whitespace; removes cancellations (values starting with 'C') |
| StockCode | String / Object | Product item code | Normalized to uppercase; excludes operational codes (POST, D, M, BANK CHARGES, etc.) |
| Description | String / Object | Nominal item name | Nulls replaced with "UNKNOWN ITEM"; used for automated category classification |
| Quantity | Integer / Float | Quantity per transaction | Coerced to numeric; filters rows where Quantity ≤ 0 |
| InvoiceDate | Datetime | Timestamp of transaction | Parsed with format validation; unparseable rows dropped |
| UnitPrice | Float | Unit price (£ Sterling) | Coerced to numeric; filters rows where UnitPrice ≤ 0 |
| CustomerID | Float / Int | 5-digit customer identifier | Converted to nullable Int64; retained for aggregate sales, excluded for customer-level ML |
| Country | String | Customer country of residence | Stripped and standardized; nulls assigned "Unspecified" |

### Built-in Intelligent Aliasing
The loader automatically maps common variations:
• invoice, invoice no → InvoiceNo
• stock code → StockCode
• unit price, price → UnitPrice
• customer id → CustomerID
• invoice date → InvoiceDate

### Automated Category Classification
Item descriptions are mapped to 8 retail categories via optimized regular expressions:
1. Toys & Games (TOY, GAME, DOLL, PUZZLE, JIGSAW, TRAIN, etc.)
2. Gifts & Seasonal (CHRISTMAS, XMAS, EASTER, GIFT, CARD, WRAP, etc.)
3. Lighting & Candles (CANDLE, LIGHT, LANTERN, LAMP)
4. Bags & Accessories (BAG, PURSE, WALLET, UMBRELLA, LUGGAGE)
5. Garden & Outdoor (GARDEN, PLANT, BIRD, TROWEL, WATERING, SEED)
6. Stationery & Crafts (NOTEBOOK, PENCIL, PEN, STICKER, STATIONERY, RIBBON, etc.)
7. Kitchen & Dining (MUG, CUP, TEAPOT, JAR, TIN, PLATE, BOWL, CAKE, etc.)
8. Home Decor (CUSHION, FRAME, CLOCK, VASE, MIRROR, DOORSTOP, HEART, etc.)

## Getting Started & Installation

### Prerequisites
• Python: Version 3.10, 3.11, or 3.12
• Package Manager: pip (or conda / uv)
• Docker (Optional): For running as an isolated container

### Local Installation

#### 1. Navigate to the Directory
```bash
cd retailpulse
```

#### 2. Create and Activate a Virtual Environment
• On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

• On Windows (Command Prompt):
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

• On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Core Dependencies
Install the required packages for the dashboard, data cleaning, and core ML algorithms:
```bash
pip install -r requirements.txt
```

#### 4. (Optional) Install Advanced ML Models
To enable official Facebook Prophet and XGBoost:
```bash
pip install -r requirements-optional.txt
```
Note for Windows users: Prophet requires C++ compilation tools. If you encounter issues, you can skip this step. RetailPulse includes built-in Fourier-Ridge and Gradient Boosting fallbacks that run with zero external compiler dependencies.

#### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your web browser at:
http://localhost:8501

### Running with Docker

RetailPulse includes a production-ready Dockerfile based on python:3.11-slim with built-in health checks.

#### Option A: Standard Build (Fast & Lightweight)
```bash
# Build the container image
docker build -t retailpulse .

# Run the container
docker run -d --name retailpulse-app -p 8501:8501 retailpulse
```

#### Option B: Build with Optional Models (Prophet + XGBoost)
```bash
# Build container with Prophet & XGBoost enabled
docker build --build-arg INSTALL_OPTIONAL=true -t retailpulse:full .

# Run the full container
docker run -d --name retailpulse-app -p 8501:8501 retailpulse:full
```

#### Access & Container Verification
• Open http://localhost:8501 in your browser.
• Verify container health status:
```bash
curl --fail http://localhost:8501/_stcore/health
```
• View container logs:
```bash
docker logs -f retailpulse-app
```
• Stop the container:
```bash
docker stop retailpulse-app && docker rm retailpulse-app
```

## Configuration & Customization

The project uses Streamlit's configuration file located at .streamlit/config.toml:

```toml
[theme]
base = "dark"
primaryColor = "#00D1B2"               # Teal / Turquoise accent
backgroundColor = "#0B1220"            # Deep Navy background
secondaryBackgroundColor = "#111A2E"   # Card & Sidebar background
textColor = "#E6EDF7"                  # Light high-contrast text
font = "sans serif"

[server]
headless = true
maxUploadSize = 500                    # Supports large datasets up to 500 MB
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[client]
toolbarMode = "minimal"
```

### Adjusting Business Constants in app.py
You can fine-tune business assumptions directly in app.py:
• CATEGORY_COST_RATIO: Cost of goods sold (COGS) ratios per category for profit estimation.
• PERSONAS: Tailor persona marketing recommendations and retention strategies.
• NON_PRODUCT_CODES: Add or remove ledger transaction codes that should be excluded from merchandise analytics.

## Project Directory Structure

```
retailpulse/
├── .streamlit/
│   └── config.toml                  # Custom theme (dark mode), server, and upload settings
├── data/
│   └── kaggle_online_retail_sample.csv # Curated multi-year Kaggle Online Retail II dataset
├── app.py                           # Core application: 8 analytical modules, ML models & UI
├── Dockerfile                       # Container definition with healthcheck and multi-stage options
├── kaggle_online_retail_sample.csv  # Root sample dataset for instant download and inspection
├── README.md                        # Platform documentation and startup guide
├── requirements.txt                 # Core dependencies (Streamlit, Scikit-learn, Plotly, Pandas)
└── requirements-optional.txt        # Optional ML packages (Prophet, XGBoost)
```

## Technology Stack

| Layer | Technologies |
|---|---|
| Web Framework & UI | Streamlit (v1.49+), Responsive Custom CSS, Modern Card Components |
| Data Processing & Analytics | Pandas (v2.0+), NumPy (v1.24+), openpyxl |
| Scientific & Statistical Computing | SciPy (scipy.stats norm, two-sample Kolmogorov-Smirnov test) |
| Machine Learning (Unsupervised) | Scikit-Learn (KMeans, DBSCAN, PCA, StandardScaler, silhouette_score) |
| Machine Learning (Supervised) | Scikit-Learn (RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, Ridge) |
| Advanced Forecasting & ML (Optional) | Prophet (v1.1.5+), XGBoost (v2.0+) |
| Interactive Visualizations | Plotly Express & Graph Objects (3D scatter, choropleth, eCDF, heatmaps) |
| Containerization & Deployment | Docker (Python 3.11-slim base image) |

## Troubleshooting & FAQs

### Q1: What dataset does RetailPulse load by default?
A: By default, RetailPulse automatically loads the pre-bundled, authentic Kaggle Online Retail II dataset (`kaggle_online_retail_sample.csv`). You can switch at any time in the sidebar to the high-volume "Synthetic Enterprise Feed" or "Upload Custom File" to analyze your own transactional data.

### Q2: How do I test or upload my own CSV or Excel dataset?
A: Scroll to the bottom of the sidebar to the "Data Source & Ingestion" section. You can download the pre-bundled Kaggle sample CSV with one click using the "Download Kaggle Sample CSV" button to inspect its format in Excel, or select "Upload Custom File (CSV/Excel)" and drag-and-drop your dataset. RetailPulse will automatically clean, validate, and compute all analytics in real time.

### Q3: What if pip install -r requirements-optional.txt fails on Windows?
A: Prophet relies on cmdstanpy and C++ toolchains which can occasionally cause issues on clean Windows environments. RetailPulse provides built-in fallback models:
• Forecasting automatically falls back to an additive Fourier-Ridge model.
• Churn modeling falls back to an optimized GradientBoostingClassifier.
All 8 modules will function seamlessly without installing requirements-optional.txt.

### Q4: How do I change the port or run on a remote server?
A: Run Streamlit with the --server.port and --server.address flags:
```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

## License & Acknowledgements

• License: Distributed under the MIT License.
• Dataset Attribution: Based on the Online Retail II data set courtesy of the UCI Machine Learning Repository (Daqing Chen, Sai Liang Sain, and Kun Guo).
• Built With: Python, Streamlit, Scikit-learn, and Plotly.
# RetailPulse
