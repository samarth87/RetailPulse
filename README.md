RetailPulse — AI-Powered Retail Intelligence
<p align="center">
  <strong>Transforming retail transactions into actionable business decisions.</strong>
</p><p align="center">
  <a href="https://retailpulse0.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p><p align="center">
  <a href="https://retailpulse0.streamlit.app/"><strong>🚀 Open the Live App</strong></a>
  &nbsp; | &nbsp;
  <a href="https://github.com/samarth87/RetailPulse"><strong>GitHub Repository</strong></a>
</p>📌 Project Overview
RetailPulse is an AI-powered retail analytics and decision-intelligence dashboard built with Python and Streamlit. It helps transform retail transaction data into useful insights for sales performance, customer behavior, demand planning, inventory decisions, and model monitoring.

The platform is designed around the UCI/Kaggle Online Retail II transaction schema and supports a bundled sample dataset, a synthetic enterprise feed, and user-uploaded CSV or Excel data.
🎯 What RetailPulse helps analyze
- Retail sales performance and business KPIs
- Customer purchasing behavior and RFM segmentation
- Demand forecasting and promotional scenarios
- Customer churn-risk indicators
- Inventory replenishment and stock-level planning
- Data quality, distribution shifts, and model monitoring
Try it: retailpulse0.streamlit.app

✨ Key Features
Module	What it does
Executive Overview	Summarizes revenue, estimated profit, orders, active customers, average order value, and period-over-period changes.
Data Hub & Automated EDA	Uploads and cleans CSV/Excel data, checks data quality, previews records, explores missing values and outliers, and exports cleaned data.
Sales & Product Analytics	Explores daily/monthly trends, moving averages, top products by revenue or volume, and country-level performance.
Customer Segmentation	Builds customer RFM metrics and supports K-Means and DBSCAN clustering, persona mapping, and PCA visualizations.
Demand Forecasting	Forecasts demand using Prophet when available or a Fourier-Ridge fallback, with autoregressive Gradient Boosting, validation metrics, and what-if demand scenarios.
Churn Risk	Uses time-aware customer features and inactivity labels to estimate churn risk, evaluate classifiers, and inspect feature importance.
Inventory Optimization	Calculates safety stock, reorder point, EOQ, and maximum stock level; supports stock status and scenario analysis.
MLOps Monitoring	Provides KS-based data drift checks, drift simulation, weekly tracking, and an MLflow-style model registry.


🧱 System Architecture
RetailPulse follows a five-layer analytics and ML workflow:
┌──────────────────────────────────────────────────────────────┐
│                    RETAILPULSE PLATFORM                   │
├──────────────────────────────────────────────────────────────┤
│  1. DATA LAYER                                             │
│     CSV / Excel / Online Retail II / Synthetic Feed        │
│     Cleaning, validation, deduplication, normalization     │
├──────────────────────────────────────────────────────────────┤
│  2. FEATURE ENGINEERING                                    │
│     RFM metrics, time-series lags, rolling statistics,     │
│     calendar features, and customer purchase-gap features  │
├──────────────────────────────────────────────────────────────┤
│  3. MODEL LAYER                                            │
│     K-Means, DBSCAN, PCA, forecasting models,              │
│     Random Forest / Gradient Boosting / optional XGBoost   │
├──────────────────────────────────────────────────────────────┤
│  4. SERVING & SIMULATION                                   │
│     Streamlit dashboard, demand scenarios, inventory        │
│     simulations, visual analytics, and downloadable outputs │
├──────────────────────────────────────────────────────────────┤
│  5. MLOPS & GOVERNANCE                                     │
│     KS drift detection, drift tracking, model registry,     │
│     run logging, and validation metrics                    │
└──────────────────────────────────────────────────────────────┘
📊 Analytics & ML Modules
1. Executive Overview
- Revenue, estimated profit, orders, active customers, and average order value (AOV)
- Comparison against a preceding period of the same duration
- Monthly revenue and profit trajectory
- Category revenue mix
2. Data Hub & Automated EDA
- Input row, cancellation, invalid quantity/price, missing customer ID, and duplicate diagnostics
- Paginated data preview and cleaned CSV export
- Missing-value and IQR-based outlier analysis
- Numerical summaries and Pearson correlation matrix
- Day-of-week versus month sales-intensity heatmap
3. Sales, Product & Regional Intelligence
- Daily trends with 7-day and 30-day moving averages
- Monthly trends with 3-month rolling averages
- Top 10 products by revenue and unit volume
- Country-level revenue rankings and geographic visualization
4. Customer Segmentation (RFM + Unsupervised ML)
Customer features include:
- Recency: Days since the customer's last completed purchase
- Frequency: Number of distinct orders
- Monetary: Lifetime gross expenditure
- AOV and Tenure: Average basket size and customer lifespan
Supported clustering:
- K-Means with configurable cluster count and silhouette evaluation
- DBSCAN with adjustable neighborhood and minimum-sample parameters
- PCA-based 2D and 3D visualizations
Customer personas include Champions, Loyalists, Potential Loyalists, At-Risk, Lost, and Outliers/Noise, with suggested business playbooks.
5. Hybrid Demand Forecasting
- Prophet when installed, with an additive Fourier-Ridge fallback
- Autoregressive Gradient Boosting using lag, rolling, and calendar features
- Adjustable blending between forecast components
- Holdout validation using MAPE, WMAPE, and RMSE
- Promotional demand-shock simulation and forecast confidence bands
6. Customer Churn Prediction
- Time-split feature and label construction intended to avoid look-ahead leakage
- Random Forest or XGBoost where available, with Gradient Boosting fallback
- ROC-AUC, F1, precision, recall, and ROC curve evaluation
- Tree-based and permutation-based feature importance
- Low-, medium-, and high-risk customer cohorts with exportable watchlists
7. Inventory Optimization
RetailPulse applies classical inventory formulas:

Safety Stock
SS = Z × σ_daily × √L
Reorder Point
ROP = (d_avg × L) + SS
Economic Order Quantity
EOQ = √(2 × D × S / H)
Maximum Stock Level
MaxLevel = ROP + EOQ
Where:
- Z = service-level z-score
- σ_daily = daily demand standard deviation
- L = supplier lead time in days
- d_avg = average daily demand
- D = annualized demand
- S = fixed ordering cost
- H = annual holding cost per unit
The dashboard classifies SKUs into reorder, optimal-stock, and overstock statuses and supports forecast-driven and sensitivity scenarios.
8. MLOps Monitoring & Model Registry
- Two-sample Kolmogorov–Smirnov (KS) drift detection
- Feature-distribution comparison and drift alerts based on p-value thresholds
- Simulated covariate shifts and empirical CDF visualizations
- Weekly drift tracking
- MLflow-style model registry stages: Archived, Staging, and Production
- Model run logging with metrics and timestamps
🗂️ Project Structure
The following is the documented project layout. Actual files may vary depending on the current repository version.
RetailPulse/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── kaggle_online_retail_sample.csv
├── app.py
├── Dockerfile
├── kaggle_online_retail_sample.csv
├── README.md
├── requirements.txt
└── requirements-optional.txt
🧾 Dataset
RetailPulse uses the Online Retail II transaction structure and supports CSV/Excel uploads.
Field	Description
InvoiceNo	Invoice identifier; cancellation invoices are identified by a C prefix
StockCode	Product/item code
Description	Product description
Quantity	Transaction quantity
InvoiceDate	Transaction date and time
UnitPrice	Unit price in GBP (£)
CustomerID	Customer identifier; may be missing
Country	Customer country


Data preparation
The documented cleaning pipeline includes:
- Column alias normalization
- Cancellation filtering
- Exclusion of operational/non-product codes
- Numeric coercion and validation
- Removal of non-positive quantity or unit-price records
- Date parsing and invalid-date handling
- Duplicate checks
- Missing customer ID handling
- Country normalization
Customer identifiers that are missing can remain useful for aggregate sales analysis but are excluded from customer-level modeling.
🛠️ Technology Stack
Area	Tools
Dashboard	Streamlit
Data processing	Pandas, NumPy, openpyxl
Visualization	Plotly
Statistical analysis	SciPy
Machine learning	Scikit-learn
Optional forecasting / ML	Prophet, XGBoost
Containerization	Docker
Language	Python 3.10+ (Python 3.11 recommended in the documented Docker setup)


🚀 Run RetailPulse Locally
Prerequisites
- Python 3.10, 3.11, or 3.12
- pip
- Git (if cloning the repository)
1. Clone the repository
git clone https://github.com/samarth87/RetailPulse.git
cd RetailPulse
2. Create a virtual environment
Windows — Git Bash:
python -m venv venv
source venv/Scripts/activate
Windows — PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1
Linux / macOS:
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Optional advanced models:
pip install -r requirements-optional.txt
Prophet may require additional build tools on some Windows setups. The documented fallback forecasting and classification approaches allow the application to run without the optional packages.
4. Start the application
streamlit run app.py
Open:
http://localhost:8501
☁️ Live Deployment
The deployed Streamlit application is available here:
👉 Launch RetailPulse Live Demo
The application is hosted on Streamlit Community Cloud and connected to this GitHub project.
🐳 Docker
RetailPulse includes a Dockerfile in the documented project setup.

Build the image:
docker build -t retailpulse .
Run the container:
docker run -d --name retailpulse-app -p 8501:8501 retailpulse
Then visit:
http://localhost:8501
Optional models can be included using the documented build argument:
docker build --build-arg INSTALL_OPTIONAL=true -t retailpulse:full .
docker run -d --name retailpulse-app -p 8501:8501 retailpulse:full
⚙️ Configuration
Streamlit configuration is documented in:
.streamlit/config.toml
The project configuration supports a dark dashboard theme, server settings, and upload-size configuration. Business assumptions such as category cost ratios, customer personas, and excluded operational codes are configurable in the application code.
🧪 Notes & Limitations
- Forecasting and churn outputs are analytical estimates, not guarantees of future business outcomes.
- Model quality depends on the uploaded dataset, its coverage, and the selected validation setup.
- Optional Prophet and XGBoost functionality depends on their installation and compatibility.
- Inventory calculations depend on assumptions such as lead time, service level, ordering cost, and holding cost.
- The bundled dataset is a sample; users can upload their own compatible CSV or Excel transactions.
📄 License & Acknowledgements
- License: MIT
- Dataset attribution: Online Retail II dataset, credited in the project documentation to the UCI Machine Learning Repository (Daqing Chen, Sai Liang Sain, and Kun Guo).
- Built with: Python, Streamlit, Pandas, Scikit-learn, and Plotly.
👨‍💻 Author
Samarth Sehdev
- GitHub: @samarth87
- Project Repository: RetailPulse
- Live Application: retailpulse0.streamlit.app
<p align="center">
  <strong>RetailPulse — Turning retail data into smarter decisions.</strong>
</p>
