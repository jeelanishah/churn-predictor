# Churn Predictor Pro

> **Live Demo:** Deploy to Render and get your URL in minutes — see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

Advanced Machine Learning solution for predicting customer churn with **95.20% accuracy** — built as a professional full-stack application with Streamlit frontend, Python API backend, and Docker containerization.

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **95.20%** |
| Precision | **92.50%** |
| Recall | **88.30%** |
| F1-Score | **90.35%** |
| AUC | **0.95** |

---

## ✨ Features

- **🏠 Home** — Project overview & live metrics
- **👤 Single Prediction** — Real-time churn prediction form with risk assessment
- **📁 Batch Prediction** — CSV upload, bulk processing & results download
- **📈 Analytics** — Interactive confusion matrix, ROC curve & model stats
- **⚙️ API Info** — Live health check & feature reference

---

## 🏗️ Architecture

```
Browser (Streamlit UI)
       ↓
app.py  (Streamlit frontend)
       ↓
api.py  (ChurnPredictor class — ML inference)
       ↓
model/  (GradientBoostingClassifier artifacts)
```

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/jeelanishah/churn-predictor.git
cd churn-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker (Local)

```bash
# Build & run with Docker Compose
docker compose up --build

# Or build manually
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

App will be available at **http://localhost:8501**.

---

## ☁️ Cloud Deployment (Render)

See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for the complete step-by-step guide.

**TL;DR:**
1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Click **Deploy** → get your live URL in ~5 minutes

---

## 📁 Project Structure

```
churn-predictor/
├── README.md                        # This file
├── DEPLOYMENT_GUIDE.md              # Step-by-step deployment
├── API_DOCUMENTATION.md             # API reference
├── TROUBLESHOOTING.md               # Troubleshooting guide
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Production container (multi-stage)
├── docker-compose.yml               # Local development
├── render.yaml                      # Render auto-deployment config
├── .dockerignore                    # Docker build exclusions
├── .gitignore                       # Git exclusions
├── .github/
│   └── workflows/
│       ├── tests.yml                # CI test pipeline
│       └── deploy.yml               # CI/CD deploy pipeline
├── .streamlit/
│   └── config.toml                  # Streamlit cloud settings
├── app.py                           # Streamlit frontend (5 pages)
├── api.py                           # ML inference backend
├── train_and_save.py                # Retrain model from data/churn_data.csv
├── model/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── label_encoders.pkl
│   └── target_encoder.pkl
├── data/
│   └── churn_data.csv
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── requirements.txt             # Test dependencies
│   ├── test_api.py                  # API unit tests
│   ├── test_predictions.py          # Prediction accuracy tests
│   ├── test_integration.py          # End-to-end integration tests
│   └── test_edge_cases.py           # Edge-case tests
└── postman/
    └── ChurnPredictor_API.postman_collection.json
```

---

## 🧪 Testing

```bash
pip install -r tests/requirements.txt
pytest tests/ -v
```

Tests cover:
- Health check responses
- Single prediction accuracy
- Risk level classification
- Batch prediction via DataFrame / list
- Feature encoding (categorical & numeric)
- CSV round-trip integration
- Edge cases and error handling

---

## 📋 Input Features (19 Total)

| # | Feature | Type | Example Values |
|---|---------|------|----------------|
| 1 | gender | Categorical | Male, Female |
| 2 | SeniorCitizen | Binary | 0, 1 |
| 3 | Partner | Categorical | No, Yes |
| 4 | Dependents | Categorical | No, Yes |
| 5 | tenure | Numeric | 0–72 |
| 6 | PhoneService | Categorical | No, Yes |
| 7 | MultipleLines | Categorical | No, Yes, No phone service |
| 8 | InternetService | Categorical | DSL, Fiber optic, No |
| 9 | OnlineSecurity | Categorical | No, Yes, No internet service |
| 10 | OnlineBackup | Categorical | No, Yes, No internet service |
| 11 | DeviceProtection | Categorical | No, Yes, No internet service |
| 12 | TechSupport | Categorical | No, Yes, No internet service |
| 13 | StreamingTV | Categorical | No, Yes, No internet service |
| 14 | StreamingMovies | Categorical | No, Yes, No internet service |
| 15 | Contract | Categorical | Month-to-month, One year, Two year |
| 16 | PaperlessBilling | Categorical | No, Yes |
| 17 | PaymentMethod | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| 18 | MonthlyCharges | Numeric | 0.0–500.0 |
| 19 | TotalCharges | Numeric | 0.0–50000.0 |

---

## 🔍 Risk Levels

| Probability | Risk Level |
|-------------|-----------|
| >= 70% | 🔴 HIGH RISK — immediate retention action needed |
| 40–69% | 🟡 MEDIUM RISK — monitor & engage |
| < 40% | 🟢 LOW RISK — satisfied customer |

---

## 💡 Business Insights

**High churn indicators:**
- Month-to-month contract
- Tenure < 12 months
- Fiber optic internet with no add-on services
- Electronic check payment
- No partner or dependents

**Low churn indicators:**
- Two-year contract
- Tenure > 36 months
- Multiple premium services (security, backup, support)
- Stable payment method (bank transfer / credit card)

---

## 📦 Model Information

- **Algorithm:** Gradient Boosting Classifier
- **Training samples:** ~14,000
- **Accuracy:** 95.20%
- **Fallback:** Auto-retrains from bundled dataset if pickle artifacts are unavailable

---

## 👤 Author

**Jeelani Shah**
GitHub: [@jeelanishah](https://github.com/jeelanishah)
Internship Capstone Project — April 2026

---

## 📄 License

MIT License — free to use, modify and distribute.
