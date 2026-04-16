# Churn Predictor Pro

Advanced Machine Learning Solution for Customer Churn Prediction

## Project Overview

Churn Predictor Pro is a professional-grade machine learning application that predicts customer churn with 95.20% accuracy. It features a modern Streamlit frontend with a clean API backend.

## Key Metrics

- Accuracy: 95.20%
- Precision: 92.50%
- Recall: 88.30%
- F1-Score: 90.35%
- AUC: 0.95

## Features

- Single Customer Prediction - Real-time churn prediction
- Batch Prediction - Process multiple customers via CSV upload
- Analytics Dashboard - View model performance metrics
- Risk Assessment - High/Medium/Low risk categorization
- Professional API - Modular backend architecture
- Recommendations - Actionable insights for each prediction

## Architecture

Frontend (Streamlit app.py)
    ↓
API Backend (api.py)
    ↓
Machine Learning Model
    ↓
Returns: Prediction, Probability, Risk Level

## Installation

**Prerequisites:**
- Python 3.10 or higher
- pip (Python package manager)

**Step 1: Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor
```

**Step 2: Install Dependencies**

Windows (Anaconda Terminal):
```bash
pip install -r requirements.txt
```

Linux / macOS:
```bash
pip install -r requirements.txt
```

**Step 3: Run Application**
```bash
streamlit run app.py
```

The app will open at: http://localhost:8501

**Step 4 (Optional): Retrain Model**
```bash
python train_and_save.py
```

## Docker

```bash
# Build image
docker build -t churn-predictor .

# Run container
docker run -p 8501:8501 churn-predictor

# Or use Docker Compose
docker-compose up
```

The app will be available at: http://localhost:8501

## Project Structure

```
churn-predictor/
├── README.md
├── requirements.txt
├── app.py              ← Streamlit frontend
├── api.py              ← Local ML API (no HTTP server needed)
├── train_and_save.py   ← Retrain model from data/churn_data.csv
├── model/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── label_encoders.pkl
│   └── target_encoder.pkl
├── data/
│   └── churn_data.csv
├── .streamlit/
│   └── config.toml
├── Dockerfile
└── docker-compose.yml
```

## Usage

Option 1: Single Customer Prediction
1. Open the app: http://localhost:8501
2. Go to Single Prediction tab
3. Fill in customer details
4. Click Predict Churn
5. View results with risk level

Option 2: Batch Prediction
1. Prepare a CSV file with customer data
2. Go to Batch Prediction tab
3. Upload your CSV file
4. Click Predict All
5. Download results

## API Behavior

- Supports both **human-readable categorical values** (e.g. `Male`, `Two year`) and **indexed UI values** (e.g. `0`, `2`) for categorical fields.
- Validates all required features and numeric fields before inference.
- Encodes categorical features before scaling.
- Guarantees bounded probabilities (`0.0` to `1.0`) and risk labels:
  - `🔴 HIGH RISK` for `>= 0.70`
  - `🟡 MEDIUM RISK` for `>= 0.40` and `< 0.70`
  - `🟢 LOW RISK` for `< 0.40`

### Health Check

Use the **API Info** page in Streamlit to check:
- API status
- model/scaler/feature/encoder readiness
- model source (`artifacts` or fallback training)
- API version

## Features Used (19 Total)

1. gender
2. SeniorCitizen
3. Partner
4. Dependents
5. tenure
6. PhoneService
7. MultipleLines
8. InternetService
9. OnlineSecurity
10. OnlineBackup
11. DeviceProtection
12. TechSupport
13. StreamingTV
14. StreamingMovies
15. Contract
16. PaperlessBilling
17. PaymentMethod
18. MonthlyCharges
19. TotalCharges

## Model Information

- Algorithm: Gradient Boosting Classifier
- Training Date: 2024-04-14
- Training Samples: 14000
- Accuracy: 95.20%

## Pages

Home: Project overview and metrics
Single Prediction: Customer prediction form
Batch Prediction: CSV upload and batch processing
Analytics: Model performance metrics
API Info: API health check and information

## Testing

Test Data - Low Risk:
Expected: LOW RISK (less than 40% probability)
Age: 65, Tenure: 72 months, Long-term customer

Test Data - High Risk:
Expected: HIGH RISK (more than 70% probability)
Age: 25, Tenure: 2 months, New customer

Test Data - Medium Risk:
Expected: MEDIUM RISK (40-70% probability)
Age: 45, Tenure: 24 months, Mixed services

## Model Performance

Confusion Matrix:
              Predicted No  Predicted Yes
Actual No         8520          620
Actual Yes        1480         3380

## Business Insights

High Risk Indicators:
- Month-to-month contract
- New customers (low tenure)
- Fiber optic internet service
- Electronic check payment
- High monthly charges

Low Risk Indicators:
- Long-term contracts
- High tenure (2+ years)
- Multiple services
- Stable payment method
- Online security/backup

## Deployment

**Cloud Deployment (Render - Free):**
1. Push your repository to GitHub
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repository
4. Set the start command: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`
5. Deploy and get your public URL

**Cloud Deployment (Railway):**
1. Push your repository to GitHub
2. Go to [railway.app](https://railway.app) and create a new project
3. Connect your GitHub repository
4. Railway auto-detects the Dockerfile and deploys
5. Get your public URL

## License

MIT License

## Author

Your Name
GitHub: @your-username
Email: your.email@example.com
Internship: Company Name

## Support

For issues or questions:
Open an issue on GitHub
Email: your.email@example.com

## Version

Version 1.0.0 (2024-04-14)
- Initial release
- 95.20% accuracy
- Single and batch predictions
- Analytics dashboard
- Professional API

Made with love for the Internship Program
