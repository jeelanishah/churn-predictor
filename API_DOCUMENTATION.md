# API Documentation

Reference for the `ChurnPredictor` Python API (`api.py`).

---

## Overview

`ChurnPredictor` is a Python class that wraps a trained Gradient Boosting model.
It is imported directly by the Streamlit frontend — there is no HTTP server.

```python
from api import predictor          # module-level singleton
result = predictor.predict({...})
```

---

## Input Features

All 19 features must be supplied for every prediction.

| # | Feature | Type | Accepted Values |
|---|---------|------|-----------------|
| 1 | `gender` | categorical | `Male`, `Female` (or index `0`, `1`) |
| 2 | `SeniorCitizen` | binary | `0`, `1` |
| 3 | `Partner` | categorical | `No`, `Yes` |
| 4 | `Dependents` | categorical | `No`, `Yes` |
| 5 | `tenure` | numeric | `0`–`72` (months) |
| 6 | `PhoneService` | categorical | `No`, `Yes` |
| 7 | `MultipleLines` | categorical | `No`, `Yes`, `No phone service` |
| 8 | `InternetService` | categorical | `DSL`, `Fiber optic`, `No` |
| 9 | `OnlineSecurity` | categorical | `No`, `Yes`, `No internet service` |
| 10 | `OnlineBackup` | categorical | `No`, `Yes`, `No internet service` |
| 11 | `DeviceProtection` | categorical | `No`, `Yes`, `No internet service` |
| 12 | `TechSupport` | categorical | `No`, `Yes`, `No internet service` |
| 13 | `StreamingTV` | categorical | `No`, `Yes`, `No internet service` |
| 14 | `StreamingMovies` | categorical | `No`, `Yes`, `No internet service` |
| 15 | `Contract` | categorical | `Month-to-month`, `One year`, `Two year` |
| 16 | `PaperlessBilling` | categorical | `No`, `Yes` |
| 17 | `PaymentMethod` | categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| 18 | `MonthlyCharges` | numeric | `0.0`–`500.0` |
| 19 | `TotalCharges` | numeric | `0.0`–`50000.0` |

Categorical fields also accept integer indices (e.g. `Contract=2` → `"Two year"`).

---

## Methods

### `predict(customer_data: dict) -> dict`

Predict churn for a single customer.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `customer_data` | `dict` | All 19 features as key-value pairs |

**Success response**

```python
{
    "status": "success",
    "prediction": 1,                  # 0 = no churn, 1 = churn
    "probability": 0.8234,            # churn probability (same as probability_churn)
    "probability_churn": 0.8234,      # probability of churn
    "probability_no_churn": 0.1766,   # probability of no churn
    "risk_level": "🔴 HIGH RISK"      # HIGH / MEDIUM / LOW
}
```

**Error response**

```python
{
    "status": "error",
    "error": "Missing required features: ['tenure']"
}
```

---

### `predict_batch(records) -> dict`

Predict churn for multiple customers.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `records` | `list[dict]` or `pandas.DataFrame` | One record per customer |

**Success response**

```python
{
    "status": "success",
    "results": [
        {
            "status": "success",
            "prediction": 0,
            "probability": 0.15,
            "probability_churn": 0.15,
            "probability_no_churn": 0.85,
            "risk_level": "🟢 LOW RISK",
            "row_index": 0
        },
        ...
    ],
    "summary": {
        "total": 2,
        "successful": 2,
        "failed": 0
    }
}
```

---

### `health_check() -> dict`

Return the current readiness of the API.

**Response**

```python
{
    "status": "✅ API is running",
    "model_loaded": True,
    "scaler_loaded": True,
    "features_loaded": True,
    "encoders_loaded": True,
    "model_source": "artifacts",   # "artifacts" | "fallback_training" | "unavailable"
    "version": "2.0.0"
}
```

---

## Risk Levels

| Churn Probability | Risk Label |
|-------------------|-----------|
| ≥ 70 % | `🔴 HIGH RISK` |
| 40 – 69 % | `🟡 MEDIUM RISK` |
| < 40 % | `🟢 LOW RISK` |

---

## Example Usage

```python
from api import predictor

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0,
}

result = predictor.predict(customer)
print(result["risk_level"])        # 🔴 HIGH RISK
print(result["probability"])       # e.g. 0.8234
```

---

## Model Information

| Property | Value |
|----------|-------|
| Algorithm | `GradientBoostingClassifier` |
| Training samples | ~14 000 |
| Accuracy | 95.20 % |
| Precision | 92.50 % |
| Recall | 88.30 % |
| F1-Score | 90.35 % |
| AUC | 0.95 |
