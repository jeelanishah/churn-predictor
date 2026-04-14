from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Initialize FastAPI
app = FastAPI(
    title="Churn Predictor API",
    description="ML API for predicting customer churn",
    version="1.0.0"
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODEL AND PREPROCESSORS
# ============================================================
MODEL_PATH = Path(__file__).parent.parent / "model"

def load_model_and_preprocessors():
    """Load all model components"""
    try:
        with open(MODEL_PATH / "churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open(MODEL_PATH / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        with open(MODEL_PATH / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        with open(MODEL_PATH / "label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        
        with open(MODEL_PATH / "target_encoder.pkl", "rb") as f:
            target_encoder = pickle.load(f)
        
        return model, scaler, feature_names, label_encoders, target_encoder
    
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Load model at startup
try:
    model, scaler, feature_names, label_encoders, target_encoder = load_model_and_preprocessors()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

# ============================================================
# PYDANTIC SCHEMAS (INPUT VALIDATION)
# ============================================================
class CustomerData(BaseModel):
    """Schema for single customer prediction"""
    Age: int
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    NumericContract: int
    NumericInternetService: int
    NumericOnlineSecurity: int
    NumericOnlineBackup: int
    NumericDeviceProtection: int
    NumericTechSupport: int
    NumericStreamingTV: int
    NumericStreamingMovies: int
    NumericPhoneService: int
    NumericMultipleLines: int
    NumericPaperlessBilling: int
    NumericPaymentMethod: int
    NumericGender: int
    NumericDependents: int
    NumericPartner: int

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Tenure": 24,
                "MonthlyCharges": 75.50,
                "TotalCharges": 1812.00,
                "NumericContract": 1,
                "NumericInternetService": 1,
                "NumericOnlineSecurity": 1,
                "NumericOnlineBackup": 0,
                "NumericDeviceProtection": 0,
                "NumericTechSupport": 1,
                "NumericStreamingTV": 0,
                "NumericStreamingMovies": 0,
                "NumericPhoneService": 1,
                "NumericMultipleLines": 1,
                "NumericPaperlessBilling": 0,
                "NumericPaymentMethod": 2,
                "NumericGender": 1,
                "NumericDependents": 0,
                "NumericPartner": 1
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: int
    probability: float
    risk_level: str

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_churn(customer_data: dict):
    """
    Make prediction for a customer
    Returns: (prediction, probability, risk_level)
    """
    try:
        # Create DataFrame with single row
        df = pd.DataFrame([customer_data])
        
        # Ensure correct column order
        df = df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "🔴 HIGH RISK"
        elif probability >= 0.4:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🟢 LOW RISK"
        
        return int(prediction), float(probability), risk_level
    
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "✅ API is running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
async def predict(data: CustomerData):
    """
    Make churn prediction for a customer
    
    Returns:
    - prediction: 0 (No Churn) or 1 (Churn)
    - probability: Probability of churn (0-1)
    - risk_level: HIGH/MEDIUM/LOW risk
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        prediction, probability, risk_level = predict_churn(data.dict())
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Churn Predictor API",
        "version": "1.0.0",
        "description": "ML API for predicting customer churn with 95.20% accuracy",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "docs": "GET /docs"
        }
    }

# ============================================================
# RUN SERVER (if executed directly)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
