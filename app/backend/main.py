from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your pre-trained model here, place your .pkl in the model directory
model = joblib.load('model/model.pkl')  # Make sure you have this file!

class CustomerInput(BaseModel):
    gender: str
    senior_citizen: int
    # ... add all your input fields here (use snake_case for Python style) ...

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(c: CustomerInput):
    # 1. Preprocess c into NumPy array X in the same order as during training!
    X = np.array([[c.gender, c.senior_citizen, ...]])  # You must preprocess as needed!
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # Always probability for class 1 (Churn)
    return {"prediction": int(pred), "probability": float(proba)}