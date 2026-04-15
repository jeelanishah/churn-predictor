"""
Churn Predictor API - Pure Python Backend
"""

import pickle
import pandas as pd
from pathlib import Path

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = None
        self.target_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load all model components"""
        try:
            model_path = Path(__file__).parent / "model"
            
            with open(model_path / "churn_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            
            with open(model_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            with open(model_path / "feature_names.pkl", "rb") as f:
                self.feature_names = pickle.load(f)
            
            with open(model_path / "label_encoders.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)
            
            with open(model_path / "target_encoder.pkl", "rb") as f:
                self.target_encoder = pickle.load(f)
            
            print("✅ Model loaded successfully!")
            return True
        
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict(self, customer_data):
        """
        Make prediction for a customer
        
        Args:
            customer_data: dict with customer features
        
        Returns:
            dict with prediction, probability, and risk level
        """
        try:
            # Create DataFrame with single row
            df = pd.DataFrame([customer_data])
            
            # Ensure correct column order (IMPORTANT!)
            df = df[self.feature_names]

            # Encode categorical features before scaling
            for column, encoder in self.label_encoders.items():
                if column in df.columns:
                    try:
                        df[column] = encoder.transform(df[column])
                    except ValueError as e:
                        return {
                            "error": f"Invalid value for '{column}': {df[column].iloc[0]}. {str(e)}",
                            "status": "error"
                        }
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            # Determine risk level
            if probability >= 0.7:
                risk_level = "🔴 HIGH RISK"
            elif probability >= 0.4:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            return {
                "prediction": int(prediction),
                "probability": round(float(probability), 4),
                "risk_level": risk_level,
                "status": "success"
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def health_check(self):
        """Check if API is ready"""
        return {
            "status": "✅ API is running",
            "model_loaded": self.model is not None,
            "version": "1.0.0"
        }

# Initialize predictor globally
predictor = ChurnPredictor()
