import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load saved model and artifacts
print("📂 Loading model artifacts...")
with open('model/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('model/target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

print("✅ Model loaded successfully!\n")

# Example: Predict for a NEW customer
print("=" * 60)
print("🔮 CHURN PREDICTION - NEW CUSTOMER")
print("=" * 60)

# Create sample customer data
sample_customer = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 24,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 65.0,
    'TotalCharges': 1560.0
}

print("\n📊 Customer Data:")
for key, value in sample_customer.items():
    print(f"   {key}: {value}")

# Convert to DataFrame
customer_df = pd.DataFrame([sample_customer])

# Encode categorical features
for col in label_encoders:
    if col in customer_df.columns:
        customer_df[col] = label_encoders[col].transform(customer_df[col])

# Scale features
customer_scaled = scaler.transform(customer_df[feature_names])

# Make prediction
prediction = model.predict(customer_scaled)[0]
probability = model.predict_proba(customer_scaled)[0]

print("\n" + "=" * 60)
print("🎯 PREDICTION RESULT")
print("=" * 60)

if prediction == 1:
    churn_status = "⚠️ WILL CHURN (Leave company)"
else:
    churn_status = "✅ WON'T CHURN (Stay with company)"

print(f"\n{churn_status}")
print(f"\nConfidence:")
print(f"   Stay Probability:  {probability[0]*100:.2f}%")
print(f"   Churn Probability: {probability[1]*100:.2f}%")

print("\n" + "=" * 60)