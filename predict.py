import pandas as pd
import joblib

# Step 1: Load the pre-trained artifacts
model = joblib.load('model/churn_model.pkl')       # Trained model
scaler = joblib.load('model/scaler.pkl')           # Scaler for numerical features
label_encoders = joblib.load('model/label_encoders.pkl')  # Encoders for categorical variables

# Step 2: Define new customer data
new_data = pd.DataFrame({
    "tenure": [12],                   # Example: customer has 12 months tenure
    "monthly_charges": [100.5],       # Monthly charges ($)
    "contract": ["Month-to-month"],   # Example categorical feature
})

# Step 3: Ensure `new_data` matches the model's training features
# Rename and add missing columns
new_data.rename(
    columns={
        "monthly_charges": "MonthlyCharges", 
        "contract": "Contract"
    },
    inplace=True
)
for col in scaler.feature_names_in_:
    if col not in new_data.columns:
        new_data[col] = 0 if col not in label_encoders else ""

new_data = new_data[scaler.feature_names_in_]

# Step 3d: Encode categorical features using LabelEncoders
for col, encoder in label_encoders.items():
    if col in new_data.columns:
        new_data[col] = new_data[col].apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )

# Step 4: Scale the data
scaled_data = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)

# Step 5: Make predictions using the trained model
prediction = model.predict(scaled_data.values)  # Suppress warning
probability = model.predict_proba(scaled_data.values)[0]  # Suppress warning

prediction_label = "Churn" if prediction[0] == 1 else "No Churn"

# Step 6: Display the result
print(f"Prediction: {prediction_label}")
print(f"Churn Probability: {probability[1]:.2f}, No Churn Probability: {probability[0]:.2f}")