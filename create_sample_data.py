import pandas as pd
import numpy as np

# Create sample churn data
np.random.seed(42)

data = {
    'gender': np.random.choice(['Male', 'Female'], 35),
    'SeniorCitizen': np.random.choice([0, 1], 35),
    'Partner': np.random.choice(['Yes', 'No'], 35),
    'Dependents': np.random.choice(['Yes', 'No'], 35),
    'tenure': np.random.randint(1, 72, 35),
    'PhoneService': np.random.choice(['Yes', 'No'], 35),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], 35),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 35),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 35),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], 35),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 35),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 35),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 35),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], 35),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 35),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], 35),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 35),
    'MonthlyCharges': np.random.uniform(20, 120, 35),
    'TotalCharges': np.random.uniform(100, 5000, 35),
    'Churn': np.random.choice(['Yes', 'No'], 35)
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('../data/Churn_Modelling.csv', index=False)
print(f"✅ Sample data created! Shape: {df.shape}")
print(f"✅ File saved to: ../data/Churn_Modelling.csv")
print(f"\nFirst few rows:")
print(df.head())