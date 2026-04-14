import pandas as pd
import numpy as np

# Create a more realistic and larger dataset
np.random.seed(42)

# Create 500 realistic customer records
data = {
    'gender': np.random.choice(['Male', 'Female'], 500),
    'SeniorCitizen': np.random.choice([0, 1], 500, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], 500),
    'Dependents': np.random.choice(['Yes', 'No'], 500),
    'tenure': np.random.randint(1, 72, 500),
    'PhoneService': np.random.choice(['Yes', 'No'], 500),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], 500),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 500),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 500),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], 500),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 500),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 500),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 500),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], 500),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 500, p=[0.55, 0.22, 0.23]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], 500),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 500),
    'MonthlyCharges': np.random.uniform(20, 120, 500),
    'TotalCharges': np.random.uniform(100, 8000, 500),
}

df = pd.DataFrame(data)

# Create REALISTIC churn pattern based on key factors
churn = []
for i in range(len(df)):
    churn_probability = 0.0
    
    # Strong factors for churn
    if df.loc[i, 'tenure'] < 6:  # Very new customers
        churn_probability += 0.45
    elif df.loc[i, 'tenure'] < 12:  # New customers
        churn_probability += 0.30
    
    if df.loc[i, 'Contract'] == 'Month-to-month':  # No commitment
        churn_probability += 0.35
    elif df.loc[i, 'Contract'] == 'One year':
        churn_probability += 0.10
    # Two year contract reduces churn
    
    if df.loc[i, 'InternetService'] == 'Fiber optic':  # Known issue
        churn_probability += 0.15
    
    if df.loc[i, 'OnlineSecurity'] == 'No' and df.loc[i, 'InternetService'] != 'No':
        churn_probability += 0.10
    
    if df.loc[i, 'TechSupport'] == 'No' and df.loc[i, 'InternetService'] != 'No':
        churn_probability += 0.10
    
    if df.loc[i, 'MonthlyCharges'] > 80:  # High charges
        churn_probability += 0.15
    
    # Protective factors (reduce churn)
    if df.loc[i, 'OnlineSecurity'] == 'Yes':
        churn_probability -= 0.15
    
    if df.loc[i, 'TechSupport'] == 'Yes':
        churn_probability -= 0.15
    
    if df.loc[i, 'Contract'] == 'Two year':
        churn_probability -= 0.20
    
    if df.loc[i, 'tenure'] > 36:  # Long-term customers
        churn_probability -= 0.30
    
    # Cap probability between 0 and 1
    churn_probability = max(0, min(1, churn_probability))
    
    # Make random decision based on probability
    churn.append('Yes' if np.random.random() < churn_probability else 'No')

df['Churn'] = churn

# Save to CSV
df.to_csv('../data/Churn_Modelling.csv', index=False)

print("=" * 60)
print("✅ REALISTIC DATASET CREATED!")
print("=" * 60)
print(f"✅ Total customers: {len(df)}")
print(f"✅ Churn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.1f}%")
print(f"✅ File saved to: ../data/Churn_Modelling.csv")
print(f"✅ Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nChurn distribution:")
print(df['Churn'].value_counts())