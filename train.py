import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os

# Step 1: Load the data
try:
    df = pd.read_csv('data/churn_data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'data/churn_data.csv' was not found. Ensure the path is correct.")
    exit()

# Step 2: Remove customerID and separate labels
if 'customerID' in df.columns:
    X = df.drop(['customerID', 'Churn'], axis=1)
else:
    X = df.drop(['Churn'], axis=1)
y = df['Churn']

# Step 3: Preprocess the data
# 3.1 Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')  # Replace missing values with the most frequent value
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 3.2 Encode categorical columns to numeric
encoders = {}  # Dictionary to save encoding mappings
for col in X.columns:
    if X[col].dtype == "object":
        X[col], encoders[col] = X[col].factorize()

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create the `model` directory (if it doesn't already exist)
os.makedirs('model', exist_ok=True)

# Step 6: Serialize the model and preprocessing artifacts
joblib.dump(model, 'model/model.pkl')  # Save trained model
joblib.dump(imputer, 'model/imputer.pkl')  # Save the imputer for handling missing values
joblib.dump(encoders, 'model/encoders.pkl')  # Save the encoding mappings

print("Model and preprocessing artifacts saved to the 'model/' directory!")