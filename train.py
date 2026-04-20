import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Load the data
df = pd.read_csv('data/churn_data.csv')

# 2. Remove customerID and separate labels
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# 3. Encode string (categorical) columns to numbers
for col in X.columns:
    if X[col].dtype == "object":
        X[col], _ = X[col].factorize()

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Save the model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')

print("Model trained and saved to model/model.pkl")