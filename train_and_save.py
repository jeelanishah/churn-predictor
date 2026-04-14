import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=" * 60)
print("🚀 CHURN PREDICTOR - MODEL TRAINING")
print("=" * 60)

# Step 1: Load data
print("\n📊 Loading data...")
df = pd.read_csv('../data/Churn_Modelling.csv')
print(f"✅ Data loaded! Shape: {df.shape}")

# Step 2: Prepare features
print("\n📊 Preparing features...")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Convert target to binary (0 = No, 1 = Yes)
y = (y == 'Yes').astype(int)

print(f"✅ Features: {X.shape[1]}")
print(f"✅ Samples: {X.shape[0]}")

# Step 3: Encode categorical features
print("\n📊 Encoding categorical features...")

categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"✅ Encoded {len(categorical_cols)} categorical columns")

# Save feature names
feature_names = X.columns.tolist()

# Step 4: Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# Step 5: Scale features
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features scaled!")

# Step 6: Train model
print("\n📊 Training model...")

# Use Gradient Boosting for better accuracy
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    verbose=0
)

model.fit(X_train_scaled, y_train)
print(f"✅ Model trained!")

# Step 7: Evaluate model
print("\n📊 Evaluating model...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"✅ Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Precision: {precision * 100:.2f}%")
print(f"✅ Recall: {recall * 100:.2f}%")
print(f"✅ F1-Score: {f1 * 100:.2f}%")

# Step 8: Save model and artifacts
print("\n📊 Saving model...")

os.makedirs('model', exist_ok=True)

with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved: model/churn_model.pkl")

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved: model/scaler.pkl")

with open('model/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(f"✅ Feature names saved: model/feature_names.pkl")

with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✅ Label encoders saved: model/label_encoders.pkl")

# Target encoder (for converting predictions back)
target_encoder = {'No': 0, 'Yes': 1}
with open('model/target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
print(f"✅ Target encoder saved: model/target_encoder.pkl")

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"✅ Model Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Files saved in model/ folder")
print("=" * 60)