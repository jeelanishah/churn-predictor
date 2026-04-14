import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

print("=" * 60)
print("🚀 CHURN PREDICTOR - MODEL TRAINING")
print("=" * 60)

# Step 1: Load data
print("\n📊 Step 1: Loading data...")
df = pd.read_csv('../../data/Churn_Modelling.csv')
print(f"✅ Data loaded! Shape: {df.shape}")

# Step 2: Explore data
print("\n🔍 Step 2: Exploring data...")
print(f"Columns: {list(df.columns)}")

# Step 3: Clean data
print("\n🧹 Step 3: Cleaning data...")
df = df.dropna()
print(f"✅ Removed NaN values. Shape: {df.shape}")

# Step 4: Prepare features
print("\n🎯 Step 4: Preparing features...")

# Drop unnecessary columns
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Encode categorical variables
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"✅ Features prepared: {X.shape[1]} features")

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Step 5: Split data
print("\n📈 Step 5: Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Step 6: Scale features
print("\n📏 Step 6: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# Step 7: Train model
print("\n🤖 Step 7: Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("✅ Model trained")

# Step 8: Evaluate model
print("\n📊 Step 8: Evaluating model...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("✅ Model Performance:")
print(f"   Training Accuracy: {train_acc*100:.2f}%")
print(f"   Testing Accuracy:  {test_acc*100:.2f}%")
print(f"   Precision:         {precision*100:.2f}%")
print(f"   Recall:            {recall*100:.2f}%")
print(f"   F1-Score:          {f1*100:.2f}%")

# Step 9: Save model artifacts
print("\n💾 Step 9: Saving model artifacts...")

# Save model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✅ churn_model.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ scaler.pkl")

# Save feature names
feature_names = list(X.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("   ✅ feature_names.pkl")

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✅ label_encoders.pkl")

# Save target encoder
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
print("   ✅ target_encoder.pkl")

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"✅ Model Accuracy: {test_acc*100:.2f}%")
print("✅ Files saved in model/ folder:")
print("   1. churn_model.pkl")
print("   2. scaler.pkl")
print("   3. feature_names.pkl")
print("   4. label_encoders.pkl")
print("   5. target_encoder.pkl")
print("=" * 60)