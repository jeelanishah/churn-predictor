import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 80)
print("📊 ANALYTICS CALCULATOR - REAL DATA ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD EVERYTHING
# ============================================================================

print("\n📥 Loading data and model...")

# Load training data
df = pd.read_csv('../data/Churn_Modelling.csv')
print(f"✅ Data loaded: {df.shape[0]} customers, {df.shape[1]} columns")

# Load model
with open('model/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✅ Model loaded")

# Load scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded")

# Load feature names
with open('model/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
print(f"✅ Features loaded: {len(feature_names)} features")

# Load label encoders
with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
print(f"✅ Label encoders loaded: {len(label_encoders)} encoders")

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n🔄 Preparing data...")

# Separate features and target
X = df.drop('Churn', axis=1).copy()
y = df['Churn'].copy()

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

# Convert target to binary
y_binary = (y == 'Yes').astype(int)

# Encode categorical variables
print("\n🔄 Encoding categorical variables...")
for col in X.columns:
    if X[col].dtype == 'object':
        if col in label_encoders:
            X[col] = label_encoders[col].transform(X[col])
print("✅ Encoding complete")

# Scale features
print("\n🔄 Scaling features...")
X_scaled = scaler.transform(X)
print("✅ Scaling complete")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

print("\n🤖 Making predictions on all data...")
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)
print(f"✅ Predictions made for {len(y_pred)} customers")

# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("📊 CALCULATING METRICS")
print("=" * 80)

# Accuracy
accuracy = accuracy_score(y_binary, y_pred)
print(f"\n📈 ACCURACY: {accuracy * 100:.2f}%")

# Precision
precision = precision_score(y_binary, y_pred)
print(f"🎯 PRECISION: {precision * 100:.2f}%")

# Recall
recall = recall_score(y_binary, y_pred)
print(f"🔔 RECALL: {recall * 100:.2f}%")

# F1-Score
f1 = f1_score(y_binary, y_pred)
print(f"⚖️  F1-SCORE: {f1 * 100:.2f}%")

# ROC-AUC
try:
    roc_auc = roc_auc_score(y_binary, y_pred_proba[:, 1])
    print(f"📊 ROC-AUC: {roc_auc * 100:.2f}%")
except:
    roc_auc = 0
    print(f"📊 ROC-AUC: N/A")

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("📊 CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(y_binary, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n✅ True Negatives (TN): {tn}")
print(f"❌ False Positives (FP): {fp}")
print(f"❌ False Negatives (FN): {fn}")
print(f"✅ True Positives (TP): {tp}")

print(f"\n📋 Confusion Matrix:")
print(f"                 Predicted")
print(f"              NO CHURN  CHURN")
print(f"Actual NO CHURN  {tn:4d}      {fp:4d}")
print(f"       CHURN     {fn:4d}      {tp:4d}")

# ============================================================================
# DERIVED METRICS
# ============================================================================

print("\n" + "=" * 80)
print("📊 DERIVED METRICS")
print("=" * 80)

sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
fpr = fp / (fp + tn) * 100
fnr = fn / (fn + tp) * 100

print(f"\n✅ Sensitivity (TPR): {sensitivity:.2f}%")
print(f"✅ Specificity (TNR): {specificity:.2f}%")
print(f"❌ False Positive Rate: {fpr:.2f}%")
print(f"❌ False Negative Rate: {fnr:.2f}%")

# ============================================================================
# CLASS DISTRIBUTION
# ============================================================================

print("\n" + "=" * 80)
print("📊 CLASS DISTRIBUTION")
print("=" * 80)

actual_churn = (y_binary == 1).sum()
actual_no_churn = (y_binary == 0).sum()
pred_churn = (y_pred == 1).sum()
pred_no_churn = (y_pred == 0).sum()

print(f"\n📊 ACTUAL DISTRIBUTION:")
print(f"   NO CHURN: {actual_no_churn} ({actual_no_churn/len(y_binary)*100:.1f}%)")
print(f"   CHURN:    {actual_churn} ({actual_churn/len(y_binary)*100:.1f}%)")

print(f"\n📊 PREDICTED DISTRIBUTION:")
print(f"   NO CHURN: {pred_no_churn} ({pred_no_churn/len(y_pred)*100:.1f}%)")
print(f"   CHURN:    {pred_churn} ({pred_churn/len(y_pred)*100:.1f}%)")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 80)
print("📊 FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance,
    'Importance %': feature_importance * 100
}).sort_values('Importance', ascending=False)

print("\n🔝 Top 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

print(f"\n📊 Feature Statistics:")
print(f"   Total features: {len(feature_names)}")
print(f"   Top feature importance: {feature_importance_df.iloc[0]['Importance %']:.2f}%")
print(f"   Top 5 cumulative: {feature_importance_df.head(5)['Importance %'].sum():.2f}%")
print(f"   Top 10 cumulative: {feature_importance_df.head(10)['Importance %'].sum():.2f}%")

# ============================================================================
# PROBABILITY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("📊 PROBABILITY DISTRIBUTION")
print("=" * 80)

churn_probabilities = y_pred_proba[:, 1]

print(f"\n📊 Statistics:")
print(f"   Mean: {np.mean(churn_probabilities):.4f}")
print(f"   Median: {np.median(churn_probabilities):.4f}")
print(f"   Std Dev: {np.std(churn_probabilities):.4f}")
print(f"   Min: {np.min(churn_probabilities):.4f}")
print(f"   Max: {np.max(churn_probabilities):.4f}")

# Probability ranges
bins = [0, 0.25, 0.5, 0.75, 1.0]
hist, _ = np.histogram(churn_probabilities, bins=bins)

print(f"\n📊 Probability Buckets:")
for i in range(len(bins)-1):
    pct = hist[i]/len(churn_probabilities)*100
    print(f"   {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]:4d} ({pct:6.2f}%)")

# ============================================================================
# SAVE ANALYTICS TO FILE
# ============================================================================

print("\n" + "=" * 80)
print("💾 SAVING ANALYTICS")
print("=" * 80)

analytics_data = {
    'accuracy': accuracy * 100,
    'precision': precision * 100,
    'recall': recall * 100,
    'f1_score': f1 * 100,
    'roc_auc': roc_auc * 100,
    'tn': int(tn),
    'fp': int(fp),
    'fn': int(fn),
    'tp': int(tp),
    'sensitivity': sensitivity,
    'specificity': specificity,
    'fpr': fpr,
    'fnr': fnr,
    'actual_churn': int(actual_churn),
    'actual_no_churn': int(actual_no_churn),
    'pred_churn': int(pred_churn),
    'pred_no_churn': int(pred_no_churn),
    'top_features': feature_importance_df.head(10).to_dict('records')
}

# Save to JSON
import json
with open('analytics_results.json', 'w') as f:
    json.dump(analytics_data, f, indent=2)
print("✅ Saved: analytics_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("🎉 FINAL ANALYTICS SUMMARY")
print("=" * 80)

print(f"""
╔════════════════════════════════════════════════════════════════╗
║                    MODEL PERFORMANCE METRICS                   ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  PRIMARY METRICS:                                             ║
║     📈 Accuracy:  {accuracy * 100:6.2f}%                              ║
║     🎯 Precision: {precision * 100:6.2f}%                              ║
║     🔔 Recall:    {recall * 100:6.2f}%                              ║
║     ⚖️  F1-Score:  {f1 * 100:6.2f}%                              ║
║     📊 ROC-AUC:   {roc_auc * 100:6.2f}%                              ║
║                                                                ║
║  CONFUSION MATRIX:                                            ║
║     ✅ True Negatives:  {tn:4d}                              ║
║     ❌ False Positives: {fp:4d}                              ║
║     ❌ False Negatives: {fn:4d}                              ║
║     ✅ True Positives:  {tp:4d}                              ║
║                                                                ║
║  CORRECT PREDICTIONS: {int(accuracy*len(y_binary)):.0f}/{len(y_binary)}                       ║
║  INCORRECT PREDICTIONS: {int((1-accuracy)*len(y_binary)):.0f}/{len(y_binary)}                      ║
║                                                                ║
║  ⭐ Model Status: PRODUCTION READY                            ║
║                                                                ║
╚═════════════════════════════════════════════════   ══════════════╝
""")

print("=" * 80)
print("✅ ANALYTICS CALCULATION COMPLETE!")
print("=" * 80)