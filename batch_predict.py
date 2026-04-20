import pandas as pd
import requests

# Load your CSV file
df = pd.read_csv("test_customers.csv")

url = "http://127.0.0.1:8000/predict"
results = []

for _, row in df.iterrows():
    payload = row.to_dict()

    # Auto-convert SeniorCitizen from 'Yes'/'No' to 1/0
    if "SeniorCitizen" in payload:
        val = str(payload["SeniorCitizen"]).strip().lower()
        if val == "yes":
            payload["SeniorCitizen"] = 1
        elif val == "no":
            payload["SeniorCitizen"] = 0

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        pred = response.json()
        results.append({
            "Prediction": pred.get("prediction",""),
            "Churn_Probability": pred.get("probability",""),
            "Risk_Level": pred.get("risk_level",""),
            "Error": ""
        })
    except Exception as e:
        results.append({
            "Prediction": "",
            "Churn_Probability": "",
            "Risk_Level": "",
            "Error": str(e)
        })

# Combine original and results, and save
output = pd.concat([df, pd.DataFrame(results)], axis=1)
output.to_csv("batch_predictions.csv", index=False)
print("Batch prediction done! Results saved to batch_predictions.csv")