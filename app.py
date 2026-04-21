from flask import Flask, request, jsonify, render_template_string, send_file, redirect, url_for
import joblib
import pandas as pd
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import tempfile
import os

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

app = Flask(__name__)

model = joblib.load('model/churn_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

CHOICES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

TOOLTIPS = {
    "gender": "Customer's gender.",
    "SeniorCitizen": "Senior citizen? (1: Yes, 0: No)",
    "Partner": "Has a partner?",
    "Dependents": "Has dependents?",
    "tenure": "Customer tenure in months.",
    "PhoneService": "Phone service included?",
    "MultipleLines": "Has multiple lines?",
    "InternetService": "Type of internet service.",
    "OnlineSecurity": "Online security enabled?",
    "OnlineBackup": "Online backup enabled?",
    "DeviceProtection": "Device protection enabled?",
    "TechSupport": "Tech support included?",
    "StreamingTV": "Streaming TV subscribed?",
    "StreamingMovies": "Streaming movies subscribed?",
    "Contract": "Contract type.",
    "PaperlessBilling": "Paperless billing?",
    "PaymentMethod": "Payment method.",
    "MonthlyCharges": "Monthly charges.",
    "TotalCharges": "Total charges."
}

DOWNLOAD_SAMPLE = """gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,actual
Female,0,Yes,No,12,Yes,No,Fiber optic,No,Yes,Yes,No,Yes,No,Month-to-month,Yes,Electronic check,85.45,1025.2,No Churn
Male,1,No,No,24,Yes,Yes,DSL,Yes,No,No,Yes,No,Yes,Two year,No,Credit card (automatic),65.90,1583.8,No Churn
Female,0,Yes,Yes,6,No,No phone service,No,No,No,No,No,No,Month-to-month,Yes,Bank transfer (automatic),42.35,210.1,Churn
Male,1,No,No,1,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,99.99,99.99,Churn
Female,0,Yes,No,60,Yes,Yes,DSL,Yes,Yes,No,Yes,Yes,Yes,Two year,Yes,Bank transfer (automatic),56.45,3450.0,No Churn
"""

_single_form = """
<form method="post">
  <div class="section"><h2>Personal Details</h2>
    <label for="gender">Gender</label><div class="tooltip">{{tooltips['gender']}}</div>
    <select name="gender" id="gender" required>{%for opt in choices['gender']%}<option value="{{opt}}">{{opt}}</option>{%endfor%}</select>
    <label for="SeniorCitizen">Senior Citizen</label><div class="tooltip">{{tooltips['SeniorCitizen']}}</div>
    <select name="SeniorCitizen" id="SeniorCitizen" required><option value="0">No</option><option value="1">Yes</option></select>
    <label for="Partner">Partner</label><div class="tooltip">{{tooltips['Partner']}}</div>
    <select name="Partner" id="Partner" required>{%for opt in choices['Partner']%}<option value="{{opt}}">{{opt}}</option>{%endfor%}</select>
    <label for="Dependents">Dependents</label><div class="tooltip">{{tooltips['Dependents']}}</div>
    <select name="Dependents" id="Dependents" required>{%for opt in choices['Dependents']%}<option value="{{opt}}">{{opt}}</option>{%endfor%}</select>
  </div>
  <div class="section"><h2>Services</h2>
    {%for svc in ['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']%}
    <label for="{{svc}}">{{svc.replace('_',' ').replace('Service',' Service').replace('DeviceProtection','Device Protection').replace('TechSupport','Tech Support').replace('StreamingTV','Streaming TV').replace('StreamingMovies','Streaming Movies').title()}}</label><div class="tooltip">{{tooltips[svc]}}</div>
    <select name="{{svc}}" id="{{svc}}" required>{%for opt in choices[svc]%}<option value="{{opt}}">{{opt}}</option>{%endfor%}</select>
    {%endfor%}
  </div>
  <div class="section"><h2>Contract & Billing</h2>
    <label for="Contract">Contract</label><div class="tooltip">{{tooltips['Contract']}}</div>
    <select name="Contract" id="Contract" required>{%for opt in choices['Contract']%}<option value="{{opt}}">{{opt}}</option>{%endfor%}</select>
    <label for="PaperlessBilling">Paperless Billing</label><div class="tooltip">{{tooltips['PaperlessBilling']}}</div>
    <select name="PaperlessBilling" id="PaperlessBilling" required>{%for opt in choices['PaperlessBilling']%}<option value="{{opt}}">{{opt}}</option>{%endfor%}</select>
    <label for="PaymentMethod">Payment Method</label><div class="tooltip">{{tooltips['PaymentMethod']}}</div>
    <select name="PaymentMethod" id="PaymentMethod" required>{%for opt in choices['PaymentMethod']%}<option value="{{opt}}">{{opt}}</option>{%endfor%}</select>
    <label for="tenure">Tenure (months)</label><div class="tooltip">{{tooltips['tenure']}}</div>
    <input type="number" name="tenure" step="1" min="0" required>
    <label for="MonthlyCharges">Monthly Charges</label><div class="tooltip">{{tooltips['MonthlyCharges']}}</div>
    <input type="number" name="MonthlyCharges" step="0.01" min="0" required>
    <label for="TotalCharges">Total Charges</label><div class="tooltip">{{tooltips['TotalCharges']}}</div>
    <input type="number" name="TotalCharges" step="0.01" min="0" required>
  </div>
  <input class="button" type="submit" value="Predict">
</form>
"""

def match_feature_names(df, expected_names):
    mapping = {}
    lower_expected = {e.lower(): e for e in expected_names}
    for col in df.columns:
        norm_col = col.replace("_", "").replace(" ", "").lower()
        for expected in expected_names:
            norm_expected = expected.replace("_", "").replace(" ", "").lower()
            if norm_col == norm_expected:
                mapping[col] = expected
                break
        else:
            if col.lower() in lower_expected:
                mapping[col] = lower_expected[col.lower()]
    return df.rename(columns=mapping)

def decode_categoricals(df):
    """Decodes label-encoded columns back to human-readable format."""
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.inverse_transform(df[col].astype(int))
            except Exception:
                pass
    return df

def smart_encode_column(series, encoder, colname):
    mapping_dict = {v.lower(): i for i, v in enumerate(encoder.classes_)}
    def safe_transform(val):
        try:
            return encoder.transform([val])[0]
        except Exception:
            sval = str(val).strip().lower()
            if sval in mapping_dict:
                return mapping_dict[sval]
            if sval in ["yes", "1", "true", "churn"]:
                return mapping_dict.get("yes", mapping_dict.get("churn", 1))
            if sval in ["no", "0", "false", "no churn", "nochurn"]:
                return mapping_dict.get("no", mapping_dict.get("no churn", 0))
            return -1
    return series.apply(safe_transform)

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0 if col not in label_encoders else ""
    df = df[scaler.feature_names_in_]
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = smart_encode_column(df[col], encoder, col)
    scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return scaled

HTML_MAIN = """
<!doctype html>
<html>
<head>
  <title>Churn Predictor Suite</title>
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;400&family=Roboto&display=swap" rel="stylesheet">
  <style>
    body {background: linear-gradient(120deg,#f3f6fb 80%, #e6ecfa 100%); font-family: 'Roboto', Arial, sans-serif; margin:0; min-height:100vh;}
    .container {max-width:950px; margin:35px auto; background:#fff; border-radius:22px; padding:38px 45px 24px 45px; box-shadow:0 7px 44px #00286012,0 1px 7px #2360a514;}
    .tabbar {display:flex; gap:24px; margin-bottom:32px;}
    .tab {font-family:'Montserrat',Arial,sans-serif; font-size:1.13em; color:#1950a6; font-weight:600; text-decoration:none; padding:12px 28px; border-radius:15px 15px 0 0; background:#f3f6fb; border-bottom:2.5px solid #e7eeff;transition:.18s; }
    .tab.active {background:#fff; border-bottom:2.5px solid #227FC4; color:#227FC4;}
    .tab:hover {color:#227FC4;}
    h1 {font-size:2.1em; font-family:'Montserrat'; letter-spacing:1px;color:#1565c0;margin-top:0;margin-bottom:16px;}
    .section {margin-bottom:32px;padding:13px 19px 11px 19px; background:#fafdff; border-radius:13px; box-shadow:0 1px 8px #eaf4fdad;}
    h2 {color:#153963;font-size:1.06em;font-family:'Montserrat';margin-top:25px;margin-bottom:2px; }
    label {font-weight:600; margin:13px 0 4px 0; color:#1565c0;font-size:1.01em;}
    .tooltip {font-size:0.9em;color:#638bb1;margin-bottom:2px; padding-left:3px;}
    select, input[type=number], input[type=text] {width:98%;padding:9px 8px;margin-bottom:8px; border-radius:5px;border:1.2px solid #bbdaefaf; font-size:1em; background:#f0f4fb; transition:.17s;}
    select:focus, input:focus {border-color:#227FC4; box-shadow:0 0 0 2px #bedafd73;outline:none;}
    .button {background:linear-gradient(92deg,#227FC4,#2998df 94%);color:#fff;padding:14px 0;border:0; border-radius:8px;width:100%;font-weight:600;font-size:1.05em;cursor:pointer;box-shadow:0 1px 7px #41caf71b;margin-top:16px; transition:.16s;}
    .button:hover {background:linear-gradient(93deg,#175891 9%,#238ee0 90%);}
    .resultbox {background:#edfbec; margin:auto; margin-top:36px; padding:28px 23px 18px 23px; max-width:410px; border:2.5px solid #89e8bbaf; border-radius:13px; text-align:center; font-size:1.13em; color:#196824; font-family:'Montserrat'; font-weight:600; box-shadow:0 1px 7px #5ecea518;}
    .resultbox.negative {background:#fff1f1; color:#a72525; border-color:#efc9bab5;}
    table {margin-top:24px;border-collapse: collapse; width:98%; font-size:1em;}
    th,td {border:1.2px solid #f1eaff;padding:7px 13px;text-align:left;}
    th {background:#fafdff;color:#1950a6;font-weight:700;}
    .analytic-plot {margin:28px auto 8px auto; display:block; max-width:390px;}
    .msg {padding:23px;background:#e8f5fa; border-radius:12px;max-width:440px;margin:25px auto 14px auto;color:#196824;text-align:center;font-size:1.09em;}
    .file-upload {margin-bottom:22px;}
    .download-link {display:inline-block;background:#227FC4;color:#fff; padding:8px 24px;border-radius:7px;text-decoration:none;margin-top:14px;}
    .download-link:hover {background:#135480;}
    .sample-link {display:inline-block; background:#e2eafc; color:#227fc4; padding:7px 22px; border-radius:6px; text-decoration:none; margin-left:12px;}
    .sample-link:hover {background:#bedafd;}
    @media (max-width:900px){.container{padding:3vw 1vw;}}
  </style>
</head>
<body>
  <div class="container">
    <h1>Churn Predictor Suite</h1>
    <div class="tabbar">
      <a class="tab {% if tab=='single' %}active{% endif %}" href="{{ url_for('home') }}">Single Prediction</a>
      <a class="tab {% if tab=='batch' %}active{% endif %}" href="{{ url_for('batch') }}">Batch Prediction</a>
      <a class="tab {% if tab=='analytics' %}active{% endif %}" href="{{ url_for('analytics') }}">Analytics</a>
    </div>
    {{ body | safe }}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    body = ""
    prediction = prob_churn = prob_no_churn = None
    if request.method == "POST":
        form_data = {feat: request.form[feat] for feat in FEATURES}
        form_data["SeniorCitizen"] = int(form_data["SeniorCitizen"])
        form_data["tenure"] = int(form_data["tenure"])
        form_data["MonthlyCharges"] = float(form_data["MonthlyCharges"])
        form_data["TotalCharges"] = float(form_data["TotalCharges"])
        scaled = preprocess_input(form_data)
        pred = model.predict(scaled.values)
        proba = model.predict_proba(scaled.values)[0]
        prediction = "Churn" if pred[0] == 1 else "No Churn"
        prob_churn = "%.2f" % proba[1]
        prob_no_churn = "%.2f" % proba[0]
        box_class = "negative" if pred[0] == 1 else ""
        body += f'''
        <div class="resultbox {box_class}">
          <b>Prediction:</b> {prediction}<br>
          <b>Churn probability:</b> {prob_churn}<br>
          <b>No Churn probability:</b> {prob_no_churn}
        </div>
        <form method="get"><button class="predict-again" type="submit">Predict Again</button></form>
        '''
        return render_template_string(HTML_MAIN, tab='single', body=body, tooltips=TOOLTIPS, features=FEATURES, choices=CHOICES)
    form_html = render_template_string(_single_form, tooltips=TOOLTIPS, features=FEATURES, choices=CHOICES)
    body += form_html
    return render_template_string(HTML_MAIN, tab='single', body=body, tooltips=TOOLTIPS, features=FEATURES, choices=CHOICES)

_batch_form = '''
<form method="POST" enctype="multipart/form-data" class="file-upload">
  <input type="file" name="file" required accept=".csv" style="margin-bottom:10px;">
  <button class="button" type="submit">Upload & Predict</button>
  <a class="sample-link" href="/download_sample">Download Sample CSV</a>
</form>
'''

BATCH_RESULTS = {}
@app.route("/batch", methods=["GET", "POST"])
def batch():
    global BATCH_RESULTS
    msg = ""
    missing_cols = []
    if request.method == "POST":
        if 'file' not in request.files or request.files['file'].filename == '':
            msg = "<div class='msg'>Please select a CSV file to upload.</div>"
        else:
            file = request.files['file']
            try:
                df = pd.read_csv(file)
                # Flexibly rename input headers to match model's expected features
                df = match_feature_names(df, FEATURES + ["actual", "ground_truth", "label", "y_true"])
                gt_col = None
                for gt in ["actual", "ground_truth", "label", "y_true"]:
                    if gt in df.columns:
                        gt_col = gt
                        break
                for col in FEATURES:
                    if col not in df.columns:
                        missing_cols.append(col)
                if missing_cols:
                    raise Exception(f"Missing required columns: {', '.join(missing_cols)}<br> See sample for correct headers.")
                # Only keep the right columns
                final_cols = FEATURES + ([gt_col] if gt_col else [])
                df = df[[col for col in final_cols if col in df.columns]]

                # ----- INDUSTRY-LEVEL: ENCODING & FORMATTING -----
                # Label encode all categoricals according to label_encoders dictionary, robust to 'yes'/'no', etc.
                for col, encoder in label_encoders.items():
                    if col in df.columns:
                        df[col] = smart_encode_column(df[col], encoder, col)
                # For numeric/continuous features, robust conversion
                for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                # For "SeniorCitizen", ensure numeric (also, don't cast if already int after encoding)
                if "SeniorCitizen" in df.columns:
                    if str(df["SeniorCitizen"].dtypes).startswith("int") or str(df["SeniorCitizen"].dtypes).startswith("float"):
                        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
                    else:
                        df["SeniorCitizen"] = df["SeniorCitizen"].map(lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0)
                
                if df[["tenure", "MonthlyCharges", "TotalCharges"]].isnull().any().any():
                    raise Exception("Some rows have missing or invalid numeric values in tenure/MonthlyCharges/TotalCharges. Please clean your CSV.")

                # ----- Scaling and Prediction -----
                df_scaled = pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES)
                preds = model.predict(df_scaled.values)
                proba = model.predict_proba(df_scaled.values)
                df['prediction'] = ["Churn" if y==1 else "No Churn" for y in preds]
                df['churn_prob'] = [round(float(p[1]),2) for p in proba]
                df['no_churn_prob'] = [round(float(p[0]),2) for p in proba]

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                df.to_csv(tmp.name, index=False)
                BATCH_RESULTS = {"df": df.copy(), "csv_path": tmp.name}
                msg = "<div class='msg'>Batch prediction complete. <a class='download-link' href='/download_batch'>Download results CSV</a></div>"
                # Display table preview for first 5 results
                preview = decode_categoricals(df.head(5)).to_html(index=False, classes="small", border=0)
                msg += "<h2>Sample Results (top 5):</h2>" + preview
            except Exception as e:
                msg = f"<div class='msg'>Upload failed: {e}</div>"
    body = _batch_form + msg
    return render_template_string(HTML_MAIN, tab='batch', body=body, tooltips=TOOLTIPS, features=FEATURES, choices=CHOICES)


@app.route("/download_batch")
def download_batch():
    global BATCH_RESULTS
    fpath = BATCH_RESULTS.get("csv_path")
    if not fpath:
        return redirect(url_for("batch"))
    return send_file(fpath, as_attachment=True, download_name="churn_predictions.csv")

@app.route("/download_sample")
def download_sample():
    return DOWNLOAD_SAMPLE, {
        "Content-Disposition": "attachment;filename=sample_batch.csv",
        "Content-type": "text/csv"
    }

@app.route("/analytics")
def analytics():
    global BATCH_RESULTS
    df = BATCH_RESULTS.get("df")
    body = ""
    if df is None or df.empty:
        body += "<div class='msg'>No batch results available. Please upload a CSV in 'Batch Prediction' first.</div>"
    else:
        df_viz = decode_categoricals(df.copy())
        # Pie chart: Churn / No Churn
        img_io = io.BytesIO()
        plt.figure(figsize=(4,4))
        df_viz['prediction'].value_counts().plot.pie(autopct='%.1f%%', startangle=90, colors=['#2ac07d','#e589a3']);
        plt.title("Churn Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(img_io, format='png', bbox_inches="tight")
        plt.close()
        img_io.seek(0)
        img_data = base64.b64encode(img_io.read()).decode('utf-8')
        body += '<h2>Overall Churn Distribution</h2>'
        body += f'<img class="analytic-plot" src="data:image/png;base64,{img_data}"/>'

        # Bar: Churn by Contract
        plt.figure(figsize=(5,3))
        sns.countplot(data=df_viz, x="Contract", hue="prediction", palette=["#2ac07d","#e589a3"])
        plt.title("Churn by Contract Type"); plt.ylabel("Count")
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close(); buf.seek(0)
        img_data2 = base64.b64encode(buf.read()).decode('utf-8')
        body += '<h2>Churn by Contract Type</h2>'
        body += f'<img class="analytic-plot" src="data:image/png;base64,{img_data2}"/>'

        # Grouped summary
        summary = df_viz.groupby("prediction").agg({"tenure":"mean","MonthlyCharges":"mean","TotalCharges":"mean"}).round(2).reset_index()
        body += "<h2>Summary by Prediction</h2>" + summary.to_html(index=False,border=0)

        # Classification report, if ground-truth present
        gt_col = None
        for col in df.columns:
            if str(col).lower() in ["actual", "ground_truth", "label", "y_true"]:
                gt_col = col
                break
        if gt_col:
            y_true = df[gt_col].apply(lambda x: 1 if str(x).strip().lower() in ["churn", "1", "yes"] else 0)
            y_pred = df["prediction"].apply(lambda x: 1 if str(x).strip().lower().startswith("churn") else 0)
            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            body += f"""
              <h2>Classification Metrics</h2>
              <table>
              <tr><th>F1 Score</th><th>Precision</th><th>Recall</th><th>Accuracy</th></tr>
              <tr>
                <td>{f1:.2f}</td>
                <td>{prec:.2f}</td>
                <td>{rec:.2f}</td>
                <td>{acc:.2f}</td>
              </tr>
              </table>
              <h3>Confusion Matrix</h3>
              <pre>{cm}</pre>
            """
        else:
            body += "<div class='msg'>Add an <b>actual</b> or <b>ground_truth</b> column in your CSV to see classification metrics like F1 and accuracy here!</div>"

    return render_template_string(HTML_MAIN, tab='analytics', body=body, tooltips=TOOLTIPS, features=FEATURES, choices=CHOICES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    data["SeniorCitizen"] = int(data["SeniorCitizen"])
    data["tenure"] = int(data["tenure"])
    data["MonthlyCharges"] = float(data["MonthlyCharges"])
    data["TotalCharges"] = float(data["TotalCharges"])
    scaled = preprocess_input(data)
    pred = model.predict(scaled.values)
    proba = model.predict_proba(scaled.values)[0]
    return jsonify({
      "prediction": "Churn" if pred[0] == 1 else "No Churn",
      "probabilities": {
          "churn": round(float(proba[1]), 2),
          "no_churn": round(float(proba[0]), 2)
      }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)