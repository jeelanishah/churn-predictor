"""
Integration tests — full predict flow using CSV-like data.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import ChurnPredictor

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


CSV_DATA = """gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Male,0,Yes,No,24,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,65.0,1560.0
Female,1,No,Yes,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Bank transfer,45.5,2730.0
Male,0,No,No,2,No,No phone service,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,105.5,210.0
Female,0,Yes,Yes,48,Yes,Yes,DSL,Yes,No,Yes,No,Yes,Yes,One year,Yes,Credit card,55.25,2652.0
Male,1,No,No,1,Yes,No,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Month-to-month,Yes,Mailed check,20.0,20.0
Female,0,Yes,No,36,No,No phone service,DSL,No,Yes,No,Yes,No,Yes,One year,No,Bank transfer,50.0,1800.0
Male,0,No,Yes,72,Yes,Yes,Fiber optic,Yes,Yes,Yes,Yes,Yes,Yes,Two year,No,Credit card,110.0,7920.0
Female,0,Yes,No,6,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,45.0,270.0"""


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestCSVBatchIntegration:
    @pytest.fixture
    def df(self):
        return pd.read_csv(io.StringIO(CSV_DATA))

    def test_batch_from_dataframe_succeeds(self, predictor, df):
        result = predictor.predict_batch(df)
        assert result["status"] == "success"

    def test_batch_all_rows_processed(self, predictor, df):
        result = predictor.predict_batch(df)
        assert result["summary"]["total"] == 8

    def test_batch_no_failures(self, predictor, df):
        result = predictor.predict_batch(df)
        assert result["summary"]["failed"] == 0

    def test_batch_results_have_predictions(self, predictor, df):
        result = predictor.predict_batch(df)
        for row in result["results"]:
            assert row["status"] == "success"
            assert row["prediction"] in (0, 1)
            assert 0.0 <= row["probability"] <= 1.0

    def test_high_tenure_low_churn(self, predictor, df):
        """Customers with tenure >= 48 and long contract should be low risk."""
        high_tenure_rows = df[df["tenure"] >= 48].to_dict(orient="records")
        for customer in high_tenure_rows:
            result = predictor.predict(customer)
            assert result["status"] == "success"
            assert result["prediction"] == 0, (
                f"Expected no-churn for high-tenure customer: {customer}"
            )

    def test_short_tenure_month_to_month_high_churn(self, predictor, df):
        """Customers with tenure <= 6 and month-to-month contract tend to churn."""
        risky_rows = df[
            (df["tenure"] <= 6) & (df["Contract"] == "Month-to-month")
        ].to_dict(orient="records")
        assert len(risky_rows) > 0, "Test data should have at least one risky customer"
        for customer in risky_rows:
            result = predictor.predict(customer)
            assert result["status"] == "success"


class TestFullRoundTrip:
    def test_predict_then_risk_level_consistent(self, predictor):
        customer = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 72,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer",
            "MonthlyCharges": 65.50,
            "TotalCharges": 4716.00,
        }
        result = predictor.predict(customer)
        assert result["status"] == "success"
        prob = result["probability"]
        expected_risk = predictor._risk_level(prob)
        assert result["risk_level"] == expected_risk

    def test_model_health_before_predict(self, predictor):
        health = predictor.health_check()
        assert health["model_loaded"] is True
        # Prediction should work when model is loaded
        customer = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 72,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer",
            "MonthlyCharges": 65.50,
            "TotalCharges": 4716.00,
        }
        result = predictor.predict(customer)
        assert result["status"] == "success"
