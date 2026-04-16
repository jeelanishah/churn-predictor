"""
Tests for churn predictions covering various customer scenarios.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import ChurnPredictor

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


# ---------------------------------------------------------------------------
# Batch prediction tests
# ---------------------------------------------------------------------------

BATCH_CUSTOMERS = [
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.0,
        "TotalCharges": 1560.0,
    },
    {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer",
        "MonthlyCharges": 45.5,
        "TotalCharges": 2730.0,
    },
]


class TestBatchPredictions:
    def test_batch_predict_returns_success(self, predictor):
        result = predictor.predict_batch(BATCH_CUSTOMERS)
        assert result["status"] == "success"

    def test_batch_predict_has_results(self, predictor):
        result = predictor.predict_batch(BATCH_CUSTOMERS)
        assert "results" in result
        assert len(result["results"]) == len(BATCH_CUSTOMERS)

    def test_batch_predict_summary(self, predictor):
        result = predictor.predict_batch(BATCH_CUSTOMERS)
        summary = result["summary"]
        assert summary["total"] == len(BATCH_CUSTOMERS)
        assert summary["successful"] + summary["failed"] == summary["total"]

    def test_batch_predict_all_rows_have_status(self, predictor):
        result = predictor.predict_batch(BATCH_CUSTOMERS)
        for row in result["results"]:
            assert "status" in row

    def test_batch_predict_empty_list(self, predictor):
        result = predictor.predict_batch([])
        assert result["status"] == "success"
        assert result["summary"]["total"] == 0

    def test_batch_predict_invalid_type(self, predictor):
        result = predictor.predict_batch("invalid")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Feature encoding tests
# ---------------------------------------------------------------------------

class TestFeatureEncoding:
    def test_numeric_senior_citizen_zero(self, predictor):
        customer = dict(BATCH_CUSTOMERS[0])
        customer["SeniorCitizen"] = 0
        result = predictor.predict(customer)
        assert result["status"] == "success"

    def test_numeric_senior_citizen_one(self, predictor):
        customer = dict(BATCH_CUSTOMERS[0])
        customer["SeniorCitizen"] = 1
        result = predictor.predict(customer)
        assert result["status"] == "success"

    def test_no_internet_service_value(self, predictor):
        customer = dict(BATCH_CUSTOMERS[0])
        customer["InternetService"] = "No"
        customer["OnlineSecurity"] = "No internet service"
        customer["OnlineBackup"] = "No internet service"
        customer["DeviceProtection"] = "No internet service"
        customer["TechSupport"] = "No internet service"
        customer["StreamingTV"] = "No internet service"
        customer["StreamingMovies"] = "No internet service"
        result = predictor.predict(customer)
        assert result["status"] == "success"

    def test_payment_method_variants(self, predictor):
        for method in ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]:
            customer = dict(BATCH_CUSTOMERS[0])
            customer["PaymentMethod"] = method
            result = predictor.predict(customer)
            assert result["status"] == "success", f"Failed for PaymentMethod={method}"

    def test_contract_variants(self, predictor):
        for contract in ["Month-to-month", "One year", "Two year"]:
            customer = dict(BATCH_CUSTOMERS[0])
            customer["Contract"] = contract
            result = predictor.predict(customer)
            assert result["status"] == "success", f"Failed for Contract={contract}"
