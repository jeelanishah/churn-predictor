"""
Unit tests for API predictions (api.py ChurnPredictor class).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import ChurnPredictor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


LOW_RISK_CUSTOMER = {
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

HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
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
    "MonthlyCharges": 89.99,
    "TotalCharges": 150.00,
}


# ---------------------------------------------------------------------------
# Health-check tests
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_returns_dict(self, predictor):
        result = predictor.health_check()
        assert isinstance(result, dict)

    def test_health_check_model_loaded(self, predictor):
        result = predictor.health_check()
        assert result["model_loaded"] is True

    def test_health_check_status_field(self, predictor):
        result = predictor.health_check()
        assert "status" in result

    def test_health_check_version_field(self, predictor):
        result = predictor.health_check()
        assert "version" in result


# ---------------------------------------------------------------------------
# Single prediction tests
# ---------------------------------------------------------------------------

class TestSinglePrediction:
    def test_predict_returns_success(self, predictor):
        result = predictor.predict(LOW_RISK_CUSTOMER)
        assert result["status"] == "success"

    def test_predict_has_required_fields(self, predictor):
        result = predictor.predict(LOW_RISK_CUSTOMER)
        for field in ("prediction", "probability", "risk_level"):
            assert field in result, f"Missing field: {field}"

    def test_predict_low_risk_customer(self, predictor):
        result = predictor.predict(LOW_RISK_CUSTOMER)
        assert result["status"] == "success"
        assert result["prediction"] == 0
        assert result["probability"] < 0.4
        assert "LOW RISK" in result["risk_level"]

    def test_predict_high_risk_customer(self, predictor):
        result = predictor.predict(HIGH_RISK_CUSTOMER)
        assert result["status"] == "success"
        assert result["prediction"] == 1
        assert result["probability"] >= 0.4

    def test_predict_probability_in_range(self, predictor):
        result = predictor.predict(LOW_RISK_CUSTOMER)
        prob = result["probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_missing_feature_returns_error(self, predictor):
        bad_data = dict(LOW_RISK_CUSTOMER)
        del bad_data["tenure"]
        result = predictor.predict(bad_data)
        assert result["status"] == "error"

    def test_predict_invalid_value_returns_error(self, predictor):
        bad_data = dict(LOW_RISK_CUSTOMER)
        bad_data["tenure"] = "not_a_number"
        result = predictor.predict(bad_data)
        assert result["status"] == "error"

    def test_predict_empty_input_returns_error(self, predictor):
        result = predictor.predict({})
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Risk-level tests
# ---------------------------------------------------------------------------

class TestRiskLevels:
    def test_high_risk_threshold(self, predictor):
        level = predictor._risk_level(0.75)
        assert "HIGH RISK" in level

    def test_medium_risk_threshold(self, predictor):
        level = predictor._risk_level(0.55)
        assert "MEDIUM RISK" in level

    def test_low_risk_threshold(self, predictor):
        level = predictor._risk_level(0.20)
        assert "LOW RISK" in level

    def test_boundary_high(self, predictor):
        level = predictor._risk_level(0.70)
        assert "HIGH RISK" in level

    def test_boundary_medium(self, predictor):
        level = predictor._risk_level(0.40)
        assert "MEDIUM RISK" in level
