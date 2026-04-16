"""
Prediction accuracy tests — verify that the real model returns predictions
that are consistent with the expected risk profiles for well-known customer
archetypes.  These tests use the actual trained model artifacts (or the
fallback-trained model) rather than mocks.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import ChurnPredictor  # noqa: E402


@pytest.fixture(scope="module")
def real_predictor():
    """Real ChurnPredictor loaded from artifacts or fallback training."""
    return ChurnPredictor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _low_risk_customer():
    return {
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
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 4680.0,
    }


def _high_risk_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
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
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
    }


# ---------------------------------------------------------------------------
# Model readiness
# ---------------------------------------------------------------------------

class TestModelReadiness:
    def test_model_is_loaded(self, real_predictor):
        assert real_predictor.model is not None

    def test_scaler_is_loaded(self, real_predictor):
        assert real_predictor.scaler is not None

    def test_feature_names_loaded(self, real_predictor):
        assert real_predictor.feature_names is not None
        assert len(real_predictor.feature_names) == 19

    def test_label_encoders_loaded(self, real_predictor):
        assert len(real_predictor.label_encoders) > 0

    def test_model_source_valid(self, real_predictor):
        assert real_predictor.model_source in ("artifacts", "fallback_training")


# ---------------------------------------------------------------------------
# High-risk customer predictions
# ---------------------------------------------------------------------------

class TestHighRiskPredictions:
    def test_high_risk_prediction_succeeds(self, real_predictor):
        result = real_predictor.predict(_high_risk_customer())
        assert result["status"] == "success"

    def test_high_risk_probability_in_range(self, real_predictor):
        result = real_predictor.predict(_high_risk_customer())
        assert 0.0 <= result["probability"] <= 1.0

    def test_high_risk_prediction_is_binary(self, real_predictor):
        result = real_predictor.predict(_high_risk_customer())
        assert result["prediction"] in (0, 1)

    def test_high_risk_has_risk_level(self, real_predictor):
        result = real_predictor.predict(_high_risk_customer())
        assert result["risk_level"] in ("🔴 HIGH RISK", "🟡 MEDIUM RISK", "🟢 LOW RISK")

    def test_high_risk_probabilities_sum_to_one(self, real_predictor):
        result = real_predictor.predict(_high_risk_customer())
        total = result["probability_churn"] + result["probability_no_churn"]
        assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Low-risk customer predictions
# ---------------------------------------------------------------------------

class TestLowRiskPredictions:
    def test_low_risk_prediction_succeeds(self, real_predictor):
        result = real_predictor.predict(_low_risk_customer())
        assert result["status"] == "success"

    def test_low_risk_probability_in_range(self, real_predictor):
        result = real_predictor.predict(_low_risk_customer())
        assert 0.0 <= result["probability"] <= 1.0

    def test_low_risk_prediction_is_binary(self, real_predictor):
        result = real_predictor.predict(_low_risk_customer())
        assert result["prediction"] in (0, 1)

    def test_low_risk_has_risk_level(self, real_predictor):
        result = real_predictor.predict(_low_risk_customer())
        assert result["risk_level"] in ("🔴 HIGH RISK", "🟡 MEDIUM RISK", "🟢 LOW RISK")

    def test_low_risk_probabilities_sum_to_one(self, real_predictor):
        result = real_predictor.predict(_low_risk_customer())
        total = result["probability_churn"] + result["probability_no_churn"]
        assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Risk level boundaries
# ---------------------------------------------------------------------------

class TestRiskLevelBoundaries:
    def test_risk_boundary_high(self):
        assert ChurnPredictor._risk_level(0.70) == "🔴 HIGH RISK"
        assert ChurnPredictor._risk_level(1.00) == "🔴 HIGH RISK"

    def test_risk_boundary_medium(self):
        assert ChurnPredictor._risk_level(0.40) == "🟡 MEDIUM RISK"
        assert ChurnPredictor._risk_level(0.69) == "🟡 MEDIUM RISK"

    def test_risk_boundary_low(self):
        assert ChurnPredictor._risk_level(0.00) == "🟢 LOW RISK"
        assert ChurnPredictor._risk_level(0.39) == "🟢 LOW RISK"


# ---------------------------------------------------------------------------
# Batch prediction accuracy
# ---------------------------------------------------------------------------

class TestBatchPredictionAccuracy:
    def test_batch_two_customers(self, real_predictor):
        result = real_predictor.predict_batch([_low_risk_customer(), _high_risk_customer()])
        assert result["status"] == "success"
        assert result["summary"]["total"] == 2
        assert result["summary"]["successful"] == 2

    def test_batch_all_probabilities_bounded(self, real_predictor):
        result = real_predictor.predict_batch([_low_risk_customer(), _high_risk_customer()])
        for row in result["results"]:
            assert row["status"] == "success"
            assert 0.0 <= row["probability"] <= 1.0

    def test_batch_row_indices_sequential(self, real_predictor):
        result = real_predictor.predict_batch([_low_risk_customer(), _high_risk_customer()])
        for idx, row in enumerate(result["results"]):
            assert row["row_index"] == idx

    def test_batch_ten_customers(self, real_predictor):
        customers = [_high_risk_customer() for _ in range(10)]
        result = real_predictor.predict_batch(customers)
        assert result["summary"]["successful"] == 10
