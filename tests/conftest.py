"""
Shared pytest fixtures for Churn Predictor tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

# Ensure repository root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import ChurnPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyScaler:
    """Passes data through unchanged; records the last input for inspection."""

    def __init__(self):
        self.last_input = None

    def transform(self, df):
        self.last_input = df.copy()
        return df.to_numpy(dtype=float)


class _DummyModel:
    """Returns a deterministic prediction based on churn_probability."""

    def __init__(self, churn_prob: float = 0.8):
        self._prob = churn_prob

    def predict(self, X):
        return np.array([1 if self._prob >= 0.5 else 0])

    def predict_proba(self, X):
        # scikit-learn ordering: [P(class=0)=P(no churn), P(class=1)=P(churn)]
        return np.array([[1 - self._prob, self._prob]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_FEATURE_NAMES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_ENCODERS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def _build_label_encoders():
    encoders = {}
    for feature, classes in CATEGORICAL_ENCODERS.items():
        enc = LabelEncoder()
        enc.fit(classes)
        encoders[feature] = enc
    return encoders


def _make_predictor(churn_prob: float = 0.8) -> ChurnPredictor:
    """Return a ChurnPredictor instance backed by dummy model + scaler."""
    predictor = ChurnPredictor.__new__(ChurnPredictor)
    predictor.model = _DummyModel(churn_prob)
    predictor.scaler = _DummyScaler()
    predictor.feature_names = ALL_FEATURE_NAMES
    predictor.label_encoders = _build_label_encoders()
    predictor.target_encoder = {"No": 0, "Yes": 1}
    predictor.model_source = "test"
    predictor.CATEGORY_INDEX_MAPPINGS = ChurnPredictor.CATEGORY_INDEX_MAPPINGS
    return predictor


@pytest.fixture()
def predictor():
    """ChurnPredictor with full 19-feature schema backed by dummy inference."""
    return _make_predictor(churn_prob=0.8)


@pytest.fixture()
def low_risk_predictor():
    """Predictor configured to return a low-risk probability (0.2)."""
    return _make_predictor(churn_prob=0.2)


@pytest.fixture()
def medium_risk_predictor():
    """Predictor configured to return a medium-risk probability (0.55)."""
    return _make_predictor(churn_prob=0.55)


# ---------------------------------------------------------------------------
# Standard customer profiles
# ---------------------------------------------------------------------------

@pytest.fixture()
def low_risk_customer():
    """Long-tenure customer on a two-year contract — expected LOW RISK."""
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


@pytest.fixture()
def high_risk_customer():
    """New customer on a month-to-month contract — expected HIGH RISK."""
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


@pytest.fixture()
def medium_risk_customer():
    """Mid-tenure customer — expected MEDIUM RISK."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 75.0,
        "TotalCharges": 1800.0,
    }
