import unittest

import numpy as np
from sklearn.preprocessing import LabelEncoder

from api import ChurnPredictor


class DummyScaler:
    def __init__(self):
        self.last_input = None

    def transform(self, df):
        self.last_input = df.copy()
        return np.array([[1.0, 2.0, 3.0]])


class DummyModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


class TestPredictEncoding(unittest.TestCase):
    def _build_predictor(self):
        predictor = ChurnPredictor.__new__(ChurnPredictor)
        predictor.model = DummyModel()
        predictor.scaler = DummyScaler()
        predictor.feature_names = ["gender", "tenure", "Contract"]

        gender_encoder = LabelEncoder()
        gender_encoder.fit(["Female", "Male"])

        contract_encoder = LabelEncoder()
        contract_encoder.fit(["Month-to-month", "One year", "Two year"])

        predictor.label_encoders = {
            "gender": gender_encoder,
            "Contract": contract_encoder,
        }
        predictor.target_encoder = None
        return predictor

    def test_predict_encodes_categorical_features_before_scaling(self):
        predictor = self._build_predictor()

        result = predictor.predict(
            {
                "gender": "Male",
                "tenure": 24,
                "Contract": "Month-to-month",
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["probability"], 0.8)

        self.assertEqual(predictor.scaler.last_input["gender"].iloc[0], 1)
        self.assertEqual(predictor.scaler.last_input["Contract"].iloc[0], 0)
        self.assertEqual(predictor.scaler.last_input["tenure"].iloc[0], 24)

    def test_predict_returns_error_for_unknown_categorical_value(self):
        predictor = self._build_predictor()

        result = predictor.predict(
            {
                "gender": "Unknown",
                "tenure": 24,
                "Contract": "Month-to-month",
            }
        )

        self.assertEqual(result["status"], "error")
        self.assertTrue(len(result["error"]) > 0)
        self.assertIn("gender", result["error"])
        self.assertIn("Unknown", result["error"])


if __name__ == "__main__":
    unittest.main()
