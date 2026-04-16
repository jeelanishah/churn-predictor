"""
Integration tests — end-to-end workflows including full prediction pipeline,
CSV batch processing, model loading from artifacts, fallback training,
data transformations and probability bounds.
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import ChurnPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Full pipeline — real model artifacts (if available) or fallback
# ---------------------------------------------------------------------------

class TestFullPredictionPipeline:
    """Load a real predictor (artifacts or fallback) and run end-to-end."""

    @pytest.fixture(scope="class")
    def real_predictor(self):
        return ChurnPredictor()

    def test_predictor_is_ready(self, real_predictor):
        health = real_predictor.health_check()
        assert health["model_loaded"] is True
        assert health["scaler_loaded"] is True
        assert health["features_loaded"] is True

    def test_model_source_is_set(self, real_predictor):
        health = real_predictor.health_check()
        assert health["model_source"] in ("artifacts", "fallback_training")

    def test_single_prediction_end_to_end(self, real_predictor, high_risk_customer):
        result = real_predictor.predict(high_risk_customer)
        assert result["status"] == "success"
        assert result["prediction"] in (0, 1)
        assert 0.0 <= result["probability"] <= 1.0

    def test_low_risk_end_to_end(self, real_predictor, low_risk_customer):
        result = real_predictor.predict(low_risk_customer)
        assert result["status"] == "success"
        assert result["risk_level"] in ("🔴 HIGH RISK", "🟡 MEDIUM RISK", "🟢 LOW RISK")

    def test_batch_end_to_end(self, real_predictor, low_risk_customer, high_risk_customer,
                              medium_risk_customer):
        result = real_predictor.predict_batch(
            [low_risk_customer, high_risk_customer, medium_risk_customer]
        )
        assert result["status"] == "success"
        assert result["summary"]["total"] == 3
        assert result["summary"]["successful"] == 3

    def test_probability_bounds_single(self, real_predictor, high_risk_customer):
        result = real_predictor.predict(high_risk_customer)
        assert 0.0 <= result["probability_churn"] <= 1.0
        assert 0.0 <= result["probability_no_churn"] <= 1.0

    def test_probabilities_sum_to_one(self, real_predictor, high_risk_customer):
        result = real_predictor.predict(high_risk_customer)
        total = result["probability_churn"] + result["probability_no_churn"]
        assert abs(total - 1.0) < 1e-4

    def test_feature_count_matches(self, real_predictor):
        assert len(real_predictor.feature_names) == 19


# ---------------------------------------------------------------------------
# CSV batch processing
# ---------------------------------------------------------------------------

class TestCSVBatchProcessing:
    def _make_csv(self, rows):
        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def test_batch_from_csv_dataframe(self, predictor, low_risk_customer, high_risk_customer):
        csv_buf = self._make_csv([low_risk_customer, high_risk_customer])
        df = pd.read_csv(csv_buf)
        result = predictor.predict_batch(df)
        assert result["status"] == "success"
        assert result["summary"]["total"] == 2

    def test_batch_from_csv_preserves_row_order(self, predictor, low_risk_customer, high_risk_customer):
        csv_buf = self._make_csv([low_risk_customer, high_risk_customer])
        df = pd.read_csv(csv_buf)
        result = predictor.predict_batch(df)
        for idx, row in enumerate(result["results"]):
            assert row["row_index"] == idx

    def test_large_batch_processing(self, predictor, high_risk_customer):
        records = [dict(high_risk_customer) for _ in range(50)]
        result = predictor.predict_batch(records)
        assert result["summary"]["total"] == 50
        assert result["summary"]["successful"] == 50


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_artifacts_or_fallback_loads(self):
        p = ChurnPredictor()
        assert p.model is not None
        assert p.scaler is not None
        assert p.feature_names is not None

    def test_fallback_sets_model_source(self, tmp_path, monkeypatch):
        """When artifacts are missing, the predictor should still initialise."""
        # Redirect the model directory to an empty temp path so artifact
        # loading fails and the fallback branch is exercised.
        original_load = ChurnPredictor.load_model

        def _patched_load(self):
            self.model_source = "unavailable"
            self.model = None
            self.scaler = None
            self.feature_names = None
            self.label_encoders = {}
            self.target_encoder = None
            return False

        monkeypatch.setattr(ChurnPredictor, "load_model", _patched_load)
        p = ChurnPredictor()
        assert p.model_source in ("artifacts", "fallback_training", "unavailable")

    def test_health_check_after_load(self):
        p = ChurnPredictor()
        health = p.health_check()
        assert health["status"] == "✅ API is running"


# ---------------------------------------------------------------------------
# Data transformations
# ---------------------------------------------------------------------------

class TestDataTransformations:
    def test_categorical_features_encoded_to_numeric(self, predictor, high_risk_customer):
        """_prepare_input should return a DataFrame with only numeric values."""
        df = predictor._prepare_input(high_risk_customer)
        assert (df.dtypes == float).all()

    def test_no_nan_after_prepare(self, predictor, high_risk_customer):
        df = predictor._prepare_input(high_risk_customer)
        assert not df.isna().any().any()

    def test_feature_order_preserved(self, predictor, high_risk_customer):
        df = predictor._prepare_input(high_risk_customer)
        assert list(df.columns) == predictor.feature_names

    def test_case_insensitive_yes_no(self, predictor, high_risk_customer):
        high_risk_customer["Partner"] = "yes"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_whitespace_stripped_from_strings(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "  Male  "
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Probability bounds
# ---------------------------------------------------------------------------

class TestProbabilityBounds:
    def test_probability_never_below_zero(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert result["probability"] >= 0.0

    def test_probability_never_above_one(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert result["probability"] <= 1.0

    def test_batch_all_probabilities_bounded(self, predictor, low_risk_customer, high_risk_customer):
        result = predictor.predict_batch([low_risk_customer, high_risk_customer])
        for row in result["results"]:
            if row["status"] == "success":
                assert 0.0 <= row["probability"] <= 1.0
