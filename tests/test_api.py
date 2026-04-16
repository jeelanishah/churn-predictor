"""
Core API tests — validate single predictions, batch predictions,
categorical encoding, numeric validation, risk levels, error handling
and the health-check endpoint.
"""

import pytest


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_returns_running_status(self, predictor):
        result = predictor.health_check()
        assert result["status"] == "✅ API is running"

    def test_health_check_model_loaded(self, predictor):
        result = predictor.health_check()
        assert result["model_loaded"] is True

    def test_health_check_scaler_loaded(self, predictor):
        result = predictor.health_check()
        assert result["scaler_loaded"] is True

    def test_health_check_features_loaded(self, predictor):
        result = predictor.health_check()
        assert result["features_loaded"] is True

    def test_health_check_encoders_loaded(self, predictor):
        result = predictor.health_check()
        assert result["encoders_loaded"] is True

    def test_health_check_has_version(self, predictor):
        result = predictor.health_check()
        assert "version" in result

    def test_health_check_has_model_source(self, predictor):
        result = predictor.health_check()
        assert "model_source" in result


# ---------------------------------------------------------------------------
# Single prediction — happy path
# ---------------------------------------------------------------------------

class TestSinglePrediction:
    def test_predict_returns_success_status(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_predict_returns_binary_prediction(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert result["prediction"] in (0, 1)

    def test_predict_probability_in_range(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_probability_churn_in_range(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert 0.0 <= result["probability_churn"] <= 1.0

    def test_predict_probability_no_churn_in_range(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert 0.0 <= result["probability_no_churn"] <= 1.0

    def test_predict_contains_risk_level(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert "risk_level" in result

    def test_predict_low_risk_customer(self, low_risk_predictor, low_risk_customer):
        result = low_risk_predictor.predict(low_risk_customer)
        assert result["status"] == "success"
        assert result["probability"] < 0.4
        assert "LOW" in result["risk_level"]

    def test_predict_high_risk_customer(self, predictor, high_risk_customer):
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"
        assert result["probability"] >= 0.7
        assert "HIGH" in result["risk_level"]

    def test_predict_medium_risk_customer(self, medium_risk_predictor, medium_risk_customer):
        result = medium_risk_predictor.predict(medium_risk_customer)
        assert result["status"] == "success"
        assert 0.4 <= result["probability"] < 0.7
        assert "MEDIUM" in result["risk_level"]


# ---------------------------------------------------------------------------
# Risk level calculation
# ---------------------------------------------------------------------------

class TestRiskLevel:
    def test_risk_level_high_at_0_7(self):
        from api import ChurnPredictor
        assert "HIGH" in ChurnPredictor._risk_level(0.70)

    def test_risk_level_high_above_0_7(self):
        from api import ChurnPredictor
        assert "HIGH" in ChurnPredictor._risk_level(0.95)

    def test_risk_level_medium_at_0_4(self):
        from api import ChurnPredictor
        assert "MEDIUM" in ChurnPredictor._risk_level(0.40)

    def test_risk_level_medium_below_0_7(self):
        from api import ChurnPredictor
        assert "MEDIUM" in ChurnPredictor._risk_level(0.69)

    def test_risk_level_low_below_0_4(self):
        from api import ChurnPredictor
        assert "LOW" in ChurnPredictor._risk_level(0.39)

    def test_risk_level_low_at_zero(self):
        from api import ChurnPredictor
        assert "LOW" in ChurnPredictor._risk_level(0.0)


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

class TestCategoricalEncoding:
    def test_gender_male_string(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "Male"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_gender_female_string(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "Female"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_gender_by_index(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = 0  # index 0 -> "Male"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_contract_month_to_month(self, predictor, high_risk_customer):
        high_risk_customer["Contract"] = "Month-to-month"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_contract_two_year_string(self, predictor, low_risk_customer):
        low_risk_customer["Contract"] = "Two year"
        result = predictor.predict(low_risk_customer)
        assert result["status"] == "success"

    def test_contract_by_index_two_year(self, predictor, low_risk_customer):
        low_risk_customer["Contract"] = 2  # index 2 -> "Two year"
        result = predictor.predict(low_risk_customer)
        assert result["status"] == "success"

    def test_internet_service_fiber_optic(self, predictor, high_risk_customer):
        high_risk_customer["InternetService"] = "Fiber optic"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_internet_service_dsl(self, predictor, high_risk_customer):
        high_risk_customer["InternetService"] = "DSL"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_payment_method_electronic_check(self, predictor, high_risk_customer):
        high_risk_customer["PaymentMethod"] = "Electronic check"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_payment_method_bank_transfer(self, predictor, low_risk_customer):
        low_risk_customer["PaymentMethod"] = "Bank transfer (automatic)"
        result = predictor.predict(low_risk_customer)
        assert result["status"] == "success"

    def test_yes_no_fields(self, predictor, high_risk_customer):
        for field in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
            for val in ("Yes", "No"):
                high_risk_customer[field] = val
                result = predictor.predict(high_risk_customer)
                assert result["status"] == "success", f"{field}={val} failed"


# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------

class TestNumericValidation:
    def test_senior_citizen_zero(self, predictor, high_risk_customer):
        high_risk_customer["SeniorCitizen"] = 0
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_senior_citizen_one(self, predictor, high_risk_customer):
        high_risk_customer["SeniorCitizen"] = 1
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_tenure_minimum(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = 0
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_tenure_maximum(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = 72
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_monthly_charges_float(self, predictor, high_risk_customer):
        high_risk_customer["MonthlyCharges"] = 89.99
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_total_charges_float(self, predictor, high_risk_customer):
        high_risk_customer["TotalCharges"] = 4999.50
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_non_numeric_tenure_returns_error(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = "abc"
        assert predictor.predict(high_risk_customer)["status"] == "error"

    def test_none_monthly_charges_returns_error(self, predictor, high_risk_customer):
        high_risk_customer["MonthlyCharges"] = None
        assert predictor.predict(high_risk_customer)["status"] == "error"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_single_feature_returns_error(self, predictor, high_risk_customer):
        del high_risk_customer["tenure"]
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"
        assert "Missing required features" in result["error"]

    def test_empty_dict_returns_error(self, predictor):
        result = predictor.predict({})
        assert result["status"] == "error"

    def test_invalid_gender_returns_error(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "Unknown"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"
        assert "gender" in result["error"]

    def test_invalid_contract_returns_error(self, predictor, high_risk_customer):
        high_risk_customer["Contract"] = "Weekly"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_non_dict_input_returns_error(self, predictor):
        result = predictor.predict("not a dict")
        assert result["status"] == "error"

    def test_model_none_returns_error(self, predictor, high_risk_customer):
        predictor.model = None
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"
        assert "Model is not available" in result["error"]

    def test_scaler_none_returns_error(self, predictor, high_risk_customer):
        predictor.scaler = None
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Batch predictions
# ---------------------------------------------------------------------------

class TestBatchPredictions:
    def test_batch_returns_success(self, predictor, low_risk_customer, high_risk_customer):
        result = predictor.predict_batch([low_risk_customer, high_risk_customer])
        assert result["status"] == "success"

    def test_batch_result_count(self, predictor, low_risk_customer, high_risk_customer):
        result = predictor.predict_batch([low_risk_customer, high_risk_customer])
        assert result["summary"]["total"] == 2

    def test_batch_all_successful(self, predictor, low_risk_customer, high_risk_customer):
        result = predictor.predict_batch([low_risk_customer, high_risk_customer])
        assert result["summary"]["successful"] == 2
        assert result["summary"]["failed"] == 0

    def test_batch_partial_failure(self, predictor, high_risk_customer):
        bad = {"gender": "INVALID"}
        result = predictor.predict_batch([high_risk_customer, bad])
        assert result["summary"]["successful"] == 1
        assert result["summary"]["failed"] == 1

    def test_batch_row_index_present(self, predictor, low_risk_customer, high_risk_customer):
        result = predictor.predict_batch([low_risk_customer, high_risk_customer])
        for idx, row in enumerate(result["results"]):
            assert row["row_index"] == idx

    def test_batch_single_item(self, predictor, high_risk_customer):
        result = predictor.predict_batch([high_risk_customer])
        assert result["summary"]["total"] == 1

    def test_batch_invalid_type_returns_error(self, predictor):
        result = predictor.predict_batch("not a list")
        assert result["status"] == "error"

    def test_batch_accepts_dataframe(self, predictor, low_risk_customer, high_risk_customer):
        import pandas as pd
        df = pd.DataFrame([low_risk_customer, high_risk_customer])
        result = predictor.predict_batch(df)
        assert result["status"] == "success"
        assert result["summary"]["total"] == 2
