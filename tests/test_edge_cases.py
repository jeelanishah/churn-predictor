"""
Edge-case tests — boundary conditions, missing fields, invalid categoricals,
out-of-range numerics, case sensitivity, special characters, unicode, and
extreme numeric values.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Missing / empty fields
# ---------------------------------------------------------------------------

class TestMissingFields:
    def test_completely_empty_dict(self, predictor):
        result = predictor.predict({})
        assert result["status"] == "error"
        assert "Missing required features" in result["error"]

    def test_single_missing_field(self, predictor, high_risk_customer):
        del high_risk_customer["MonthlyCharges"]
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"
        assert "MonthlyCharges" in result["error"]

    def test_multiple_missing_fields(self, predictor, high_risk_customer):
        del high_risk_customer["tenure"]
        del high_risk_customer["TotalCharges"]
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_none_as_categorical_value(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = None
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_none_as_numeric_value(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = None
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_empty_string_for_categorical(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = ""
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_empty_string_for_numeric(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = ""
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Invalid categorical values
# ---------------------------------------------------------------------------

class TestInvalidCategoricals:
    def test_invalid_gender(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "Other"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_invalid_contract(self, predictor, high_risk_customer):
        high_risk_customer["Contract"] = "Weekly"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_invalid_internet_service(self, predictor, high_risk_customer):
        high_risk_customer["InternetService"] = "Satellite"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_invalid_payment_method(self, predictor, high_risk_customer):
        high_risk_customer["PaymentMethod"] = "Cryptocurrency"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_integer_value_for_string_feature(self, predictor, high_risk_customer):
        # The API logs a warning and falls back to the first allowed category
        # when a numeric index is out of the valid mapping range, so both
        # success (with fallback) and error are acceptable outcomes.
        high_risk_customer["Contract"] = 99
        result = predictor.predict(high_risk_customer)
        assert result["status"] in ("success", "error")


# ---------------------------------------------------------------------------
# Out-of-range numeric values
# ---------------------------------------------------------------------------

class TestOutOfRangeNumerics:
    def test_negative_tenure(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = -1
        # Negative tenure is technically numeric — API should either succeed or
        # return an error, but must not crash
        result = predictor.predict(high_risk_customer)
        assert result["status"] in ("success", "error")

    def test_very_large_tenure(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = 9999
        result = predictor.predict(high_risk_customer)
        assert result["status"] in ("success", "error")

    def test_zero_monthly_charges(self, predictor, high_risk_customer):
        high_risk_customer["MonthlyCharges"] = 0.0
        result = predictor.predict(high_risk_customer)
        assert result["status"] in ("success", "error")

    def test_very_large_monthly_charges(self, predictor, high_risk_customer):
        high_risk_customer["MonthlyCharges"] = 1_000_000.0
        result = predictor.predict(high_risk_customer)
        assert result["status"] in ("success", "error")

    def test_zero_total_charges(self, predictor, high_risk_customer):
        high_risk_customer["TotalCharges"] = 0.0
        result = predictor.predict(high_risk_customer)
        assert result["status"] in ("success", "error")

    def test_negative_monthly_charges(self, predictor, high_risk_customer):
        high_risk_customer["MonthlyCharges"] = -50.0
        result = predictor.predict(high_risk_customer)
        assert result["status"] in ("success", "error")


# ---------------------------------------------------------------------------
# Case sensitivity
# ---------------------------------------------------------------------------

class TestCaseSensitivity:
    def test_gender_uppercase(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "MALE"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_gender_lowercase(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "female"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_yes_lowercase(self, predictor, high_risk_customer):
        high_risk_customer["Partner"] = "yes"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_no_uppercase(self, predictor, high_risk_customer):
        high_risk_customer["Dependents"] = "NO"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_contract_mixed_case(self, predictor, high_risk_customer):
        high_risk_customer["Contract"] = "month-to-month"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_multiple_lines_no_phone_service_lowercase(self, predictor, high_risk_customer):
        """Lowercase 'no phone service' should normalise to 'No phone service'."""
        high_risk_customer["MultipleLines"] = "no phone service"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_online_security_no_internet_service_lowercase(self, predictor, high_risk_customer):
        """Lowercase 'no internet service' should normalise to 'No internet service'."""
        high_risk_customer["OnlineSecurity"] = "no internet service"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Whitespace handling
# ---------------------------------------------------------------------------

class TestWhitespace:
    def test_leading_trailing_spaces_in_categorical(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "  Male  "
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_spaces_in_contract(self, predictor, high_risk_customer):
        high_risk_customer["Contract"] = "  Month-to-month  "
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Special characters & unicode
# ---------------------------------------------------------------------------

class TestSpecialCharacters:
    def test_special_chars_in_categorical_returns_error(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "M@le!"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_unicode_in_categorical_returns_error(self, predictor, high_risk_customer):
        high_risk_customer["gender"] = "Männlich"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "error"

    def test_numeric_string_for_numeric_field(self, predictor, high_risk_customer):
        high_risk_customer["tenure"] = "24"  # valid numeric string
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"

    def test_float_string_for_numeric_field(self, predictor, high_risk_customer):
        high_risk_customer["MonthlyCharges"] = "75.50"
        result = predictor.predict(high_risk_customer)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Boundary conditions for SeniorCitizen
# ---------------------------------------------------------------------------

class TestSeniorCitizenBoundary:
    def test_senior_citizen_zero(self, predictor, high_risk_customer):
        high_risk_customer["SeniorCitizen"] = 0
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_senior_citizen_one(self, predictor, high_risk_customer):
        high_risk_customer["SeniorCitizen"] = 1
        assert predictor.predict(high_risk_customer)["status"] == "success"

    def test_senior_citizen_invalid_value(self, predictor, high_risk_customer):
        high_risk_customer["SeniorCitizen"] = 2
        result = predictor.predict(high_risk_customer)
        # Value 2 is out of the expected 0/1 range — must not crash
        assert result["status"] in ("success", "error")


# ---------------------------------------------------------------------------
# Batch edge cases
# ---------------------------------------------------------------------------

class TestBatchEdgeCases:
    def test_empty_batch_list(self, predictor):
        result = predictor.predict_batch([])
        assert result["status"] == "success"
        assert result["summary"]["total"] == 0

    def test_batch_with_all_invalid(self, predictor):
        result = predictor.predict_batch([{"bad": "data"}, {"also": "bad"}])
        assert result["summary"]["failed"] == 2
        assert result["summary"]["successful"] == 0

    def test_batch_with_mixed_valid_invalid(self, predictor, high_risk_customer):
        bad = {"gender": "INVALID_VALUE"}
        result = predictor.predict_batch([high_risk_customer, bad])
        assert result["summary"]["successful"] == 1
        assert result["summary"]["failed"] == 1
