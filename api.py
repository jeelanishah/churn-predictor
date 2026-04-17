"""
Churn Predictor API backend with robust validation and inference.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ChurnPredictor:
    CATEGORY_INDEX_MAPPINGS = {
        "gender": {0: "Male", 1: "Female"},
        "SeniorCitizen": {0: 0, 1: 1},
        "Partner": {0: "No", 1: "Yes"},
        "Dependents": {0: "No", 1: "Yes"},
        "PhoneService": {0: "No", 1: "Yes"},
        "MultipleLines": {0: "No", 1: "Yes", 2: "No phone service"},
        "InternetService": {0: "DSL", 1: "Fiber optic", 2: "No"},
        "OnlineSecurity": {0: "No", 1: "Yes", 2: "No internet service"},
        "OnlineBackup": {0: "No", 1: "Yes", 2: "No internet service"},
        "DeviceProtection": {0: "No", 1: "Yes", 2: "No internet service"},
        "TechSupport": {0: "No", 1: "Yes", 2: "No internet service"},
        "StreamingTV": {0: "No", 1: "Yes", 2: "No internet service"},
        "StreamingMovies": {0: "No", 1: "Yes", 2: "No internet service"},
        "Contract": {0: "Month-to-month", 1: "One year", 2: "Two year"},
        "PaperlessBilling": {0: "No", 1: "Yes"},
        "PaymentMethod": {
            0: "Electronic check",
            1: "Mailed check",
            2: "Bank transfer",
            3: "Credit card",
        },
    }

    NUMERIC_FEATURES = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoders = {}
        self.target_encoder = None
        self.model_source = "unavailable"
        self.load_model()

    def load_model(self):
        """Load model artifacts; fallback to local training if artifacts are not loadable."""
        model_path = Path(__file__).parent / "model"
        try:
            with open(model_path / "churn_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(model_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(model_path / "feature_names.pkl", "rb") as f:
                self.feature_names = pickle.load(f)
            with open(model_path / "label_encoders.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)
            with open(model_path / "target_encoder.pkl", "rb") as f:
                self.target_encoder = pickle.load(f)
            self.model_source = "artifacts"
            logger.info("Model artifacts loaded successfully")
            return True
        except Exception as exc:
            logger.exception("Failed to load model artifacts: %s", exc)
            return self._train_fallback_model()

    def _train_fallback_model(self):
        """Train a fallback model from bundled dataset to keep API functional."""
        data_path = Path(__file__).parent / "data" / "churn_data.csv"
        try:
            data = pd.read_csv(data_path)
            data = data.drop(columns=["customerID"], errors="ignore")
            if "Churn" not in data.columns:
                raise ValueError("Fallback dataset missing 'Churn' column")

            features = data.drop(columns=["Churn"]).copy()
            target = (data["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

            self.feature_names = features.columns.tolist()
            self.label_encoders = {}
            for col in self.feature_names:
                if features[col].dtype == "object":
                    encoder = LabelEncoder()
                    features[col] = encoder.fit_transform(features[col].astype(str).str.strip())
                    self.label_encoders[col] = encoder
                else:
                    features[col] = pd.to_numeric(features[col], errors="coerce")
                    features[col] = features[col].fillna(features[col].median())

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features[self.feature_names].astype(float))

            self.model = GradientBoostingClassifier(random_state=42)
            self.model.fit(X_scaled, target)
            self.target_encoder = {"No": 0, "Yes": 1}
            self.model_source = "fallback_training"
            logger.warning("Using fallback trained model from local dataset")
            return True
        except Exception as exc:
            logger.exception("Fallback training failed: %s", exc)
            self.model = None
            return False

    def _normalize_categorical_value(self, feature, value):
        if isinstance(value, str):
            normalized = value.strip()
            if feature == "PaymentMethod":
                normalized = normalized.replace("(automatic)", "").strip()
            if feature in {"MultipleLines"} and normalized.lower() == "no phone service":
                return "No phone service"
            if feature in {
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            } and normalized.lower() == "no internet service":
                return "No internet service"
            return normalized

        if feature in self.CATEGORY_INDEX_MAPPINGS:
            mapping = self.CATEGORY_INDEX_MAPPINGS[feature]
            if isinstance(value, (int, np.integer, float, np.floating)) and float(value).is_integer():
                idx = int(value)
                if idx in mapping:
                    return mapping[idx]
        return value

    def _resolve_categorical_value(self, feature, value, allowed_values, original_value):
        if value in allowed_values:
            return value

        if isinstance(value, str):
            raw = value.strip().lower()
            alias_map = {item.lower(): item for item in allowed_values}
            if raw in alias_map:
                return alias_map[raw]

            if feature == "PaymentMethod":
                automatic = f"{value.strip()} (automatic)".lower()
                if automatic in alias_map:
                    return alias_map[automatic]

            if feature == "InternetService" and raw == "no":
                if "No internet service" in allowed_values:
                    return "No internet service"
                if allowed_values:
                    logger.warning(
                        "InternetService='No' not available in encoder classes %s; using '%s'",
                        allowed_values,
                        allowed_values[0],
                    )
                    return allowed_values[0]

            if raw == "no internet service" and "No" in allowed_values:
                logger.warning(
                    "'No internet service' not available for %s classes %s; using 'No'",
                    feature,
                    allowed_values,
                )
                return "No"

            if raw == "no phone service" and "No" in allowed_values:
                logger.warning(
                    "'No phone service' not available for %s classes %s; using 'No'",
                    feature,
                    allowed_values,
                )
                return "No"

        if (
            feature in self.CATEGORY_INDEX_MAPPINGS
            and allowed_values
            and isinstance(original_value, (int, np.integer, float, np.floating))
        ):
            logger.warning(
                "Numeric category input for feature '%s' is not available in classes; using fallback '%s'",
                feature,
                allowed_values[0],
            )
            return allowed_values[0]

        raise ValueError(f"Invalid value '{value}' for '{feature}'. Allowed values: {allowed_values}")

    def _prepare_input(self, customer_data):
        if not isinstance(customer_data, dict):
            raise ValueError("customer_data must be a dictionary")
        if not self.feature_names:
            raise RuntimeError("Model artifacts are not initialized")

        missing = [f for f in self.feature_names if f not in customer_data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        row = {}
        for feature in self.feature_names:
            value = customer_data[feature]
            if feature in self.label_encoders:
                normalized = self._normalize_categorical_value(feature, value)
                encoder = self.label_encoders[feature]
                allowed_values = list(encoder.classes_)
                resolved = self._resolve_categorical_value(feature, normalized, allowed_values, value)
                row[feature] = int(encoder.transform([resolved])[0])
            else:
                numeric_value = pd.to_numeric(value, errors="coerce")
                if pd.isna(numeric_value):
                    raise ValueError(f"Invalid numeric value '{value}' for '{feature}'")
                row[feature] = float(numeric_value)

        df = pd.DataFrame([row], columns=self.feature_names).astype(float)
        if df.isna().any().any():
            raise ValueError("Input contains NaN values after preprocessing")
        if not np.isfinite(df.to_numpy()).all():
            raise ValueError("Input contains non-finite values")
        return df

    @staticmethod
    def _risk_level(probability):
        if probability >= 0.7:
            return "🔴 HIGH RISK"
        if probability >= 0.4:
            return "🟡 MEDIUM RISK"
        return "🟢 LOW RISK"

    def predict(self, customer_data):
        """Predict churn for a single customer."""
        if self.model is None or self.scaler is None:
            return {
                "status": "error",
                "error": "Model is not available",
            }
        try:
            prepared = self._prepare_input(customer_data)
            scaled = self.scaler.transform(prepared)
            if not np.all(np.isfinite(scaled)):
                raise ValueError("Scaling produced invalid values")

            prediction = int(self.model.predict(scaled)[0])
            probabilities = self.model.predict_proba(scaled)[0]
            # probabilities[0] = P(class=0) = P(NO CHURN)
            # probabilities[1] = P(class=1) = P(CHURN)
            churn_probability = float(np.clip(probabilities[1], 0.0, 1.0))
            no_churn_probability = float(np.clip(probabilities[0], 0.0, 1.0))

            return {
                "status": "success",
                "prediction": prediction,
                "probability": round(churn_probability, 4),
                "probability_churn": round(churn_probability, 4),
                "probability_no_churn": round(no_churn_probability, 4),
                "risk_level": self._risk_level(churn_probability),
            }
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def predict_batch(self, records):
        """Predict churn for batch input as DataFrame or list[dict]."""
        if isinstance(records, pd.DataFrame):
            payload = records.to_dict(orient="records")
        elif isinstance(records, list):
            payload = records
        else:
            return {"status": "error", "error": "records must be a DataFrame or list of dictionaries"}

        results = []
        for index, item in enumerate(payload):
            result = self.predict(item)
            result["row_index"] = index
            results.append(result)

        success_count = sum(1 for item in results if item["status"] == "success")
        return {
            "status": "success",
            "results": results,
            "summary": {
                "total": len(results),
                "successful": success_count,
                "failed": len(results) - success_count,
            },
        }

    def health_check(self):
        """Check if API is ready."""
        return {
            "status": "✅ API is running",
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "features_loaded": bool(self.feature_names),
            "encoders_loaded": bool(self.label_encoders),
            "model_source": self.model_source,
            "version": "2.0.0",
        }


predictor = ChurnPredictor()
