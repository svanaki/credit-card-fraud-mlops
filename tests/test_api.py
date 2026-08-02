"""
Tests for the Credit Card Fraud Detection FastAPI service.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

import src.api as api_module

VALID_TRANSACTION = {
    "Time": 406.0,
    "V1": -2.3122265423263,
    "V2": 1.95199201064158,
    "V3": -1.60985073229769,
    "V4": 3.9979055875468,
    "V5": -0.522187864667764,
    "V6": -1.42654531920595,
    "V7": -2.53738730624579,
    "V8": 1.39165724829804,
    "V9": -2.77008927719433,
    "V10": -2.77227214465915,
    "V11": 3.20203320709635,
    "V12": -2.89990738849473,
    "V13": -0.595221881324605,
    "V14": -4.28925378244217,
    "V15": 0.389724120274487,
    "V16": -1.14074717980657,
    "V17": -2.83005567450437,
    "V18": -0.0168224681808257,
    "V19": 0.416955705037907,
    "V20": 0.126910559061474,
    "V21": 0.517232370861764,
    "V22": -0.0350493686052974,
    "V23": -0.465211076182388,
    "V24": 0.320198198514526,
    "V25": 0.0445191674731724,
    "V26": 0.177839798284401,
    "V27": 0.261145002567677,
    "V28": -0.143275874698919,
    "Amount": 0.0,
}


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """Create an API client with temporary model artifacts."""

    feature_names = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ]

    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, len(feature_names))),
        columns=feature_names,
    )

    X["Time"] = np.abs(X["Time"] * 1000)
    X["Amount"] = np.abs(X["Amount"] * 100)

    y = np.array([0] * 90 + [1] * 10)

    scaler = RobustScaler()
    X_scaled = X.copy()

    X_scaled[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_scaled, y)

    model_path = tmp_path / "fraud_model.pkl"
    scaler_path = tmp_path / "scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    monkeypatch.setattr(api_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(api_module, "SCALER_PATH", scaler_path)

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Root endpoint should return API information."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Credit Card Fraud Detection API"


def test_health_endpoint(client):
    """Health endpoint should confirm that artifacts are loaded."""
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["scaler_loaded"] is True


def test_version_endpoint(client):
    """Version endpoint should return API metadata."""
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["api_version"] == "1.0.0"


def test_model_info_endpoint(client):
    """Model-info endpoint should describe the loaded model."""
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_type"] == "LogisticRegression"
    assert body["number_of_features"] == 30
    assert len(body["features"]) == 30


def test_predict_valid_transaction(client):
    """Valid input should return a fraud prediction."""
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]
    assert body["label"] in ["legitimate", "fraudulent"]
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_missing_feature_returns_422(client):
    """A missing required feature should fail validation."""
    invalid_transaction = VALID_TRANSACTION.copy()
    invalid_transaction.pop("V28")

    response = client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_predict_extra_feature_returns_422(client):
    """Unexpected input fields should fail validation."""
    invalid_transaction = {
        **VALID_TRANSACTION,
        "Class": 1,
    }

    response = client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422


def test_predict_negative_amount_returns_422(client):
    """Negative transaction amounts should fail validation."""
    invalid_transaction = {
        **VALID_TRANSACTION,
        "Amount": -10.0,
    }

    response = client.post(
        "/predict",
        json=invalid_transaction,
    )

    assert response.status_code == 422
