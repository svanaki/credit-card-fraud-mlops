"""
FastAPI application for credit card fraud prediction.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

import logging
import os

from src.config import load_config
from src.schemas import HealthResponse, PredictionResponse, TransactionInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

config = load_config()

MODEL_DIR = Path(config["paths"]["model_dir"])

DEFAULT_MODEL_PATH = MODEL_DIR / config["paths"]["model_name"]
DEFAULT_SCALER_PATH = MODEL_DIR / config["paths"]["scaler_name"]

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
)

SCALER_PATH = Path(
    os.getenv("SCALER_PATH", str(DEFAULT_SCALER_PATH))
)

MODEL_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the API starts."""
    global model, scaler

    logger.info("Starting Credit Card Fraud Detection API")

    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    if not SCALER_PATH.exists():
        logger.error("Scaler file not found: %s", SCALER_PATH)
        raise RuntimeError(f"Scaler not found at: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    logger.info("Model loaded from %s", MODEL_PATH)
    logger.info("Scaler loaded from %s", SCALER_PATH)

    yield

    model = None
    scaler = None

    logger.info("API shutdown complete")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict whether a credit card transaction is fraudulent.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "Credit Card Fraud Detection API",
        "documentation": "/docs",
        "health": "/health",
    }

@app.get("/version")
def version():
    """Return API version information."""
    return {
        "api_version": app.version,
        "model_type": "LogisticRegression",
    }

@app.get("/model-info")
def model_info():
    """Return information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "model_type": type(model).__name__,
        "number_of_features": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check whether the model and scaler are loaded."""
    return HealthResponse(
        status="healthy" if model is not None and scaler is not None else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    """Predict whether a transaction is fraudulent."""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded.",
        )

    logger.info(
        "Prediction request received: Time=%s, Amount=%s",
        transaction.Time,
        transaction.Amount,
    )
    
    transaction_df = pd.DataFrame(
        [transaction.model_dump()],
        columns=MODEL_FEATURES,
    )

    transaction_df[["Time", "Amount"]] = scaler.transform(
        transaction_df[["Time", "Amount"]]
    )

    prediction = int(model.predict(transaction_df)[0])
    fraud_probability = float(model.predict_proba(transaction_df)[0, 1])

    label = "fraudulent" if prediction == 1 else "legitimate"

    logger.info(
        "Prediction completed: prediction=%s, probability=%.6f",
        prediction,
        fraud_probability,
    )
    
    return PredictionResponse(
        prediction=prediction,
        label=label,
        fraud_probability=round(fraud_probability, 6),
    )