"""
Model training script for the Credit Card Fraud Detection MLOps project.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from config import load_config
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def load_training_data(train_path: Path, target_column: str):
    """Load processed training data."""
    df = pd.read_csv(train_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def train_model(X, y, config: dict):
    """Train a baseline Logistic Regression model."""
    model_config = config["model"]

    model = LogisticRegression(
        max_iter=model_config["max_iter"],
        random_state=model_config["random_state"],
        class_weight=model_config["class_weight"],
    )

    model.fit(X, y)

    return model


def save_model(model, model_dir: Path, model_name: str):
    """Save trained model."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    joblib.dump(model, model_path)

    return model_path


def main():
    """Run model training."""
    config = load_config()

    processed_dir = Path(config["data"]["processed_dir"])
    target_column = config["data"]["target_column"]

    train_path = processed_dir / "train.csv"

    model_dir = Path(config["paths"]["model_dir"])
    model_name = config["paths"]["model_name"]
    
    val_path = processed_dir / "val.csv"

    print("Loading training data...")
    X_train, y_train = load_training_data(train_path, target_column)
    
    print("Loading validation data...")
    X_val, y_val = load_training_data(val_path, target_column)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-card-fraud-detection")
    
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_params(config["model"])

        model = train_model(X_train, y_train, config)

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        mlflow.log_metric("val_precision", precision_score(y_val, y_val_pred, zero_division=0))
        mlflow.log_metric("val_recall", recall_score(y_val, y_val_pred, zero_division=0))
        mlflow.log_metric("val_f1", f1_score(y_val, y_val_pred, zero_division=0))
        mlflow.log_metric("val_roc_auc", roc_auc_score(y_val, y_val_proba))
        mlflow.log_metric("val_pr_auc", average_precision_score(y_val, y_val_proba))

        model_path = save_model(model, model_dir, model_name)

        mlflow.sklearn.log_model(model, "model")

        print(f"Model saved to: {model_path}")

    
if __name__ == "__main__":
    main()