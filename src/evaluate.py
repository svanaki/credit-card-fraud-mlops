"""
Model evaluation script for the Credit Card Fraud Detection MLOps project.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def load_config(config_path: Path = Path("params.yaml")) -> dict:
    """Load project configuration from params.yaml."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_test_data(test_path: Path, target_column: str):
    """Load processed test data."""
    df = pd.read_csv(test_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def load_model(model_path: Path):
    """Load trained model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics


def save_metrics(metrics: dict, output_path: Path):
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)


def main():
    """Run model evaluation."""
    config = load_config()

    processed_dir = Path(config["data"]["processed_dir"])
    target_column = config["data"]["target_column"]

    model_dir = Path(config["paths"]["model_dir"])
    model_name = config["paths"]["model_name"]
    model_path = model_dir / model_name

    metrics_path = Path("reports/metrics/metrics.json")

    print("Loading test data...")
    X_test, y_test = load_test_data(processed_dir / "test.csv", target_column)

    print("Loading trained model...")
    model = load_model(model_path)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("Saving metrics...")
    save_metrics(metrics, metrics_path)

    print("Evaluation completed.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()