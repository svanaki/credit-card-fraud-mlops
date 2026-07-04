"""
Model training script for the Credit Card Fraud Detection MLOps project.
"""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression


def load_config(config_path: Path = Path("params.yaml")) -> dict:
    """Load project configuration from params.yaml."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


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

    print("Loading training data...")
    X_train, y_train = load_training_data(train_path, target_column)

    print("Training baseline Logistic Regression model...")
    model = train_model(X_train, y_train, config)

    print("Saving model...")
    model_path = save_model(model, model_dir, model_name)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()