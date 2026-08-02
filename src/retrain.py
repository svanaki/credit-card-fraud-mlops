"""
Automatic retraining workflow for the Credit Card Fraud Detection project.

Workflow:
1. Read the Evidently retraining decision.
2. Stop when retraining is not recommended.
3. Train a candidate model using the configured production parameters.
4. Evaluate the current and candidate models on the same validation set.
5. Promote the candidate only when it passes the promotion rules.
6. Save comparison results and log the workflow to MLflow.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.model_comparison import (
    compare_models,
    save_comparison_result,
)

DEFAULT_DECISION_PATH = Path("reports/monitoring/simulated/retraining_decision.json")
DEFAULT_CURRENT_DATA_PATH = Path("data/monitoring/simulated_drift.csv")
DEFAULT_OUTPUT_DIR = Path("reports/retraining")

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "credit-card-fraud-retraining"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_labeled_dataset(
    path: Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a labeled dataset and separate features from the target."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' is missing from {path}.")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    return features, target


def validate_feature_columns(
    first_features: pd.DataFrame,
    second_features: pd.DataFrame,
    first_name: str,
    second_name: str,
) -> None:
    """Ensure two datasets have the same feature columns and order."""
    first_columns = list(first_features.columns)
    second_columns = list(second_features.columns)

    if first_columns != second_columns:
        raise ValueError(
            f"Feature columns do not match between {first_name} and {second_name}."
        )


def train_candidate_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: dict[str, Any],
    retraining_config: dict[str, Any],
) -> LogisticRegression:
    """Train the candidate Logistic Regression model."""
    best_model_config = retraining_config["best_model"]

    candidate_model = LogisticRegression(
        max_iter=model_config["max_iter"],
        random_state=model_config["random_state"],
        class_weight=model_config["class_weight"],
        C=best_model_config["C"],
        solver=best_model_config["solver"],
    )

    candidate_model.fit(features, target)

    return candidate_model


def save_model(model, path: Path) -> None:
    """Save a model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def promote_candidate_model(
    candidate_model_path: Path,
    production_model_path: Path,
    archive_dir: Path,
) -> Path | None:
    """
    Promote the candidate model and archive the current production model.

    Returns the backup path when an existing production model was archived.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)

    backup_path = None

    if production_model_path.exists():
        backup_path = archive_dir / "fraud_model_before_retraining.pkl"

        shutil.copy2(
            production_model_path,
            backup_path,
        )

    shutil.copy2(
        candidate_model_path,
        production_model_path,
    )

    return backup_path


def save_workflow_result(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the retraining workflow result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)


def log_retraining_to_mlflow(
    retraining_decision: dict[str, Any],
    comparison_result: dict[str, Any],
    candidate_model,
    candidate_model_path: Path,
    comparison_path: Path,
    workflow_result_path: Path,
    retraining_config: dict[str, Any],
) -> None:
    """Log retraining parameters, metrics, decisions, and artifacts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="automatic_retraining"):
        best_model_config = retraining_config["best_model"]

        mlflow.log_params(
            {
                "candidate_C": best_model_config["C"],
                "candidate_solver": best_model_config["solver"],
                "primary_metric": comparison_result["primary_metric"],
                "recall_tolerance": comparison_result["recall_tolerance"],
                "drifted_features": retraining_decision.get(
                    "number_of_drifted_features",
                    0,
                ),
                "drift_share": retraining_decision.get(
                    "share_of_drifted_features",
                    0.0,
                ),
            }
        )

        current_metrics = comparison_result["current_metrics"]
        candidate_metrics = comparison_result["candidate_metrics"]

        for metric_name, metric_value in current_metrics.items():
            mlflow.log_metric(
                f"current_{metric_name}",
                float(metric_value),
            )

        for metric_name, metric_value in candidate_metrics.items():
            mlflow.log_metric(
                f"candidate_{metric_name}",
                float(metric_value),
            )

        mlflow.log_metric(
            "candidate_promoted",
            int(comparison_result["promote_candidate"]),
        )

        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(workflow_result_path))
        mlflow.log_artifact(str(candidate_model_path))

        mlflow.sklearn.log_model(
            candidate_model,
            name="candidate_model",
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Automatically retrain, compare, and conditionally promote "
            "a candidate fraud-detection model."
        )
    )

    parser.add_argument(
        "--decision",
        type=Path,
        default=DEFAULT_DECISION_PATH,
        help="Path to the Evidently retraining decision JSON.",
    )

    parser.add_argument(
        "--current-data",
        type=Path,
        default=DEFAULT_CURRENT_DATA_PATH,
        help="Path to labeled current data used for retraining.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for retraining outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the automatic retraining workflow."""
    arguments = parse_arguments()
    config = load_config()

    target_column = config["data"]["target_column"]
    processed_dir = Path(config["data"]["processed_dir"])

    model_dir = Path(config["paths"]["model_dir"])
    production_model_path = model_dir / config["paths"]["model_name"]
    candidate_model_path = model_dir / config["paths"]["candidate_model_name"]

    retraining_config = config["retraining"]
    promotion_config = retraining_config["promotion"]

    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = arguments.output_dir / "model_comparison.json"
    workflow_result_path = arguments.output_dir / "retraining_result.json"

    print(f"Loading retraining decision: {arguments.decision}")
    retraining_decision = load_json(arguments.decision)

    retraining_recommended = bool(
        retraining_decision.get("retraining_recommended", False)
    )

    if not retraining_config.get("enabled", True):
        result = {
            "status": "skipped",
            "reason": "Retraining is disabled in params.yaml.",
            "candidate_promoted": False,
        }

        save_workflow_result(result, workflow_result_path)

        print("Retraining skipped: disabled in configuration.")
        return

    if not retraining_recommended:
        result = {
            "status": "skipped",
            "reason": "Monitoring did not recommend retraining.",
            "candidate_promoted": False,
            "monitoring_decision": retraining_decision,
        }

        save_workflow_result(result, workflow_result_path)

        print("Retraining skipped: no significant drift detected.")
        print(f"Workflow result: {workflow_result_path}")
        return

    print("Drift detected. Preparing retraining data...")

    X_train, y_train = load_labeled_dataset(
        processed_dir / "train.csv",
        target_column,
    )

    X_current, y_current = load_labeled_dataset(
        arguments.current_data,
        target_column,
    )

    X_validation, y_validation = load_labeled_dataset(
        processed_dir / "val.csv",
        target_column,
    )

    validate_feature_columns(
        X_train,
        X_current,
        "training data",
        "current data",
    )

    validate_feature_columns(
        X_train,
        X_validation,
        "training data",
        "validation data",
    )

    # Combine the historical training set with the newly observed,
    # labeled current data.
    X_retraining = pd.concat(
        [X_train, X_current],
        ignore_index=True,
    )

    y_retraining = pd.concat(
        [y_train, y_current],
        ignore_index=True,
    )

    print(f"Retraining dataset shape: {X_retraining.shape}")

    print("Training candidate model...")

    candidate_model = train_candidate_model(
        features=X_retraining,
        target=y_retraining,
        model_config=config["model"],
        retraining_config=retraining_config,
    )

    save_model(
        candidate_model,
        candidate_model_path,
    )

    print(f"Candidate model saved to: {candidate_model_path}")

    if not production_model_path.exists():
        raise FileNotFoundError(
            f"Current production model not found at: {production_model_path}"
        )

    current_model = joblib.load(production_model_path)

    print("Comparing current and candidate models...")

    comparison_result = compare_models(
        current_model=current_model,
        candidate_model=candidate_model,
        features=X_validation,
        target=y_validation,
        recall_tolerance=promotion_config["recall_tolerance"],
    )

    save_comparison_result(
        comparison_result=comparison_result,
        output_path=comparison_path,
    )

    candidate_promoted = comparison_result["promote_candidate"]
    backup_path = None

    if candidate_promoted:
        print("Candidate passed promotion checks.")

        backup_path = promote_candidate_model(
            candidate_model_path=candidate_model_path,
            production_model_path=production_model_path,
            archive_dir=model_dir / "archive",
        )

        print(f"Candidate promoted to production model: {production_model_path}")
    else:
        print("Candidate rejected. Production model was not changed.")

    workflow_result = {
        "status": "completed",
        "retraining_triggered": True,
        "candidate_promoted": candidate_promoted,
        "decision": comparison_result["decision"],
        "production_model_path": str(production_model_path),
        "candidate_model_path": str(candidate_model_path),
        "backup_model_path": (str(backup_path) if backup_path is not None else None),
        "comparison_path": str(comparison_path),
        "monitoring_decision": retraining_decision,
    }

    save_workflow_result(
        workflow_result,
        workflow_result_path,
    )

    log_retraining_to_mlflow(
        retraining_decision=retraining_decision,
        comparison_result=comparison_result,
        candidate_model=candidate_model,
        candidate_model_path=candidate_model_path,
        comparison_path=comparison_path,
        workflow_result_path=workflow_result_path,
        retraining_config=retraining_config,
    )

    print("Automatic retraining workflow completed.")
    print(f"Comparison report: {comparison_path}")
    print(f"Workflow result: {workflow_result_path}")
    print(f"Promotion decision: {comparison_result['decision']}")


if __name__ == "__main__":
    main()
