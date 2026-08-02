"""
Generate Evidently data-quality and data-drift reports.

Default reference data:
    data/processed/train.csv

Default current data:
    data/processed/test.csv

Outputs:
    drift_report.html
    drift_report.json
    retraining_decision.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

DEFAULT_REFERENCE_PATH = Path("data/processed/train.csv")
DEFAULT_CURRENT_PATH = Path("data/processed/test.csv")
DEFAULT_OUTPUT_DIR = Path("reports/monitoring")

TARGET_COLUMN = "Class"
DEFAULT_DRIFT_THRESHOLD = 0.5


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset and validate that it is usable."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def validate_columns(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> None:
    """Ensure reference and current datasets contain matching columns."""
    reference_columns = set(reference_data.columns)
    current_columns = set(current_data.columns)

    if reference_columns == current_columns:
        return

    missing_from_current = sorted(reference_columns - current_columns)
    extra_in_current = sorted(current_columns - reference_columns)

    raise ValueError(
        "Reference and current columns do not match. "
        f"Missing from current: {missing_from_current}. "
        f"Extra in current: {extra_in_current}."
    )


def build_data_definition(dataframe: pd.DataFrame) -> DataDefinition:
    """Define numerical features and the categorical target for Evidently."""
    numerical_columns = [
        column for column in dataframe.columns if column != TARGET_COLUMN
    ]

    categorical_columns = [TARGET_COLUMN] if TARGET_COLUMN in dataframe.columns else []

    return DataDefinition(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )


def find_drifted_columns_value(node: Any) -> dict[str, Any] | None:
    """Locate the computed DriftedColumnsCount value."""
    if isinstance(node, dict):
        metric_name = str(node.get("metric_name", "")).lower()
        value = node.get("value")

        if "driftedcolumnscount" in metric_name:
            if isinstance(value, dict):
                count = value.get("count")
                share = value.get("share")

                if isinstance(count, (int, float)) and isinstance(share, (int, float)):
                    return value

        for child in node.values():
            result = find_drifted_columns_value(child)

            if result is not None:
                return result

    elif isinstance(node, list):
        for child in node:
            result = find_drifted_columns_value(child)

            if result is not None:
                return result

    return None


def extract_feature_drift_summary(
    report_data: dict[str, Any],
    number_of_features: int,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict[str, Any]:
    """Extract a compact feature-drift decision from Evidently output."""
    metric_value = find_drifted_columns_value(report_data)

    if metric_value is None:
        raise ValueError(
            "Could not locate the computed DriftedColumnsCount result "
            "in the Evidently report output."
        )

    drifted_count = int(metric_value["count"])
    drifted_share = float(metric_value["share"])

    drift_detected = drifted_share >= drift_threshold

    return {
        "dataset_drift_detected": drift_detected,
        "number_of_features": number_of_features,
        "number_of_drifted_features": drifted_count,
        "share_of_drifted_features": round(drifted_share, 6),
        "drift_threshold": drift_threshold,
        "retraining_recommended": drift_detected,
    }


def generate_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: Path,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> tuple[Path, Path, Path]:
    """Generate HTML, full JSON, and compact drift-summary reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data_definition = build_data_definition(reference_data)

    reference_dataset = Dataset.from_pandas(
        reference_data,
        data_definition=data_definition,
    )

    current_dataset = Dataset.from_pandas(
        current_data,
        data_definition=data_definition,
    )

    report = Report(
        [
            DataDriftPreset(),
            DataSummaryPreset(),
        ],
        include_tests=True,
    )

    result = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    html_path = output_dir / "drift_report.html"
    json_path = output_dir / "drift_report.json"
    summary_path = output_dir / "retraining_decision.json"

    result.save_html(str(html_path))

    report_data = result.dict()

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            report_data,
            json_file,
            indent=2,
            default=str,
        )

    # Only model input features are used for the retraining decision.
    # The Class target is excluded from feature-drift calculation.
    feature_columns = [
        column for column in reference_data.columns if column != TARGET_COLUMN
    ]

    drift_summary = extract_feature_drift_summary(
        report_data=report_data,
        number_of_features=len(feature_columns),
        drift_threshold=drift_threshold,
    )

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            drift_summary,
            summary_file,
            indent=2,
        )

    return html_path, json_path, summary_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an Evidently monitoring report."
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Path to the reference dataset.",
    )

    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT_PATH,
        help="Path to the current dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for monitoring outputs.",
    )

    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help=(
            "Minimum share of drifted model features required to recommend retraining."
        ),
    )

    arguments = parser.parse_args()

    if not 0.0 <= arguments.drift_threshold <= 1.0:
        parser.error("--drift-threshold must be between 0 and 1.")

    return arguments


def main() -> None:
    """Run the monitoring workflow."""
    arguments = parse_arguments()

    print(f"Loading reference data: {arguments.reference}")
    reference_data = load_dataset(arguments.reference)

    print(f"Loading current data: {arguments.current}")
    current_data = load_dataset(arguments.current)

    validate_columns(reference_data, current_data)

    print(f"Reference shape: {reference_data.shape}")
    print(f"Current shape: {current_data.shape}")

    print("Generating Evidently monitoring report...")

    html_path, json_path, summary_path = generate_monitoring_report(
        reference_data=reference_data,
        current_data=current_data,
        output_dir=arguments.output_dir,
        drift_threshold=arguments.drift_threshold,
    )

    with summary_path.open("r", encoding="utf-8") as summary_file:
        drift_summary = json.load(summary_file)

    print("Monitoring completed successfully.")
    print(f"HTML report: {html_path}")
    print(f"Full JSON report: {json_path}")
    print(f"Drift summary: {summary_path}")

    print(f"Dataset drift detected: {drift_summary['dataset_drift_detected']}")

    print(
        "Drifted model features: "
        f"{drift_summary['number_of_drifted_features']} / "
        f"{drift_summary['number_of_features']}"
    )

    print(
        f"Share of drifted features: {drift_summary['share_of_drifted_features']:.2%}"
    )

    print(f"Retraining recommended: {drift_summary['retraining_recommended']}")


if __name__ == "__main__":
    main()
