"""
Tests for Evidently monitoring utilities.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.monitor import (
    extract_feature_drift_summary,
    load_dataset,
    validate_columns,
)


def test_load_dataset(tmp_path: Path):
    """A valid CSV should load successfully."""
    csv_path = tmp_path / "data.csv"

    pd.DataFrame(
        {
            "Time": [1.0, 2.0],
            "Amount": [10.0, 20.0],
            "Class": [0, 1],
        }
    ).to_csv(csv_path, index=False)

    dataframe = load_dataset(csv_path)

    assert dataframe.shape == (2, 3)


def test_load_dataset_missing_file():
    """A missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("missing_file.csv"))


def test_load_dataset_empty_file(tmp_path: Path):
    """An empty dataset should raise ValueError."""
    csv_path = tmp_path / "empty.csv"

    pd.DataFrame(columns=["Time", "Amount", "Class"]).to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(ValueError):
        load_dataset(csv_path)


def test_validate_columns_matching():
    """Matching datasets should pass validation."""
    reference = pd.DataFrame(
        {
            "Time": [1.0],
            "Amount": [10.0],
            "Class": [0],
        }
    )

    current = reference.copy()

    validate_columns(reference, current)


def test_validate_columns_mismatch():
    """Mismatched columns should raise ValueError."""
    reference = pd.DataFrame(
        {
            "Time": [1.0],
            "Amount": [10.0],
            "Class": [0],
        }
    )

    current = pd.DataFrame(
        {
            "Time": [1.0],
            "Class": [0],
        }
    )

    with pytest.raises(ValueError):
        validate_columns(reference, current)


def test_extract_no_drift_summary():
    """Zero drifted features should not recommend retraining."""
    report_data = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(columns=Time,Amount)",
                "value": {
                    "count": 0.0,
                    "share": 0.0,
                },
            }
        ]
    }

    summary = extract_feature_drift_summary(
        report_data=report_data,
        number_of_features=2,
        drift_threshold=0.5,
    )

    assert summary["dataset_drift_detected"] is False
    assert summary["number_of_drifted_features"] == 0
    assert summary["retraining_recommended"] is False


def test_extract_drift_summary():
    """High drift share should recommend retraining."""
    report_data = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(columns=Time,Amount)",
                "value": {
                    "count": 2.0,
                    "share": 1.0,
                },
            }
        ]
    }

    summary = extract_feature_drift_summary(
        report_data=report_data,
        number_of_features=2,
        drift_threshold=0.5,
    )

    assert summary["dataset_drift_detected"] is True
    assert summary["number_of_drifted_features"] == 2
    assert summary["share_of_drifted_features"] == 1.0
    assert summary["retraining_recommended"] is True


def test_extract_summary_missing_metric():
    """Missing drift metric should raise ValueError."""
    report_data = {"metrics": []}

    with pytest.raises(ValueError):
        extract_feature_drift_summary(
            report_data=report_data,
            number_of_features=30,
        )