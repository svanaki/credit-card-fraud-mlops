"""
Tests for the automatic retraining workflow.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.retrain import (
    load_json,
    load_labeled_dataset,
    promote_candidate_model,
    save_model,
    save_workflow_result,
    train_candidate_model,
    validate_feature_columns,
)


def test_load_json(tmp_path: Path):
    path = tmp_path / "decision.json"
    path.write_text(
        json.dumps({"retraining_recommended": True}),
        encoding="utf-8",
    )

    result = load_json(path)

    assert result["retraining_recommended"] is True


def test_load_json_missing_file():
    with pytest.raises(FileNotFoundError):
        load_json(Path("missing_decision.json"))


def test_load_labeled_dataset(tmp_path: Path):
    path = tmp_path / "data.csv"

    pd.DataFrame(
        {
            "Time": [1.0, 2.0],
            "Amount": [10.0, 20.0],
            "Class": [0, 1],
        }
    ).to_csv(path, index=False)

    features, target = load_labeled_dataset(path, "Class")

    assert list(features.columns) == ["Time", "Amount"]
    assert target.tolist() == [0, 1]


def test_load_labeled_dataset_missing_target(tmp_path: Path):
    path = tmp_path / "data.csv"

    pd.DataFrame(
        {
            "Time": [1.0],
            "Amount": [10.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_labeled_dataset(path, "Class")


def test_validate_feature_columns_matching():
    first = pd.DataFrame(
        {
            "Time": [1.0],
            "Amount": [10.0],
        }
    )

    second = first.copy()

    validate_feature_columns(
        first,
        second,
        "first",
        "second",
    )


def test_validate_feature_columns_different_order():
    first = pd.DataFrame(
        {
            "Time": [1.0],
            "Amount": [10.0],
        }
    )

    second = pd.DataFrame(
        {
            "Amount": [10.0],
            "Time": [1.0],
        }
    )

    with pytest.raises(ValueError):
        validate_feature_columns(
            first,
            second,
            "first",
            "second",
        )


def test_train_candidate_model():
    features = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0],
            "Amount": [10.0, 20.0, 100.0, 200.0],
        }
    )

    target = pd.Series([0, 0, 1, 1])

    model_config = {
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
    }

    retraining_config = {
        "best_model": {
            "C": 1.0,
            "solver": "lbfgs",
        }
    }

    model = train_candidate_model(
        features=features,
        target=target,
        model_config=model_config,
        retraining_config=retraining_config,
    )

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")


def test_save_model(tmp_path: Path):
    features = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0],
            "Amount": [10.0, 20.0, 100.0, 200.0],
        }
    )

    target = pd.Series([0, 0, 1, 1])

    model = LogisticRegression(max_iter=1000)
    model.fit(features, target)

    output_path = tmp_path / "candidate.pkl"

    save_model(model, output_path)

    assert output_path.exists()


def test_promote_candidate_model_with_existing_production_model(
    tmp_path: Path,
):
    candidate_path = tmp_path / "candidate.pkl"
    production_path = tmp_path / "fraud_model.pkl"
    archive_dir = tmp_path / "archive"

    candidate_path.write_bytes(b"candidate-model")
    production_path.write_bytes(b"current-model")

    backup_path = promote_candidate_model(
        candidate_model_path=candidate_path,
        production_model_path=production_path,
        archive_dir=archive_dir,
    )

    assert backup_path is not None
    assert backup_path.exists()
    assert backup_path.read_bytes() == b"current-model"
    assert production_path.read_bytes() == b"candidate-model"


def test_promote_candidate_model_without_existing_production_model(
    tmp_path: Path,
):
    candidate_path = tmp_path / "candidate.pkl"
    production_path = tmp_path / "fraud_model.pkl"
    archive_dir = tmp_path / "archive"

    candidate_path.write_bytes(b"candidate-model")

    backup_path = promote_candidate_model(
        candidate_model_path=candidate_path,
        production_model_path=production_path,
        archive_dir=archive_dir,
    )

    assert backup_path is None
    assert production_path.read_bytes() == b"candidate-model"


def test_save_workflow_result(tmp_path: Path):
    output_path = tmp_path / "retraining_result.json"

    result = {
        "status": "completed",
        "candidate_promoted": False,
        "decision": "reject",
    }

    save_workflow_result(result, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["status"] == "completed"
    assert saved["candidate_promoted"] is False
    assert saved["decision"] == "reject"
