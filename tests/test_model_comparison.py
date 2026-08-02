"""
Tests for model comparison and promotion decisions.
"""

from pathlib import Path

import pytest

from src.model_comparison import (
    compare_model_metrics,
    save_comparison_result,
)


def test_promote_when_pr_auc_improves_and_recall_is_acceptable():
    current_metrics = {
        "pr_auc": 0.68,
        "recall": 0.87,
    }

    candidate_metrics = {
        "pr_auc": 0.70,
        "recall": 0.86,
    }

    result = compare_model_metrics(
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        recall_tolerance=0.02,
    )

    assert result["promote_candidate"] is True
    assert result["decision"] == "promote"
    assert result["checks"]["pr_auc_acceptable"] is True
    assert result["checks"]["recall_acceptable"] is True


def test_reject_when_pr_auc_is_lower():
    current_metrics = {
        "pr_auc": 0.70,
        "recall": 0.87,
    }

    candidate_metrics = {
        "pr_auc": 0.68,
        "recall": 0.88,
    }

    result = compare_model_metrics(
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
    )

    assert result["promote_candidate"] is False
    assert result["decision"] == "reject"
    assert result["checks"]["pr_auc_acceptable"] is False


def test_reject_when_recall_drops_too_much():
    current_metrics = {
        "pr_auc": 0.68,
        "recall": 0.87,
    }

    candidate_metrics = {
        "pr_auc": 0.71,
        "recall": 0.80,
    }

    result = compare_model_metrics(
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        recall_tolerance=0.02,
    )

    assert result["promote_candidate"] is False
    assert result["decision"] == "reject"
    assert result["checks"]["pr_auc_acceptable"] is True
    assert result["checks"]["recall_acceptable"] is False


def test_promote_when_metrics_are_equal():
    current_metrics = {
        "pr_auc": 0.68,
        "recall": 0.87,
    }

    candidate_metrics = {
        "pr_auc": 0.68,
        "recall": 0.87,
    }

    result = compare_model_metrics(
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
    )

    assert result["promote_candidate"] is True
    assert result["decision"] == "promote"


def test_invalid_recall_tolerance():
    with pytest.raises(ValueError):
        compare_model_metrics(
            current_metrics={
                "pr_auc": 0.68,
                "recall": 0.87,
            },
            candidate_metrics={
                "pr_auc": 0.70,
                "recall": 0.86,
            },
            recall_tolerance=1.5,
        )


def test_missing_current_metric():
    with pytest.raises(ValueError):
        compare_model_metrics(
            current_metrics={
                "recall": 0.87,
            },
            candidate_metrics={
                "pr_auc": 0.70,
                "recall": 0.86,
            },
        )


def test_missing_candidate_metric():
    with pytest.raises(ValueError):
        compare_model_metrics(
            current_metrics={
                "pr_auc": 0.68,
                "recall": 0.87,
            },
            candidate_metrics={
                "recall": 0.86,
            },
        )


def test_save_comparison_result(tmp_path: Path):
    output_path = tmp_path / "comparison.json"

    result = {
        "promote_candidate": True,
        "decision": "promote",
    }

    save_comparison_result(
        comparison_result=result,
        output_path=output_path,
    )

    assert output_path.exists()
    assert '"decision": "promote"' in output_path.read_text(encoding="utf-8")
