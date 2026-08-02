"""
Utilities for evaluating and comparing the current and candidate models.

The candidate model is promoted only when:
1. Its PR-AUC is at least as good as the current model.
2. Its recall does not fall below the configured tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_RECALL_TOLERANCE = 0.02


def evaluate_binary_classifier(model, features, target) -> dict[str, float]:
    """
    Evaluate a binary classifier.

    PR-AUC is the primary metric because the credit-card fraud dataset
    is highly imbalanced.
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1_score": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
    }


def compare_model_metrics(
    current_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    recall_tolerance: float = DEFAULT_RECALL_TOLERANCE,
) -> dict[str, Any]:
    """
    Compare current and candidate model performance.

    Promotion conditions:
    - Candidate PR-AUC must be greater than or equal to current PR-AUC.
    - Candidate recall may not decrease by more than recall_tolerance.
    """
    if not 0.0 <= recall_tolerance <= 1.0:
        raise ValueError("recall_tolerance must be between 0 and 1.")

    required_metrics = {"pr_auc", "recall"}

    missing_current = required_metrics - current_metrics.keys()
    missing_candidate = required_metrics - candidate_metrics.keys()

    if missing_current:
        raise ValueError(
            f"Current model metrics are missing: {sorted(missing_current)}"
        )

    if missing_candidate:
        raise ValueError(
            f"Candidate model metrics are missing: {sorted(missing_candidate)}"
        )

    current_pr_auc = float(current_metrics["pr_auc"])
    candidate_pr_auc = float(candidate_metrics["pr_auc"])

    current_recall = float(current_metrics["recall"])
    candidate_recall = float(candidate_metrics["recall"])

    pr_auc_change = candidate_pr_auc - current_pr_auc
    recall_change = candidate_recall - current_recall

    pr_auc_acceptable = candidate_pr_auc >= current_pr_auc

    minimum_acceptable_recall = max(
        0.0,
        current_recall - recall_tolerance,
    )

    recall_acceptable = candidate_recall >= minimum_acceptable_recall

    promote_candidate = pr_auc_acceptable and recall_acceptable

    reasons: list[str] = []

    if pr_auc_acceptable:
        reasons.append("Candidate PR-AUC is equal to or better than the current model.")
    else:
        reasons.append("Candidate PR-AUC is lower than the current model.")

    if recall_acceptable:
        reasons.append("Candidate recall is within the allowed tolerance.")
    else:
        reasons.append("Candidate recall decreased beyond the allowed tolerance.")

    return {
        "promote_candidate": promote_candidate,
        "primary_metric": "pr_auc",
        "recall_tolerance": recall_tolerance,
        "minimum_acceptable_recall": round(
            minimum_acceptable_recall,
            6,
        ),
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_changes": {
            "pr_auc": round(pr_auc_change, 6),
            "recall": round(recall_change, 6),
        },
        "checks": {
            "pr_auc_acceptable": pr_auc_acceptable,
            "recall_acceptable": recall_acceptable,
        },
        "reasons": reasons,
        "decision": ("promote" if promote_candidate else "reject"),
    }


def compare_models(
    current_model,
    candidate_model,
    features,
    target,
    recall_tolerance: float = DEFAULT_RECALL_TOLERANCE,
) -> dict[str, Any]:
    """Evaluate and compare two models using the same dataset."""
    current_metrics = evaluate_binary_classifier(
        current_model,
        features,
        target,
    )

    candidate_metrics = evaluate_binary_classifier(
        candidate_model,
        features,
        target,
    )

    return compare_model_metrics(
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        recall_tolerance=recall_tolerance,
    )


def save_comparison_result(
    comparison_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the model comparison and promotion decision."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            comparison_result,
            output_file,
            indent=2,
        )
