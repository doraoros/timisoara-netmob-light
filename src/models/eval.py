"""Metrici de clasificare reutilizabile."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    return out
