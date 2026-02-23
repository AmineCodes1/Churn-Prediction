import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def optimize_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.30,
) -> float:
    best_threshold = 0.50
    best_recall = -1.0

    for threshold in np.linspace(0.10, 0.90, 81):
        metrics = evaluate_probabilities(y_true, y_prob, threshold=threshold)
        if metrics["precision"] >= min_precision and metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            best_threshold = float(threshold)

    return best_threshold
