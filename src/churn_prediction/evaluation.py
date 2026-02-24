import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_probability_quality(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob_clipped)),
    }


def cross_validate_pipeline(
    pipeline: object,
    features: object,
    target: object,
    cv_folds: int,
    random_state: int,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=cv,
        scoring=["roc_auc", "average_precision", "precision", "recall", "f1"],
        n_jobs=-1,
        error_score="raise",
    )

    return {
        "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(scores["test_average_precision"])),
        "cv_pr_auc_std": float(np.std(scores["test_average_precision"])),
        "cv_precision_mean": float(np.mean(scores["test_precision"])),
        "cv_precision_std": float(np.std(scores["test_precision"])),
        "cv_recall_mean": float(np.mean(scores["test_recall"])),
        "cv_recall_std": float(np.std(scores["test_recall"])),
        "cv_f1_mean": float(np.mean(scores["test_f1"])),
        "cv_f1_std": float(np.std(scores["test_f1"])),
    }


def build_calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []

    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(mask.sum())
        if count == 0:
            continue

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": float(count),
                "mean_predicted_probability": float(np.mean(y_prob[mask])),
                "observed_positive_rate": float(np.mean(y_true[mask])),
            }
        )

    return rows


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
