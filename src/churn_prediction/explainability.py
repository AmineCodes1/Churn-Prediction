from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def _to_dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _format_feature_name(feature_name: str) -> str:
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "", 1)
    if feature_name.startswith("cat__"):
        raw = feature_name.replace("cat__", "", 1)
        if "_" in raw:
            base, level = raw.split("_", 1)
            return f"{base} = {level}"
        return raw
    return feature_name


def _build_insight(feature_name: str, values: np.ndarray, shap_values: np.ndarray) -> str:
    if feature_name.startswith("num__"):
        if np.std(values) == 0 or np.std(shap_values) == 0:
            return "Influence is stable in sampled data; directionality is weak."
        corr = float(np.corrcoef(values, shap_values)[0, 1])
        if np.isnan(corr):
            return "Influence is stable in sampled data; directionality is weak."
        if corr > 0.05:
            return "Higher values tend to increase churn risk."
        if corr < -0.05:
            return "Higher values tend to reduce churn risk."
        return "Effect is non-linear or weakly monotonic in sampled data."

    on_mask = values > 0
    if on_mask.sum() == 0 or (~on_mask).sum() == 0:
        return "Category appears infrequently; effect estimate is limited."
    delta = float(np.mean(shap_values[on_mask]) - np.mean(shap_values[~on_mask]))
    if delta > 0:
        return "Presence of this category tends to increase churn risk."
    return "Presence of this category tends to reduce churn risk."


def generate_shap_artifacts(
    pipeline: object,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    random_state: int,
    top_n: int = 8,
    explain_rows: int = 500,
    background_rows: int = 400,
) -> dict[str, object]:
    try:
        import shap
    except ImportError as exc:
        raise ModuleNotFoundError("Package 'shap' is required for explainability artifacts.") from exc

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["estimator"]

    valid_n = min(explain_rows, len(x_valid))
    train_n = min(background_rows, len(x_train))

    x_valid_sample = x_valid.sample(n=valid_n, random_state=random_state)
    x_train_sample = x_train.sample(n=train_n, random_state=random_state)

    transformed_valid = preprocessor.transform(x_valid_sample)
    transformed_train = preprocessor.transform(x_train_sample)

    x_valid_dense = _to_dense(transformed_valid)
    x_train_dense = _to_dense(transformed_train)

    feature_names = preprocessor.get_feature_names_out()
    estimator_name = estimator.__class__.__name__.lower()

    if "xgb" in estimator_name or "forest" in estimator_name or "tree" in estimator_name:
        explainer = shap.TreeExplainer(estimator)
        shap_raw = explainer.shap_values(x_valid_dense)
        if isinstance(shap_raw, list):
            shap_matrix = np.asarray(shap_raw[-1])
        else:
            shap_matrix = np.asarray(shap_raw)
    elif "logisticregression" in estimator_name:
        explainer = shap.LinearExplainer(estimator, x_train_dense)
        shap_matrix = np.asarray(explainer.shap_values(x_valid_dense))
    else:
        explainer = shap.Explainer(estimator, x_train_dense)
        shap_matrix = np.asarray(explainer(x_valid_dense).values)

    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, -1]

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    top_features: list[dict[str, object]] = []
    for index in top_indices:
        fname = str(feature_names[index])
        top_features.append(
            {
                "feature": _format_feature_name(fname),
                "mean_abs_shap": float(mean_abs[index]),
                "insight": _build_insight(fname, x_valid_dense[:, index], shap_matrix[:, index]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"shap_summary_{model_name}.png"

    shap.summary_plot(
        shap_matrix,
        x_valid_dense,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "sample_size": int(valid_n),
        "background_size": int(train_n),
        "shap_summary_plot_path": str(summary_path),
        "top_features": top_features,
    }