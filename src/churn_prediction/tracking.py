from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EXPERIMENT_NAME = "churn-prediction"


def configure_tracking(
    tracking_uri: str | None,
    default_tracking_dir: Path,
    experiment_name: str | None = None,
) -> str:
    import mlflow

    resolved_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not resolved_uri:
        default_tracking_dir.parent.mkdir(parents=True, exist_ok=True)
        database_path = default_tracking_dir.parent / "mlflow.db"
        resolved_uri = f"sqlite:///{database_path.resolve().as_posix()}"

    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME))
    return resolved_uri


def _flatten_metrics(values: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        metric_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(_flatten_metrics(value, f"{metric_name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric_name] = float(value)
    return metrics


def _flatten_params(values: dict[str, Any], prefix: str = "") -> dict[str, str]:
    params: dict[str, str] = {}
    for key, value in values.items():
        param_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            params.update(_flatten_params(value, f"{param_name}."))
        elif value is not None:
            params[param_name] = str(value)
    return params


def log_training_run(
    *,
    tracking_uri: str | None,
    tracking_dir: Path,
    experiment_name: str | None,
    run_name: str | None,
    data_path: Path,
    data_shape: tuple[int, int],
    target_column: str,
    model_config: dict[str, Any],
    business_config: dict[str, Any],
    benchmark_results: dict[str, dict[str, Any]],
    best_model_name: str,
    best_pipeline: object,
    model_path: Path,
    evaluation: dict[str, Any],
    business_impact: dict[str, Any],
    explainability: dict[str, Any],
    x_train: pd.DataFrame,
) -> dict[str, str]:
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
    except ImportError as exc:
        raise ModuleNotFoundError(
            "MLflow tracking is enabled but MLflow is not installed. "
            "Install requirements.txt or rerun with --no-mlflow."
        ) from exc

    resolved_uri = configure_tracking(tracking_uri, tracking_dir, experiment_name)
    with mlflow.start_run(run_name=run_name or os.getenv("MLFLOW_RUN_NAME")) as run:
        mlflow.set_tags(
            {
                "project": "churn-prediction",
                "target_column": target_column,
                "best_model": best_model_name,
                "data_file": data_path.name,
            }
        )
        mlflow.log_params(
            _flatten_params(
                {
                    "data_rows": data_shape[0],
                    "data_columns": data_shape[1],
                    "target_column": target_column,
                    "best_model": best_model_name,
                    "model": model_config,
                    "business": business_config,
                }
            )
        )
        mlflow.log_metrics(
            _flatten_metrics(
                {
                    "benchmark": benchmark_results,
                    "evaluation": evaluation,
                    "business": business_impact,
                }
            )
        )
        mlflow.log_dict({"benchmark_results": benchmark_results}, "evaluation/benchmark_results.json")
        mlflow.log_dict({"evaluation": evaluation}, "evaluation/robust_evaluation.json")
        mlflow.log_dict({"business_impact": business_impact}, "business/business_impact.json")
        mlflow.log_dict({"explainability": explainability}, "explainability/summary.json")

        shap_plot = explainability.get("shap_summary_plot_path")
        if isinstance(shap_plot, str) and Path(shap_plot).is_file():
            mlflow.log_artifact(shap_plot, artifact_path="explainability")

        input_example = x_train.head(3)
        signature = infer_signature(input_example, best_pipeline.predict_proba(input_example))
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format="pickle",
        )
        mlflow.log_artifact(str(model_path), artifact_path="model_files")

        return {
            "tracking_uri": resolved_uri,
            "experiment_name": mlflow.get_experiment(run.info.experiment_id).name,
            "run_id": run.info.run_id,
        }
