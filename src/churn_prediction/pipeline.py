from pathlib import Path

import joblib

from .business import simulate_retention_campaign
from .config import BusinessConfig, ModelConfig, PathsConfig
from .data import (
    load_dataset,
    normalize_binary_target,
    resolve_target_column,
    split_features_target,
    train_valid_split,
)
from .evaluation import evaluate_probabilities, optimize_threshold_for_recall
from .evaluation import build_calibration_table, cross_validate_pipeline, evaluate_probability_quality
from .features import build_preprocessor, detect_feature_types
from .modeling import build_model_candidates, build_model_pipeline


def run_training_pipeline(data_file: str) -> dict[str, object]:
    paths = PathsConfig()
    model_config = ModelConfig()
    business_config = BusinessConfig()

    data_path = Path(data_file)
    if not data_path.is_absolute():
        if data_path.exists():
            data_path = data_path.resolve()
        else:
            data_path = (paths.data_raw / data_path).resolve()

    df = load_dataset(data_path)
    target_column = resolve_target_column(df, model_config.target_column)
    features, target = split_features_target(df, target_column)
    target = normalize_binary_target(target)
    x_train, x_valid, y_train, y_valid = train_valid_split(
        features,
        target,
        test_size=model_config.test_size,
        random_state=model_config.random_state,
    )

    numeric_features, categorical_features = detect_feature_types(x_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    candidates = build_model_candidates(random_state=model_config.random_state)
    benchmark_results = {}
    trained_pipelines = {}
    cv_results = {}

    for model_name, estimator in candidates.items():
        pipeline = build_model_pipeline(preprocessor, estimator)
        cv_results[model_name] = cross_validate_pipeline(
            pipeline,
            x_train,
            y_train,
            cv_folds=model_config.cv_folds,
            random_state=model_config.random_state,
        )
        pipeline.fit(x_train, y_train)
        y_prob = pipeline.predict_proba(x_valid)[:, 1]
        holdout_metrics = evaluate_probabilities(y_valid.values, y_prob)
        probability_quality = evaluate_probability_quality(y_valid.values, y_prob)
        benchmark_results[model_name] = {
            **holdout_metrics,
            **probability_quality,
            **cv_results[model_name],
        }
        trained_pipelines[model_name] = (pipeline, y_prob)

    best_model_name = max(benchmark_results, key=lambda name: benchmark_results[name]["roc_auc"])
    best_pipeline, best_prob = trained_pipelines[best_model_name]

    optimal_threshold = optimize_threshold_for_recall(y_valid.values, best_prob, min_precision=0.30)
    tuned_metrics = evaluate_probabilities(y_valid.values, best_prob, threshold=optimal_threshold)
    holdout_probability_quality = evaluate_probability_quality(y_valid.values, best_prob)
    calibration_table = build_calibration_table(y_valid.values, best_prob, n_bins=10)

    business_impact = simulate_retention_campaign(
        y_true=y_valid.values,
        y_prob=best_prob,
        retention_cost=business_config.retention_cost,
        average_clv=business_config.average_clv,
        targeting_ratio=business_config.targeting_ratio,
    )

    paths.models.mkdir(parents=True, exist_ok=True)
    model_path = paths.models / f"{best_model_name}_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)

    return {
        "best_model": best_model_name,
        "benchmark_results": benchmark_results,
        "optimal_threshold": optimal_threshold,
        "tuned_metrics": tuned_metrics,
        "robust_evaluation": {
            "validation_strategy": "80/20 stratified holdout + stratified cross-validation on training split",
            "cv_folds": model_config.cv_folds,
            "best_model_cv_summary": cv_results[best_model_name],
            "holdout_probability_quality": holdout_probability_quality,
            "calibration_table": calibration_table,
        },
        "business_impact": business_impact,
        "saved_model_path": str(model_path),
    }
