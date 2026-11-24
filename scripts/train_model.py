import argparse
import json

from churn_prediction.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction model and evaluate impact.")
    parser.add_argument("--data", required=True, help="Path to raw CSV data file.")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (defaults to local ./mlruns).")
    parser.add_argument("--experiment-name", help="MLflow experiment name.")
    parser.add_argument("--run-name", help="Optional MLflow run name.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking for this run.")
    args = parser.parse_args()

    results = run_training_pipeline(
        data_file=args.data,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        enable_mlflow=not args.no_mlflow,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
