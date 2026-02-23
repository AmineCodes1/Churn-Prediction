import argparse
import json

from churn_prediction.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction model and evaluate impact.")
    parser.add_argument("--data", required=True, help="Path to raw CSV data file.")
    args = parser.parse_args()

    results = run_training_pipeline(data_file=args.data)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
