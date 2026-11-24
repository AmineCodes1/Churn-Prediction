import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from churn_prediction.tracking import configure_tracking, log_training_run


class _RunInfo:
    experiment_id = "experiment-id"
    run_id = "run-id"


class _Run:
    info = _RunInfo()

    def __enter__(self) -> "_Run":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class TrackingTests(TestCase):
    def setUp(self) -> None:
        self.mlflow = types.ModuleType("mlflow")
        self.sklearn = types.ModuleType("mlflow.sklearn")
        self.models = types.ModuleType("mlflow.models")
        self.calls: dict[str, object] = {"params": {}, "metrics": {}, "artifacts": [], "dicts": []}

        self.mlflow.set_tracking_uri = lambda uri: self.calls.update(tracking_uri=uri)
        self.mlflow.set_experiment = lambda name: self.calls.update(experiment_name=name)
        self.mlflow.start_run = lambda run_name=None: _Run()
        self.mlflow.set_tags = lambda tags: self.calls.update(tags=tags)
        self.mlflow.log_params = lambda params: self.calls.update(params=params)
        self.mlflow.log_metrics = lambda metrics: self.calls.update(metrics=metrics)
        self.mlflow.log_dict = lambda value, path: self.calls["dicts"].append((value, path))
        self.mlflow.log_artifact = lambda path, artifact_path=None: self.calls["artifacts"].append(
            (path, artifact_path)
        )
        self.mlflow.get_experiment = lambda experiment_id: types.SimpleNamespace(name="test-experiment")
        self.sklearn.log_model = lambda *args, **kwargs: self.calls.update(model=kwargs)
        self.models.infer_signature = lambda inputs, outputs: "signature"
        self.mlflow.sklearn = self.sklearn
        self.mlflow.models = self.models
        self.previous_modules = {
            name: sys.modules.get(name)
            for name in ("mlflow", "mlflow.sklearn", "mlflow.models")
        }
        sys.modules.update(
            {
                "mlflow": self.mlflow,
                "mlflow.sklearn": self.sklearn,
                "mlflow.models": self.models,
            }
        )

    def tearDown(self) -> None:
        for name, module in self.previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_configure_tracking_defaults_to_local_store(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            uri = configure_tracking(None, Path(temporary_directory) / "mlruns", "test-experiment")

        self.assertTrue(uri.startswith("sqlite:///"))
        self.assertEqual(self.calls["experiment_name"], "test-experiment")

    def test_log_training_run_records_metrics_artifacts_and_model(self) -> None:
        class FakePipeline:
            def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
                return [[0.2, 0.8] for _ in range(len(frame))]

        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.joblib"
            model_path.write_bytes(b"model")
            run = log_training_run(
                tracking_uri="http://mlflow.example",
                tracking_dir=Path(temporary_directory) / "mlruns",
                experiment_name="test-experiment",
                run_name="smoke",
                data_path=Path("customers.csv"),
                data_shape=(10, 4),
                target_column="churn",
                model_config={"random_state": 42},
                business_config={"retention_cost": 50.0},
                benchmark_results={"logistic": {"roc_auc": 0.81}},
                best_model_name="logistic",
                best_pipeline=FakePipeline(),
                model_path=model_path,
                evaluation={"holdout_probability_quality": {"brier_score": 0.12}},
                business_impact={"net_benefit": 100.0},
                explainability={"top_features": []},
                x_train=pd.DataFrame({"tenure": [1, 2], "contract": ["month", "year"]}),
            )

        self.assertEqual(run["run_id"], "run-id")
        self.assertEqual(self.calls["metrics"]["benchmark.logistic.roc_auc"], 0.81)
        self.assertEqual(self.calls["metrics"]["business.net_benefit"], 100.0)
        self.assertIn(("evaluation/benchmark_results.json"), [path for _, path in self.calls["dicts"]])
        self.assertTrue(
            any(Path(path).name == "model.joblib" and artifact_path == "model_files"
                for path, artifact_path in self.calls["artifacts"])
        )
        self.assertEqual(self.calls["model"]["signature"], "signature")
        self.assertEqual(self.calls["model"]["serialization_format"], "pickle")
