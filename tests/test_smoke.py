from churn_prediction import run_training_pipeline


def test_pipeline_callable() -> None:
    assert callable(run_training_pipeline)
