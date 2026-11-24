def run_training_pipeline(*args: object, **kwargs: object) -> dict[str, object]:
    from .pipeline import run_training_pipeline as _run_training_pipeline

    return _run_training_pipeline(*args, **kwargs)


__all__ = ["run_training_pipeline"]
