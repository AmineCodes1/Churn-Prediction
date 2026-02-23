from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"
    models: Path = project_root / "models"
    reports: Path = project_root / "reports"


@dataclass(frozen=True)
class BusinessConfig:
    retention_cost: float = 50.0
    average_clv: float = 800.0
    targeting_ratio: float = 0.20


@dataclass(frozen=True)
class ModelConfig:
    target_column: str = "churn"
    random_state: int = 42
    test_size: float = 0.20
