from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def resolve_target_column(df: pd.DataFrame, preferred_target: str) -> str:
    if preferred_target in df.columns:
        return preferred_target

    lowered = {column.lower(): column for column in df.columns}
    if preferred_target.lower() in lowered:
        return lowered[preferred_target.lower()]

    common_candidates = ["churn", "exited", "attrition", "left", "is_churn"]
    for candidate in common_candidates:
        if candidate in lowered:
            return lowered[candidate]

    raise KeyError(
        f"Target column '{preferred_target}' not found. Available columns: {list(df.columns)}"
    )


def normalize_binary_target(target: pd.Series) -> pd.Series:
    if target.dtype.kind in {"i", "u", "f", "b"}:
        unique_values = set(pd.Series(target).dropna().unique().tolist())
        if unique_values.issubset({0, 1}):
            return target.astype(int)
        return target

    mapping = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "y": 1,
        "no": 0,
        "false": 0,
        "0": 0,
        "n": 0,
    }

    normalized = target.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().all():
        return target
    if normalized.isna().any():
        return target
    return normalized.astype(int)


def train_valid_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
