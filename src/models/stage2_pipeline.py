"""Canonical Stage 2 (engineered features) pipeline for handoff prediction.

Shared by the training/saving, single-prediction CLI and per-session
evaluation scripts so they all use the exact same feature set, preprocessing
and model definition. Kept intentionally close to
``scripts/experiment_feature_engineering.py`` (the experiment that selected
this configuration).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.features.sequential_features import SEQUENTIAL_NUMERIC, add_sequential_features
from src.models.train_handoff_models import load_dataset

TARGET = "handoff_next"
GROUP = "session_id"


def prepare_dataframe() -> pd.DataFrame:
    """Load the unified dataset and add causal sequential features."""
    df = load_dataset()
    return add_sequential_features(df)


def stage2_columns(df: pd.DataFrame):
    numeric = [c for c in ["speed_kmh", "frequency", "area", "is_peak_hour",
                           "is_high_speed"] if c in df.columns]
    numeric += [c for c in SEQUENTIAL_NUMERIC if c in df.columns]
    target_enc = [c for c in ["cid", "code"] if c in df.columns]
    categorical = [c for c in ["route_id", "transport_mode", "operator", "technology"]
                   if c in df.columns]
    return numeric, target_enc, categorical


def build_preprocessor(numeric, target_enc, categorical) -> ColumnTransformer:
    transformers = [("num", SimpleImputer(strategy="median"), numeric)]
    if target_enc:
        transformers.append(
            ("te", TargetEncoder(target_type="binary", random_state=RANDOM_STATE), target_enc)
        )
    if categorical:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical)
        )
    return ColumnTransformer(transformers)


def build_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def scale_pos_weight(y) -> float:
    pos = int((np.asarray(y) == 1).sum())
    neg = int((np.asarray(y) == 0).sum())
    return neg / pos if pos else 1.0


def build_pipeline(df: pd.DataFrame, y) -> Pipeline:
    numeric, target_enc, categorical = stage2_columns(df)
    preproc = build_preprocessor(numeric, target_enc, categorical)
    model = build_model(scale_pos_weight(y))
    return Pipeline([("preprocessor", preproc), ("model", model)])


def feature_columns(df: pd.DataFrame):
    numeric, target_enc, categorical = stage2_columns(df)
    return numeric + target_enc + categorical


def column_defaults(df: pd.DataFrame) -> dict:
    """Median (numeric/target-enc) and mode (categorical) per feature column.

    Used by the single-prediction CLI to fill unspecified inputs with realistic
    values from the training data.
    """
    numeric, target_enc, categorical = stage2_columns(df)
    defaults = {}
    for c in numeric + target_enc:
        defaults[c] = float(df[c].median())
    for c in categorical:
        defaults[c] = str(df[c].mode().iloc[0])
    return defaults
