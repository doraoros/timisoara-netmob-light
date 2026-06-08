"""Citire/scriere artefacte standard ale proiectului."""

from pathlib import Path

import pandas as pd

from src.config import (
    DATASET_RADIO_GPS,
    FEATURES_PARQUET,
    MERGED_SESSIONS_PARQUET,
    SESSIONS_CSV,
)


def load_sessions() -> pd.DataFrame:
    return pd.read_csv(SESSIONS_CSV)


def load_radio_gps_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or DATASET_RADIO_GPS
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def save_merged_sessions(df: pd.DataFrame) -> Path:
    MERGED_SESSIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MERGED_SESSIONS_PARQUET, index=False)
    return MERGED_SESSIONS_PARQUET


def load_merged_sessions() -> pd.DataFrame:
    return pd.read_parquet(MERGED_SESSIONS_PARQUET)


def save_features(df: pd.DataFrame) -> Path:
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, index=False)
    return FEATURES_PARQUET


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PARQUET)
