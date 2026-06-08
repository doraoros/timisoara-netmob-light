"""Sequential / engineered features for imminent serving-cell change prediction.

All features are computed per session, on the timestamp-sorted timeline, using
only past or present information (causal windows). This is essential to avoid
leakage: the prediction target ``handoff_next`` looks 15 s into the future,
so every engineered feature here must use strictly past/current rows.

Feature groups
--------------
* Cyclic time encoding (sin/cos of hour, minute, day-of-week) so that tree
  models do not split on raw, discontinuous time values.
* Kinematics (speed delta over 5 s, rolling mean/std of speed) to capture
  acceleration/braking and to smooth simulated-GPS noise.
* Network dynamics, used as a proxy for the missing physical-layer indicators:
  rolling count of recent handovers, dwell time on the current cell, and the
  number of distinct PCIs seen in the recent past (a cell-edge indicator).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Window sizes in rows; GPS is ~1 Hz, so 1 row ~ 1 second.
W_SPEED_MEAN = 15
W_SPEED_DELTA = 5
W_HO_SHORT = 30
W_HO_LONG = 60
W_PCI = 60

SEQUENTIAL_NUMERIC = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
    "speed_delta_5s", "rolling_mean_speed_15s", "rolling_std_speed_15s",
    "handoff_history_30s", "handoff_history_60s", "dwell_time_s",
    "unique_pci_last_60s",
]


def _cyclic(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    angle = 2.0 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def _nunique_window(arr: np.ndarray) -> float:
    return float(np.unique(arr).size)


def _add_for_session(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)

    g["hour_sin"], g["hour_cos"] = _cyclic(g["hour"], 24)
    g["minute_sin"], g["minute_cos"] = _cyclic(g["minute"], 60)
    g["dow_sin"], g["dow_cos"] = _cyclic(g["day_of_week"], 7)

    g["speed_delta_5s"] = (g["speed_kmh"] - g["speed_kmh"].shift(W_SPEED_DELTA)).fillna(0.0)
    g["rolling_mean_speed_15s"] = g["speed_kmh"].rolling(W_SPEED_MEAN, min_periods=1).mean()
    g["rolling_std_speed_15s"] = (
        g["speed_kmh"].rolling(W_SPEED_MEAN, min_periods=1).std().fillna(0.0)
    )

    # Network dynamics (past/current only).
    g["handoff_history_30s"] = g["handoff"].rolling(W_HO_SHORT, min_periods=1).sum()
    g["handoff_history_60s"] = g["handoff"].rolling(W_HO_LONG, min_periods=1).sum()
    g["unique_pci_last_60s"] = (
        g["code"].rolling(W_PCI, min_periods=1).apply(_nunique_window, raw=True)
    )

    # Dwell time: seconds since the last serving-cell change within the session.
    ho_block = g["handoff"].fillna(0).astype(int).cumsum()
    g["dwell_time_s"] = g.groupby(ho_block).cumcount().astype(float)

    return g


def add_sequential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with causal sequential features added per session.

    Requires columns: session_id, timestamp, hour, minute, day_of_week,
    speed_kmh, handoff, code.
    """
    required = {
        "session_id", "timestamp", "hour", "minute", "day_of_week",
        "speed_kmh", "handoff", "code",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for sequential features: {sorted(missing)}")

    parts = [_add_for_session(g) for _, g in df.groupby("session_id", sort=False)]
    return pd.concat(parts, ignore_index=True)
