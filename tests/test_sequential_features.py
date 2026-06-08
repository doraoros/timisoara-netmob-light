"""Tests for causal sequential feature engineering."""

import numpy as np
import pandas as pd

from src.features.sequential_features import SEQUENTIAL_NUMERIC, add_sequential_features


def _session(n=120, sid="S1"):
    ts = pd.date_range("2026-05-13 08:00:00", periods=n, freq="1s")
    handoff = np.zeros(n, dtype=int)
    handoff[[30, 70, 100]] = 1            # three serving-cell changes
    code = np.zeros(n, dtype=int)
    code[30:70] = 1
    code[70:100] = 2
    code[100:] = 3
    return pd.DataFrame({
        "session_id": sid,
        "timestamp": ts,
        "hour": ts.hour,
        "minute": ts.minute,
        "day_of_week": ts.dayofweek,
        "speed_kmh": np.linspace(0, 50, n),
        "handoff": handoff,
        "code": code,
    })


def test_adds_all_sequential_columns():
    out = add_sequential_features(_session())
    for col in SEQUENTIAL_NUMERIC:
        assert col in out.columns


def test_no_nans_introduced():
    out = add_sequential_features(_session())
    assert out[SEQUENTIAL_NUMERIC].isna().sum().sum() == 0


def test_cyclic_encoding_in_unit_range():
    out = add_sequential_features(_session())
    for col in ["hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos"]:
        assert out[col].between(-1.0, 1.0).all()


def test_dwell_time_resets_at_handoff():
    out = add_sequential_features(_session()).sort_values("timestamp").reset_index(drop=True)
    # dwell_time counts seconds since last change; it resets to 0 at each handoff row.
    assert out.loc[0, "dwell_time_s"] == 0
    assert out.loc[30, "dwell_time_s"] == 0      # handoff row
    assert out.loc[29, "dwell_time_s"] == 29     # just before the first change
    assert out.loc[69, "dwell_time_s"] == 39     # 70 - 30 - 1


def test_handoff_history_counts_recent_changes():
    out = add_sequential_features(_session()).sort_values("timestamp").reset_index(drop=True)
    # At t=0 there are no past handovers in the window.
    assert out.loc[0, "handoff_history_30s"] == 0
    # Right after the first change, the 30 s window contains exactly one handover.
    assert out.loc[30, "handoff_history_30s"] == 1


def test_features_are_causal_per_session():
    # Two sessions concatenated: features of session 2 must not depend on session 1.
    s1 = _session(sid="A")
    s2 = _session(sid="B")
    combined = add_sequential_features(pd.concat([s1, s2], ignore_index=True))
    only_b = add_sequential_features(s2)
    b_from_combined = combined[combined["session_id"] == "B"].reset_index(drop=True)
    np.testing.assert_allclose(
        b_from_combined["dwell_time_s"].to_numpy(),
        only_b["dwell_time_s"].to_numpy(),
    )
