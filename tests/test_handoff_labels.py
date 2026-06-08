"""Tests for handoff event and imminent-handoff labeling."""

import pandas as pd

from src.features.handoff_labels import label_handoff_events, label_handoff_next


def _frame(cids, codes, start="2026-05-13 08:00:00"):
    n = len(cids)
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="1s"),
        "cid": cids,
        "code": codes,
    })


def test_handoff_event_marks_cell_change():
    df = _frame([1, 1, 2, 2, 2], [10, 10, 10, 20, 20])
    df["handoff"] = label_handoff_events(df)
    # First row always 0; cid changes at index 2, code changes at index 3.
    assert df["handoff"].tolist() == [0, 0, 1, 1, 0]


def test_handoff_event_first_row_is_zero():
    df = _frame([5, 5, 5], [1, 1, 1])
    assert label_handoff_events(df).iloc[0] == 0


def test_no_handoff_when_constant_cell():
    df = _frame([7, 7, 7, 7], [3, 3, 3, 3])
    assert label_handoff_events(df).sum() == 0


def test_handoff_next_horizon_is_causal():
    # Handoff happens at index 3 (cid 1 -> 2).
    df = _frame([1, 1, 1, 2, 2, 2], [9, 9, 9, 9, 9, 9])
    df["handoff"] = label_handoff_events(df)
    nxt = label_handoff_next(df, horizon_s=2)
    # Rows within 2 s before the change (index 1, 2) must be 1; the change row
    # and later rows look only forward, so index 3+ are 0 (no future handoff).
    assert nxt.iloc[1] == 1
    assert nxt.iloc[2] == 1
    assert nxt.iloc[-1] == 0


def test_handoff_next_all_zero_without_events():
    df = _frame([1, 1, 1, 1], [2, 2, 2, 2])
    df["handoff"] = label_handoff_events(df)
    assert label_handoff_next(df, horizon_s=5).sum() == 0
