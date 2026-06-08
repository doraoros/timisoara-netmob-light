"""Etichetare handoff pe serii temporale (per sesiune)."""

import pandas as pd

from src.config import HO_HORIZON_S


def label_handoff_events(df: pd.DataFrame) -> pd.Series:
    """1 la schimbare de celulă (cid) sau PCI (code)."""
    cid_change = df["cid"].ne(df["cid"].shift())
    pci_change = df["code"].ne(df["code"].shift())
    ho = (cid_change | pci_change).astype(int)
    ho.iloc[0] = 0
    return ho


def label_handoff_next(
    df: pd.DataFrame,
    horizon_s: int | None = None,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    1 dacă apare un handoff în următoarele `horizon_s` secunde (nu N rânduri).
  """
    if horizon_s is None:
        horizon_s = HO_HORIZON_S

    g = df.sort_values(timestamp_col).reset_index(drop=True)
    out = pd.Series(0, index=g.index, dtype=int)
    ho_rows = g.index[g["handoff"] == 1].tolist()

    if not ho_rows:
        return out

    times = g[timestamp_col]
    for i in range(len(g)):
        t0 = times.iloc[i]
        t1 = t0 + pd.Timedelta(seconds=horizon_s)
        future = g.loc[i + 1 :]
        if future.empty:
            break
        future = future[future[timestamp_col] <= t1]
        if future.empty:
            continue
        if (future["handoff"] == 1).any():
            out.iloc[i] = 1

    return out
