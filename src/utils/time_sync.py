import pandas as pd

def merge_on_nearest(a: pd.DataFrame, b: pd.DataFrame, tolerance="2s") -> pd.DataFrame:
    if a.empty:
        return a
    if b is None or b.empty:
        return a.copy()
    a = a.sort_values("t").reset_index(drop=True)
    b = b.sort_values("t").reset_index(drop=True)
    a["_t"] = a["t"]; b["_t"] = b["t"]
    out = pd.merge_asof(a, b, on="_t", direction="nearest", tolerance=pd.Timedelta(tolerance), suffixes=("","_b"))
    out = out.drop(columns=["_t"])
    return out
