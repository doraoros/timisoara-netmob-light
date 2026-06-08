import pandas as pd
from src.config import HO_HORIZON_S

def label_ho_next(df: pd.DataFrame, horizon_s: int = None) -> pd.DataFrame:
    if horizon_s is None:
        horizon_s = HO_HORIZON_S
    d = df[["t","eci","pci"]].sort_values("t").reset_index(drop=True).copy()
    d["ho_next"] = False
    for i in range(len(d)):
        t0 = d.at[i,"t"]; t1 = t0 + pd.Timedelta(seconds=horizon_s)
        win = d.loc[i+1:, :]
        if win.empty:
            break
        win = win[win["t"] <= t1]
        if len(win) and ((win["eci"].diff().fillna(0) != 0).any() or (win["pci"].diff().fillna(0) != 0).any()):
            d.at[i,"ho_next"] = True
    return d[["t","ho_next"]]
