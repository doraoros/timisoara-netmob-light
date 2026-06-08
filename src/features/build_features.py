import pandas as pd
from src.utils.radio import lte_band

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("t")
    d["band"] = d["earfcn_dl"].apply(lte_band)
    # ora ciclică
    d["hour"] = d["t"].dt.hour + d["t"].dt.minute/60
    ang = d["hour"] / 24 * 2*3.14159265
    d["tod_sin"] = ang.apply(__import__("math").sin)
    d["tod_cos"] = ang.apply(__import__("math").cos)
    # derivate simple pentru semnal
    for col in ["rsrp","rsrq","sinr"]:
        if col in d.columns:
            d[f"{col}_diff"] = d[col].diff().fillna(0)
    keep_geo = [c for c in ["lat","lon","speed"] if c in d.columns]
    feat = d[["t","band","time_since_last_ho","tod_sin","tod_cos"] + keep_geo +
             [c for c in d.columns if c in ["rsrp","rsrq","sinr","rsrp_diff","rsrq_diff","sinr_diff"]]]
    return feat.reset_index(drop=True)
