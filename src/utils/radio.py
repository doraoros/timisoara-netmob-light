import pandas as pd

def lte_band(earfcn):
    if pd.isna(earfcn):
        return None
    try:
        e = int(earfcn)
    except Exception:
        return None
    if 0 <= e <= 599: return "B20"
    if 1200 <= e <= 1949: return "B3"
    if 2750 <= e <= 3449: return "B7"
    if 4750 <= e <= 5149: return "B1"
    return "UNK"

def detect_ho(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("t").reset_index(drop=True)
    df["eci_prev"] = df["eci"].shift(1)
    df["pci_prev"] = df["pci"].shift(1)
    df["ho_flag"] = ((df["eci"] != df["eci_prev"]) | (df["pci"] != df["pci_prev"]))
    df.loc[df.index[0], "ho_flag"] = False
    last_ho_time = df["t"].where(df["ho_flag"]).ffill()
    df["time_since_last_ho"] = (df["t"] - last_ho_time).dt.total_seconds()
    return df.drop(columns=["eci_prev","pci_prev"])
