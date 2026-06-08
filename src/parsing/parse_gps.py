import pandas as pd
from pathlib import Path

def load_gps_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["t","lat","lon","speed"])
    df = pd.read_csv(p)
    tcol = next((c for c in df.columns if c.lower() in ("t","time","timestamp","date","datetime")), None)
    if tcol is None:
        raise ValueError("GPS CSV trebuie să aibă coloană de timp.")
    df["t"] = pd.to_datetime(df[tcol], errors="coerce")
    latcol = next((c for c in df.columns if c.lower() in ("lat","latitude")), None)
    loncol = next((c for c in df.columns if c.lower() in ("lon","lng","longitude")), None)
    if not latcol or not loncol:
        raise ValueError("GPS CSV trebuie să aibă lat și lon.")
    spdcol = next((c for c in df.columns if "speed" in c.lower()), None)
    df["speed"] = df[spdcol] if spdcol else None
    return df[["t", latcol, loncol, "speed"]].rename(columns={latcol:"lat", loncol:"lon"}).sort_values("t").reset_index(drop=True)
