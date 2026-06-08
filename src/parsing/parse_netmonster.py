import pandas as pd
import json
from pathlib import Path

SENTINEL = {2147483647, 2147483647.0}

def load_netmonster(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for k, v in data.items():
            if not str(k).startswith("item_"):
                continue
            ts = v.get("timestamp") or v.get("time")
            mcc = v.get("network.mcc") or v.get("mcc")
            mnc = v.get("network.mnc") or v.get("mnc")
            lat_site = None if v.get("latitude") in SENTINEL else v.get("latitude")
            lon_site = None if v.get("longitude") in SENTINEL else v.get("longitude")
            rows.append({
                "t": pd.to_datetime(ts, unit="ms", errors="coerce"),
                "mcc": mcc, "mnc": mnc, "rat": v.get("technology"),
                "eci": v.get("cid"), "tac": v.get("area"), "pci": v.get("code"),
                "earfcn_dl": v.get("frequency"),
                "lat": v.get("gps.lat") or v.get("lat") or None,
                "lon": v.get("gps.lon") or v.get("lon") or None,
                "rsrp": v.get("rsrp"), "rsrq": v.get("rsrq"), "sinr": v.get("sinr"),
                "lat_site": lat_site, "lon_site": lon_site
            })
        df = pd.DataFrame(rows).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
        return df
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        rename = {
            "timestamp":"t","time":"t","cid":"eci","area":"tac","code":"pci",
            "frequency":"earfcn_dl","technology":"rat","longitude":"lon","latitude":"lat"
        }
        for k,v in rename.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        if "t" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["t"]):
            df["t"] = pd.to_datetime(df["t"], errors="coerce", unit="ms")
        return df.sort_values("t").reset_index(drop=True)
    else:
        raise ValueError("Format neacceptat (folosește .json sau .csv).")
