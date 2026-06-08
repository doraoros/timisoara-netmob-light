import pandas as pd
from pathlib import Path

def load_speedtest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["t","dl_mbps","ul_mbps","ping_ms"])
    df = pd.read_csv(p)
    # detect col timp
    tcol = next((c for c in df.columns if c.lower() in ("t","timestamp","time","date","datetime")), None)
    if tcol is None:
        raise ValueError("Speedtest CSV trebuie să aibă coloană de timp.")
    df["t"] = pd.to_datetime(df[tcol], errors="coerce")
    # normalize
    mapping = {
        "download":"dl_mbps","download_mbps":"dl_mbps","dl":"dl_mbps",
        "upload":"ul_mbps","upload_mbps":"ul_mbps","ul":"ul_mbps",
        "ping":"ping_ms","latency":"ping_ms","latency_ms":"ping_ms"
    }
    for k,v in mapping.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    keep = ["t","dl_mbps","ul_mbps","ping_ms"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    return df[keep].sort_values("t").reset_index(drop=True)
