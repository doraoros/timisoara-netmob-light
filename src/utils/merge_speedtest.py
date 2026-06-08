"""Atașează măsurători speedtest sparse pe timeline-ul sesiunilor."""

import pandas as pd

from src.config import SPEEDTEST_ALL
from src.utils.geo import haversine_m


def load_speedtest_table(path=None) -> pd.DataFrame:
    path = path or SPEEDTEST_ALL
    if not path.exists():
        return pd.DataFrame()

    st = pd.read_csv(path)
    st["timestamp"] = pd.to_datetime(st["timestamp"], errors="coerce")
    st = st.dropna(subset=["timestamp"])
    return st.sort_values("timestamp").reset_index(drop=True)


def attach_speedtest(
    df: pd.DataFrame,
    speedtest: pd.DataFrame | None = None,
    tolerance: str = "3h",
    max_dist_m: float = 15000.0,
) -> pd.DataFrame:
    """
    merge_asof temporal + filtru distanță.
    Speedtest-ul e rar; toleranța e largă, dar marcat explicit prin speedtest_matched.
    """
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    out = out.drop(columns=["dl_mbps", "ul_mbps", "ping_ms", "speedtest_matched"], errors="ignore")

    if speedtest is None:
        speedtest = load_speedtest_table()
    if speedtest.empty:
        out["dl_mbps"] = pd.NA
        out["ul_mbps"] = pd.NA
        out["ping_ms"] = pd.NA
        out["speedtest_matched"] = 0
        return out

    st = speedtest.rename(
        columns={
            "download_mbps": "dl_mbps",
            "upload_mbps": "ul_mbps",
            "latency_ms": "ping_ms",
            "latitude": "st_lat",
            "longitude": "st_lon",
        }
    )
    cols = ["timestamp", "dl_mbps", "ul_mbps", "ping_ms", "st_lat", "st_lon"]
    cols = [c for c in cols if c in st.columns]
    st = st[cols].dropna(subset=["dl_mbps"]).sort_values("timestamp")

    merged = pd.merge_asof(
        out,
        st,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    has_match = merged["dl_mbps"].notna()
    if "st_lat" in merged.columns and "latitude" in merged.columns:
        dist = merged.apply(
            lambda r: haversine_m(r["latitude"], r["longitude"], r["st_lat"], r["st_lon"])
            if pd.notna(r["dl_mbps"])
            else float("nan"),
            axis=1,
        )
        has_match = has_match & (dist <= max_dist_m)

    merged.loc[~has_match, ["dl_mbps", "ul_mbps", "ping_ms"]] = pd.NA
    merged["speedtest_matched"] = has_match.astype(int)
    for col in ("dl_mbps", "ul_mbps", "ping_ms"):
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged.drop(columns=["st_lat", "st_lon"], errors="ignore")

    return merged
