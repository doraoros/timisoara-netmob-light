"""Rezumat calitate dataset — pentru disertație și debugging."""

import json
from pathlib import Path

import pandas as pd

from src.config import DATASET_RADIO_GPS, REPORTS_TEXT, SPEEDTEST_ALL


def summarize_dataset(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_csv(DATASET_RADIO_GPS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    summary = {
        "n_rows": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()) if "session_id" in df.columns else None,
        "n_routes": int(df["route_id"].nunique()) if "route_id" in df.columns else None,
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "operators": df["operator"].value_counts().to_dict() if "operator" in df.columns else {},
        "transport_modes": df["transport_mode"].value_counts().to_dict()
        if "transport_mode" in df.columns
        else {},
        "handoff_rate": float(df["handoff"].mean()) if "handoff" in df.columns else None,
        "handoff_next_rate": float(df["handoff_next"].mean())
        if "handoff_next" in df.columns
        else None,
        "speedtest_matched_rate": float(df["speedtest_matched"].mean())
        if "speedtest_matched" in df.columns
        else None,
        "speedtest_rows_with_dl": int(df["dl_mbps"].notna().sum())
        if "dl_mbps" in df.columns
        else 0,
        "mean_speed_kmh": float(df["speed_kmh"].mean()) if "speed_kmh" in df.columns else None,
    }

    if "session_id" in df.columns:
        per_session = (
            df.groupby("session_id")
            .agg(
                rows=("timestamp", "count"),
                handoff_next_rate=("handoff_next", "mean"),
                operator=("operator", "first"),
                route_id=("route_id", "first"),
            )
            .reset_index()
        )
        summary["per_session"] = per_session.to_dict(orient="records")
    else:
        summary["n_sessions"] = None
        summary["per_session"] = []

    if SPEEDTEST_ALL.exists():
        st = pd.read_csv(SPEEDTEST_ALL)
        summary["speedtest_file_rows"] = int(len(st))

    return summary


def main():
    df = pd.read_csv(DATASET_RADIO_GPS)
    summary = summarize_dataset(df)
    out = REPORTS_TEXT / "dataset_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_session"}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
