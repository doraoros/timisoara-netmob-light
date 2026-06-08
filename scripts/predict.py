"""Single-sample handoff probability CLI.

Loads the saved Stage 2 model and returns the probability of an imminent
serving-cell change (within the configured horizon). Any feature not provided
on the command line is filled with its training-set default (median / mode),
so you only specify the variables you care about.

Note: this dataset does not contain RSRP/SINR physical-layer indicators; the
operational predictors are mobility + network-dynamics features.

Examples:
    python scripts/predict.py --speed 42 --dwell 3 --handoff-30s 2
    python scripts/predict.py --operator Orange --transport-mode car --speed 60
    python scripts/predict.py --set unique_pci_last_60s=4 --set dwell_time_s=1
"""

from __future__ import annotations

import argparse
import json
import math

import joblib
import pandas as pd

from src.config import BASE_DIR

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "handoff_xgb_stage2.joblib"
META_PATH = MODELS_DIR / "handoff_xgb_stage2.meta.json"


def _load_artifact():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}.\n"
            "Train it first:  python scripts/train_and_save_model.py"
        )
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return pipe, meta


# Friendly CLI flag -> dataframe column.
FRIENDLY = {
    "speed": "speed_kmh",
    "rolling_speed": "rolling_mean_speed_15s",
    "speed_delta": "speed_delta_5s",
    "dwell": "dwell_time_s",
    "handoff_30s": "handoff_history_30s",
    "handoff_60s": "handoff_history_60s",
    "unique_pci": "unique_pci_last_60s",
    "operator": "operator",
    "transport_mode": "transport_mode",
    "route_id": "route_id",
    "technology": "technology",
}


def _coerce(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def main():
    p = argparse.ArgumentParser(description="Predict imminent-handoff probability.")
    p.add_argument("--speed", type=float, help="Speed (km/h).")
    p.add_argument("--rolling-speed", type=float, help="Rolling mean speed 15 s (km/h).")
    p.add_argument("--speed-delta", type=float, help="Speed delta over 5 s (km/h).")
    p.add_argument("--dwell", type=float, help="Dwell time on current cell (s).")
    p.add_argument("--handoff-30s", type=float, help="Handovers in last 30 s.")
    p.add_argument("--handoff-60s", type=float, help="Handovers in last 60 s.")
    p.add_argument("--unique-pci", type=float, help="Distinct PCIs in last 60 s.")
    p.add_argument("--hour", type=int, help="Hour of day (0-23); sets cyclic encodings.")
    p.add_argument("--operator", type=str, help="Operator (e.g. Digi, Orange).")
    p.add_argument("--transport-mode", type=str, help="tram | car | bus | walking.")
    p.add_argument("--route-id", type=str, help="Route identifier.")
    p.add_argument("--technology", type=str, help="Radio technology (e.g. LTE).")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="COL=VALUE", help="Set any feature column directly.")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    args = p.parse_args()

    pipe, meta = _load_artifact()
    cols = meta["feature_columns"]
    row = dict(meta["defaults"])

    for flag, col in FRIENDLY.items():
        val = getattr(args, flag, None)
        if val is not None and col in row:
            row[col] = val

    if args.hour is not None:
        ang_h = 2 * math.pi * args.hour / 24
        if "hour_sin" in row:
            row["hour_sin"] = math.sin(ang_h)
        if "hour_cos" in row:
            row["hour_cos"] = math.cos(ang_h)
        if "is_peak_hour" in row:
            row["is_peak_hour"] = 1.0 if args.hour in (7, 8, 9, 16, 17, 18, 19) else 0.0

    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects COL=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        if key not in row:
            raise SystemExit(f"Unknown feature column '{key}'. Valid: {', '.join(cols)}")
        row[key] = _coerce(value)

    X = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    proba = float(pipe.predict_proba(X)[:, 1][0])
    decision = "HANDOFF likely" if proba >= args.threshold else "stay on cell"

    print(f"Handoff probability (next {15}s): {proba:.3f}")
    print(f"Decision @ threshold {args.threshold:.2f}: {decision}")


if __name__ == "__main__":
    main()
