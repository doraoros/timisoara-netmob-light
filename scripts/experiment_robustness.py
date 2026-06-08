"""Robustness, generalization and deployment experiments for the Stage 2
(engineered) XGBoost model on the imminent serving-cell-change task.

This complements scripts/experiment_feature_engineering.py with the analyses
that turn a good result into a defensible, "top" dissertation:

A. Per-session metric distribution (we report the spread, not just the mean).
B. Leave-one-route-out generalization (train on 8 routes, test on the unseen one).
C. GPS-jitter robustness (Gaussian noise on speed, sequential features recomputed).
D. Sampling-rate robustness (downsample the 1 Hz timeline and retrain).
E. Edge/deployment benchmark (inference latency per sample and model size).

No external dependencies beyond xgboost + scikit-learn; runnable on CPU.

Run:
    set PYTHONPATH=D:\\project   (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/experiment_robustness.py
"""

from __future__ import annotations

import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, REPORTS_FIGURES, REPORTS_TEXT
from src.features.sequential_features import SEQUENTIAL_NUMERIC, add_sequential_features
from src.models.train_handoff_models import load_dataset

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT_FIG = REPORTS_FIGURES / "thesis"
OUT_FIG.mkdir(parents=True, exist_ok=True)
JSON_PATH = REPORTS_TEXT / "robustness_experiment.json"

NUMERIC = ["speed_kmh", "frequency", "area", "is_peak_hour", "is_high_speed"] + list(SEQUENTIAL_NUMERIC)
TARGET_ENC = ["cid", "code"]
CATEGORICAL = ["route_id", "transport_mode", "operator", "technology"]
ALL_COLS = NUMERIC + TARGET_ENC + CATEGORICAL
SPEED_DERIVED = ["rolling_mean_speed_15s", "rolling_std_speed_15s", "speed_delta_5s"]


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC),
        ("te", TargetEncoder(target_type="binary", random_state=RANDOM_STATE), TARGET_ENC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def make_xgb(spw):
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1)


def _spw(y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / pos if pos else 1.0


def fit_pipe(train):
    y = train["handoff_next"].astype(int)
    pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", make_xgb(_spw(y)))])
    pipe.fit(train[ALL_COLS], y)
    return pipe


def _scores(y_true, proba):
    y_true = np.asarray(y_true)
    out = {"pr_auc": float(average_precision_score(y_true, proba))}
    out["roc_auc"] = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


# ---------------------------------------------------------------------------
# A. Per-session metric distribution (via GroupKFold out-of-fold predictions)
# ---------------------------------------------------------------------------
def per_session_distribution(df):
    gkf = GroupKFold(n_splits=5)
    groups = df["session_id"]
    rows = []
    for tr, te in gkf.split(df, df["handoff_next"], groups=groups):
        train, test = df.iloc[tr], df.iloc[te]
        pipe = fit_pipe(train)
        proba = pipe.predict_proba(test[ALL_COLS])[:, 1]
        test = test.assign(_proba=proba)
        for sid, g in test.groupby("session_id"):
            yt = g["handoff_next"].astype(int)
            if yt.nunique() < 2 or len(g) < 30:
                continue
            s = _scores(yt, g["_proba"].values)
            rows.append({"session_id": sid, "n": len(g), "prevalence": float(yt.mean()), **s})
    res = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    for ax, metric, color in zip(axes, ["roc_auc", "pr_auc"], ["#2980b9", "#e67e22"]):
        vals = res[metric].dropna()
        ax.boxplot(vals, vert=True, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.4))
        ax.scatter(np.random.normal(1, 0.04, len(vals)), vals, color=color, s=18, alpha=0.7, zorder=3)
        ax.axhline(0.5 if metric == "roc_auc" else float(df["handoff_next"].mean()),
                   color="#7f8c8d", linestyle="--", linewidth=1)
        ax.set_title(f"{metric.upper().replace('_', '-')} per session\n"
                     f"median={vals.median():.2f}, IQR=[{vals.quantile(.25):.2f}, {vals.quantile(.75):.2f}]",
                     fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "robustness_per_session.png", dpi=300)
    plt.close()
    return {
        "n_sessions_scored": int(len(res)),
        "roc_auc": {"median": float(res["roc_auc"].median()),
                    "q25": float(res["roc_auc"].quantile(.25)),
                    "q75": float(res["roc_auc"].quantile(.75)),
                    "min": float(res["roc_auc"].min()), "max": float(res["roc_auc"].max())},
        "pr_auc": {"median": float(res["pr_auc"].median()),
                   "q25": float(res["pr_auc"].quantile(.25)),
                   "q75": float(res["pr_auc"].quantile(.75)),
                   "min": float(res["pr_auc"].min()), "max": float(res["pr_auc"].max())},
    }


# ---------------------------------------------------------------------------
# B. Leave-one-route-out generalization
# ---------------------------------------------------------------------------
def leave_one_route_out(df):
    routes = sorted(df["route_id"].dropna().unique())
    rows = []
    for r in routes:
        train = df[df["route_id"] != r]
        test = df[df["route_id"] == r]
        yt = test["handoff_next"].astype(int)
        if yt.nunique() < 2 or train["handoff_next"].nunique() < 2:
            continue
        pipe = fit_pipe(train)
        proba = pipe.predict_proba(test[ALL_COLS])[:, 1]
        s = _scores(yt, proba)
        rows.append({"route": r, "n_test": int(len(test)), "prevalence": float(yt.mean()), **s})
    res = pd.DataFrame(rows).sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(res))
    ax.barh(y - 0.2, res["roc_auc"], 0.4, label="ROC-AUC", color="#2980b9")
    ax.barh(y + 0.2, res["pr_auc"], 0.4, label="PR-AUC", color="#e67e22")
    ax.axvline(0.5, color="#7f8c8d", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(res["route"], fontsize=8)
    ax.set_xlabel("Score on the held-out route (XGBoost, Stage 2)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "robustness_leave_one_route_out.png", dpi=300)
    plt.close()
    return {"routes": res.to_dict("records"),
            "roc_auc_mean": float(res["roc_auc"].mean()),
            "pr_auc_mean": float(res["pr_auc"].mean())}


# ---------------------------------------------------------------------------
# C. GPS-jitter robustness (recompute speed-derived sequential features)
# ---------------------------------------------------------------------------
def _recompute_features(raw):
    return add_sequential_features(raw)


def gps_jitter(df_raw, train_idx, test_idx, sigmas=(0, 1, 2, 5, 10)):
    train = add_sequential_features(df_raw.iloc[train_idx])
    pipe = fit_pipe(train)
    test_raw = df_raw.iloc[test_idx].copy()
    rng = np.random.default_rng(RANDOM_STATE)
    out = []
    for sig in sigmas:
        pert = test_raw.copy()
        if sig > 0:
            pert["speed_kmh"] = (pert["speed_kmh"]
                                 + rng.normal(0, sig, len(pert))).clip(lower=0)
        feats = add_sequential_features(pert)
        yt = feats["handoff_next"].astype(int)
        proba = pipe.predict_proba(feats[ALL_COLS])[:, 1]
        s = _scores(yt, proba)
        out.append({"sigma_kmh": sig, **s})
    res = pd.DataFrame(out)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(res["sigma_kmh"], res["roc_auc"], "o-", color="#2980b9", label="ROC-AUC")
    ax.plot(res["sigma_kmh"], res["pr_auc"], "s-", color="#e67e22", label="PR-AUC")
    ax.set_xlabel("GPS speed jitter sigma (km/h)")
    ax.set_ylabel("Holdout score")
    ax.set_title("Robustness to simulated GPS-speed noise", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "robustness_gps_jitter.png", dpi=300)
    plt.close()
    return res.to_dict("records")


# ---------------------------------------------------------------------------
# D. Sampling-rate robustness (downsample 1 Hz timeline and retrain)
# ---------------------------------------------------------------------------
def sampling_rate(df_raw, periods=(1, 2, 3, 5)):
    out = []
    df_sorted = df_raw.sort_values(["session_id", "timestamp"])
    within = df_sorted.groupby("session_id").cumcount()
    for k in periods:
        sub = df_sorted[within % k == 0].copy()
        feats = add_sequential_features(sub)
        groups = feats["session_id"]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        tr, te = next(splitter.split(feats, feats["handoff_next"], groups=groups))
        pipe = fit_pipe(feats.iloc[tr])
        test = feats.iloc[te]
        yt = test["handoff_next"].astype(int)
        proba = pipe.predict_proba(test[ALL_COLS])[:, 1]
        s = _scores(yt, proba)
        out.append({"period_s": k, "n_rows": int(len(feats)), **s})
    res = pd.DataFrame(out)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(res["period_s"], res["roc_auc"], "o-", color="#2980b9", label="ROC-AUC")
    ax.plot(res["period_s"], res["pr_auc"], "s-", color="#e67e22", label="PR-AUC")
    ax.set_xlabel("Sampling period (s)  [1 = full 1 Hz]")
    ax.set_ylabel("Holdout score")
    ax.set_title("Robustness to lower (battery-friendly) sampling rate", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "robustness_sampling_rate.png", dpi=300)
    plt.close()
    return res.to_dict("records")


# ---------------------------------------------------------------------------
# E. Edge / deployment benchmark
# ---------------------------------------------------------------------------
def edge_benchmark(df, train_idx, test_idx):
    pipe = fit_pipe(df.iloc[train_idx])
    X_test = df.iloc[test_idx][ALL_COLS]
    # Warm-up.
    pipe.predict_proba(X_test.iloc[:100])
    n = len(X_test)
    t0 = time.perf_counter()
    pipe.predict_proba(X_test)
    batch_s = time.perf_counter() - t0
    # Single-sample latency (the realistic streaming case).
    one = X_test.iloc[:1]
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        pipe.predict_proba(one)
    single_ms = (time.perf_counter() - t0) / reps * 1000
    booster = pipe.named_steps["model"].get_booster()
    raw = booster.save_raw()
    return {
        "n_test_samples": int(n),
        "batch_throughput_samples_per_s": float(n / batch_s),
        "batch_latency_per_sample_ms": float(batch_s / n * 1000),
        "single_sample_latency_ms": float(single_ms),
        "xgboost_model_size_kb": float(len(raw) / 1024),
        "n_trees": int(pipe.named_steps["model"].n_estimators),
    }


def main():
    df_raw = load_dataset()
    df = add_sequential_features(df_raw)
    groups = df["session_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, df["handoff_next"], groups=groups))

    print("A. Per-session distribution ...")
    a = per_session_distribution(df)
    print("   ROC-AUC median:", round(a["roc_auc"]["median"], 3),
          "| PR-AUC median:", round(a["pr_auc"]["median"], 3))

    print("B. Leave-one-route-out ...")
    b = leave_one_route_out(df)
    print("   mean ROC-AUC:", round(b["roc_auc_mean"], 3),
          "| mean PR-AUC:", round(b["pr_auc_mean"], 3))

    print("C. GPS jitter ...")
    c = gps_jitter(df_raw, tr_idx, te_idx)
    print("   ", [(r["sigma_kmh"], round(r["roc_auc"], 3)) for r in c])

    print("D. Sampling rate ...")
    d = sampling_rate(df_raw)
    print("   ", [(r["period_s"], round(r["roc_auc"], 3)) for r in d])

    print("E. Edge benchmark ...")
    e = edge_benchmark(df, tr_idx, te_idx)
    print("   single-sample latency: %.3f ms | model size: %.1f KB"
          % (e["single_sample_latency_ms"], e["xgboost_model_size_kb"]))

    JSON_PATH.write_text(json.dumps({
        "per_session": a, "leave_one_route_out": b,
        "gps_jitter": c, "sampling_rate": d, "edge": e,
    }, indent=2), encoding="utf-8")
    print("\nSaved:", JSON_PATH)
    print("Figures -> robustness_per_session.png, robustness_leave_one_route_out.png,")
    print("           robustness_gps_jitter.png, robustness_sampling_rate.png")


if __name__ == "__main__":
    main()
