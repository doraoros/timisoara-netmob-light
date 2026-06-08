"""Stage 1 (raw) vs Stage 2 (engineered) feature experiment."""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, REPORTS_FIGURES, REPORTS_TEXT
from src.features.sequential_features import SEQUENTIAL_NUMERIC, add_sequential_features
from src.models.train_handoff_models import load_dataset, score_predictions

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT_FIG = REPORTS_FIGURES / "thesis"
OUT_FIG.mkdir(parents=True, exist_ok=True)
TABLES = REPORTS_TEXT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
JSON_PATH = REPORTS_TEXT / "feature_engineering_experiment.json"

REAL_MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]


def stage1_columns(df):
    numeric = [c for c in [
        "speed_kmh", "frequency", "area", "code", "hour", "minute",
        "day_of_week", "is_peak_hour", "is_high_speed",
        "cells_seen_in_session", "radio_cell_index",
    ] if c in df.columns]
    categorical = [c for c in ["route_id", "transport_mode", "operator", "technology"]
                   if c in df.columns]
    return numeric, [], categorical


def stage2_columns(df):
    # Cleaned: drop raw minute/hour, radio_cell_index, cells_seen_in_session.
    numeric = [c for c in ["speed_kmh", "frequency", "area", "is_peak_hour",
                           "is_high_speed"] if c in df.columns]
    numeric += [c for c in SEQUENTIAL_NUMERIC if c in df.columns]
    target_enc = [c for c in ["cid", "code"] if c in df.columns]
    categorical = [c for c in ["route_id", "transport_mode", "operator", "technology"]
                   if c in df.columns]
    return numeric, target_enc, categorical


def make_preprocessor(numeric, target_enc, categorical, scale):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if target_enc:
        transformers.append(
            ("te", TargetEncoder(target_type="binary", random_state=RANDOM_STATE), target_enc)
        )
    if categorical:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical)
        )
    return ColumnTransformer(transformers)


def make_models():
    return {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Baseline (stratified)": DummyClassifier(strategy="stratified",
                                                 random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=14, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1),
    }


def _proba(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    return None


def _spw(y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / pos if pos else 1.0


def _fit_eval(name, model, preproc, X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([("preprocessor", preproc), ("model", model)])
    if name == "XGBoost":
        pipe.set_params(model__scale_pos_weight=_spw(y_tr))
    pipe.fit(X_tr, y_tr)
    yp = pipe.predict(X_te)
    pr = _proba(pipe, X_te)
    m = score_predictions(y_te, yp, pr)
    m.setdefault("roc_auc", 0.5)
    m.setdefault("pr_auc", float(np.mean(y_te)))
    return m


def run_stage(df, stage_fn, scale):
    numeric, target_enc, categorical = stage_fn(df)
    cols = numeric + target_enc + categorical
    X = df[cols].copy()
    y = df["handoff_next"].astype(int)
    groups = df["session_id"]

    models = make_models()
    results = {}

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(X, y, groups=groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    gkf = GroupKFold(n_splits=5)
    for name, model in models.items():
        preproc = make_preprocessor(numeric, target_enc, categorical, scale)
        holdout = _fit_eval(name, model, preproc, X_tr, y_tr, X_te, y_te)

        fold_metrics = []
        for f_tr, f_te in gkf.split(X, y, groups=groups):
            preproc_cv = make_preprocessor(numeric, target_enc, categorical, scale)
            fold_metrics.append(
                _fit_eval(name, model, preproc_cv,
                          X.iloc[f_tr], y.iloc[f_tr], X.iloc[f_te], y.iloc[f_te])
            )
        keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
        cv = {k: [float(np.mean([m.get(k, 0.0) for m in fold_metrics])),
                  float(np.std([m.get(k, 0.0) for m in fold_metrics]))] for k in keys}
        results[name] = {"holdout": holdout, "cv": cv}
        print(f"  [{name}] holdout ROC-AUC={holdout['roc_auc']:.3f} "
              f"PR-AUC={holdout['pr_auc']:.3f} F1={holdout['f1']:.3f}")
    return results, {"numeric": numeric, "target_enc": target_enc, "categorical": categorical}


def plot_stage_comparison(stage1, stage2, metric="roc_auc"):
    names = REAL_MODELS
    short = {"Logistic Regression": "LogReg", "Random Forest": "RandForest",
             "Gradient Boosting": "GradBoost", "XGBoost": "XGBoost"}
    s1 = [stage1[m]["holdout"][metric] for m in names]
    s2 = [stage2[m]["holdout"][metric] for m in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - 0.2, s1, 0.4, label="Stage 1 (raw features)", color="#95a5a6")
    b2 = ax.bar(x + 0.2, s2, 0.4, label="Stage 2 (engineered)", color="#27ae60")
    ax.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short[m] for m in names])
    ax.set_ylabel(f"{metric.upper().replace('_', '-')} (holdout)")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "stage_comparison.png", dpi=300)
    plt.close()


STAGE2_LABELS = {
    "num__speed_kmh": "Speed (km/h)",
    "num__rolling_mean_speed_15s": "Rolling mean speed (15 s)",
    "num__rolling_std_speed_15s": "Speed std (15 s)",
    "num__speed_delta_5s": "Speed delta (5 s)",
    "num__handoff_history_30s": "Handover history (30 s)",
    "num__handoff_history_60s": "Handover history (60 s)",
    "num__dwell_time_s": "Dwell time on cell",
    "num__unique_pci_last_60s": "Distinct PCIs (60 s)",
    "num__hour_sin": "Hour (sin)",
    "num__hour_cos": "Hour (cos)",
    "num__minute_sin": "Minute (sin)",
    "num__minute_cos": "Minute (cos)",
    "num__dow_sin": "Day of week (sin)",
    "num__dow_cos": "Day of week (cos)",
    "num__frequency": "Frequency",
    "num__area": "Tracking area",
    "num__is_peak_hour": "Peak hour",
    "num__is_high_speed": "High speed",
    "te__cid": "Cell CID (target-enc.)",
    "te__code": "Cell PCI (target-enc.)",
}


def plot_stage2_importance(df):
    numeric, target_enc, categorical = stage2_columns(df)
    cols = numeric + target_enc + categorical
    X, y, groups = df[cols].copy(), df["handoff_next"].astype(int), df["session_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, _ = next(splitter.split(X, y, groups=groups))
    preproc = make_preprocessor(numeric, target_enc, categorical, scale=False)
    rf = RandomForestClassifier(n_estimators=400, max_depth=14, min_samples_leaf=5,
                                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([("preprocessor", preproc), ("model", rf)])
    pipe.fit(X.iloc[tr], y.iloc[tr])
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pd.DataFrame({"feature": names,
                        "importance": pipe.named_steps["model"].feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(12).iloc[::-1]
    labels = [STAGE2_LABELS.get(f, f.split("__")[-1]) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(labels, imp["importance"], color="#16a085")
    ax.set_xlabel("Impurity-based importance (Random Forest, Stage 2)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "feature_importance_stage2.png", dpi=300)
    plt.close()


def main():
    df = load_dataset()
    df = add_sequential_features(df)
    print("Dataset with sequential features:", df.shape)

    print("Stage 1 (raw features):")
    stage1, cols1 = run_stage(df, stage1_columns, scale=False)
    print("Stage 2 (engineered + cleaned):")
    stage2, cols2 = run_stage(df, stage2_columns, scale=True)

    plot_stage_comparison(stage1, stage2, "roc_auc")
    plot_stage2_importance(df)

    # Holdout comparison table.
    rows = []
    for name in REAL_MODELS:
        rows.append({
            "model": name,
            "roc_auc_stage1": stage1[name]["holdout"]["roc_auc"],
            "roc_auc_stage2": stage2[name]["holdout"]["roc_auc"],
            "pr_auc_stage1": stage1[name]["holdout"]["pr_auc"],
            "pr_auc_stage2": stage2[name]["holdout"]["pr_auc"],
            "f1_stage1": stage1[name]["holdout"]["f1"],
            "f1_stage2": stage2[name]["holdout"]["f1"],
        })
    pd.DataFrame(rows).to_csv(TABLES / "stage_comparison_holdout.csv", index=False)

    JSON_PATH.write_text(json.dumps({
        "stage1": {"features": cols1, "results": stage1},
        "stage2": {"features": cols2, "results": stage2},
    }, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" ", JSON_PATH)
    print(" ", TABLES / "stage_comparison_holdout.csv")
    print(" ", OUT_FIG / "stage_comparison.png")


if __name__ == "__main__":
    main()
