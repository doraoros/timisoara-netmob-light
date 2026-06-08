"""SHAP explainability for the Stage 2 (engineered) XGBoost model.

Produces figures in ``reports/figures/thesis/`` and writes
``reports/text/explainability.json``.

Run:
    set PYTHONPATH=D:\\project   (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/experiment_explainability.py
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, REPORTS_FIGURES, REPORTS_TEXT
from src.features.sequential_features import SEQUENTIAL_NUMERIC, add_sequential_features
from src.models.train_handoff_models import load_dataset

warnings.filterwarnings("ignore")

FIG_OUT = REPORTS_FIGURES / "thesis"
FIG_OUT.mkdir(parents=True, exist_ok=True)

NUMERIC = ["speed_kmh", "frequency", "area", "is_peak_hour", "is_high_speed"] + list(SEQUENTIAL_NUMERIC)
TARGET_ENC = ["cid", "code"]
CATEGORICAL = ["route_id", "transport_mode", "operator", "technology"]
ALL_COLS = NUMERIC + TARGET_ENC + CATEGORICAL

LABELS = {
    "en": {
        "num__speed_kmh": "Speed (km/h)", "num__frequency": "Frequency",
        "num__area": "Tracking area", "num__is_peak_hour": "Peak hour",
        "num__is_high_speed": "High speed",
        "num__hour_sin": "Hour (sin)", "num__hour_cos": "Hour (cos)",
        "num__minute_sin": "Minute (sin)", "num__minute_cos": "Minute (cos)",
        "num__dow_sin": "Day of week (sin)", "num__dow_cos": "Day of week (cos)",
        "num__speed_delta_5s": "Speed delta (5 s)",
        "num__rolling_mean_speed_15s": "Rolling mean speed (15 s)",
        "num__rolling_std_speed_15s": "Speed std (15 s)",
        "num__handoff_history_30s": "Handover history (30 s)",
        "num__handoff_history_60s": "Handover history (60 s)",
        "num__dwell_time_s": "Dwell time on cell",
        "num__unique_pci_last_60s": "Distinct PCIs (60 s)",
        "te__cid": "Cell CID (target-enc.)", "te__code": "Cell PCI (target-enc.)",
        "_shap_x": "SHAP value (impact on log-odds of imminent change)",
        "_feat_val": "Feature value",
        "_imp_x": "Mean |SHAP| (mean impact on model output)",
        "_dwell_x": "Dwell time on current cell (s)",
        "_dwell_y": "SHAP value for dwell time",
        "_dwell_c": "Cell CID historical handover rate (target-enc.)",
        "_ho_x": "Recent handover count (last 30 s)",
        "_ho_y": "SHAP value for handover history",
        "_beeswarm_t": "SHAP summary (XGBoost, Stage 2)",
        "_imp_t": "Global feature importance (mean |SHAP|)",
        "_dwell_t": "Interaction: dwell time x cell identity",
        "_ho_t": "Dependence: recent handover history",
    },
    "ro": {
        "num__speed_kmh": "Viteza (km/h)", "num__frequency": "Frecventa",
        "num__area": "Aria de localizare", "num__is_peak_hour": "Ora de varf",
        "num__is_high_speed": "Viteza mare",
        "num__hour_sin": "Ora (sin)", "num__hour_cos": "Ora (cos)",
        "num__minute_sin": "Minut (sin)", "num__minute_cos": "Minut (cos)",
        "num__dow_sin": "Zi saptamana (sin)", "num__dow_cos": "Zi saptamana (cos)",
        "num__speed_delta_5s": "Variatie viteza (5 s)",
        "num__rolling_mean_speed_15s": "Viteza medie mobila (15 s)",
        "num__rolling_std_speed_15s": "Abatere std viteza (15 s)",
        "num__handoff_history_30s": "Istoric handover (30 s)",
        "num__handoff_history_60s": "Istoric handover (60 s)",
        "num__dwell_time_s": "Timp stationare pe celula",
        "num__unique_pci_last_60s": "PCI-uri distincte (60 s)",
        "te__cid": "CID celula (target-enc.)", "te__code": "PCI celula (target-enc.)",
        "_shap_x": "Valoare SHAP (impact asupra log-odds de handover iminent)",
        "_feat_val": "Valoarea caracteristicii",
        "_imp_x": "Medie |SHAP| (impact mediu asupra iesirii modelului)",
        "_dwell_x": "Timp de stationare pe celula curenta (s)",
        "_dwell_y": "Valoare SHAP pentru timpul de stationare",
        "_dwell_c": "Rata istorica de handover a celulei CID (target-enc.)",
        "_ho_x": "Numar handover-uri recente (ultimele 30 s)",
        "_ho_y": "Valoare SHAP pentru istoricul de handover",
        "_beeswarm_t": "Sinteza SHAP (XGBoost, Stage 2)",
        "_imp_t": "Importanta globala a caracteristicilor (medie |SHAP|)",
        "_dwell_t": "Interactiune: timp de stationare x identitatea celulei",
        "_ho_t": "Dependenta: istoricul recent de handover",
    },
}


def _label(lang, name):
    d = LABELS[lang]
    if name in d:
        return d[name]
    if name.startswith("cat__"):
        return name[len("cat__"):]
    return name.split("__")[-1]


def make_pipeline(y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC),
        ("te", TargetEncoder(target_type="binary", random_state=RANDOM_STATE), TARGET_ENC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=neg / pos, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("preprocessor", pre), ("model", model)])


def beeswarm(shap_values, X_trans, names, lang, out_dir):
    plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=[_label(lang, n) for n in names],
                      max_display=14, show=False, plot_size=(7.5, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_xlabel(_label(lang, "_shap_x"), fontsize=9)
    if len(fig.axes) > 1:
        fig.axes[-1].set_ylabel(_label(lang, "_feat_val"), fontsize=9)
    plt.title(_label(lang, "_beeswarm_t"), fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()


def importance_bar(shap_values, names, lang, out_dir):
    mean_abs = np.abs(shap_values).mean(0)
    order = np.argsort(mean_abs)[::-1][:14][::-1]
    labels = [_label(lang, names[i]) for i in order]
    plt.figure(figsize=(6.8, 4.8))
    plt.barh(labels, mean_abs[order], color="#8e44ad")
    plt.xlabel(_label(lang, "_imp_x"), fontsize=9)
    plt.title(_label(lang, "_imp_t"), fontsize=10)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance_bar.png", dpi=300)
    plt.close()
    return {names[i]: float(mean_abs[i]) for i in np.argsort(mean_abs)[::-1]}


def dependence(shap_values, names, x_raw, color_vals, key_x, lab_x, lab_y, lab_c, title, fname, lang, out_dir):
    idx = names.index(key_x)
    sv = shap_values[:, idx]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sc = ax.scatter(x_raw, sv, c=color_vals, cmap="viridis", s=10, alpha=0.6)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(_label(lang, lab_c), fontsize=8)
    ax.axhline(0, color="#7f8c8d", linewidth=1, linestyle="--")
    ax.set_xlabel(_label(lang, lab_x), fontsize=9)
    ax.set_ylabel(_label(lang, lab_y), fontsize=9)
    ax.set_title(_label(lang, title), fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300)
    plt.close()


def main():
    df = add_sequential_features(load_dataset())
    y = df["handoff_next"].astype(int)
    groups = df["session_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(df, y, groups=groups))

    pipe = make_pipeline(y.iloc[tr])
    pipe.fit(df.iloc[tr][ALL_COLS], y.iloc[tr])

    pre = pipe.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    X_test_trans = pre.transform(df.iloc[te][ALL_COLS])
    if hasattr(X_test_trans, "toarray"):
        X_test_trans = X_test_trans.toarray()
    X_test_trans = np.asarray(X_test_trans, dtype=np.float32)

    print("Computing SHAP values on", X_test_trans.shape[0], "holdout rows ...")
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_test_trans)

    # Raw (interpretable) values for the dependence plots.
    test_raw = df.iloc[te]
    dwell_raw = test_raw["dwell_time_s"].to_numpy()
    ho_raw = test_raw["handoff_history_30s"].to_numpy()
    cid_te = X_test_trans[:, names.index("te__cid")]

    ranking = {}
    lang, out_dir = "ro", FIG_OUT
    print(f"Figures -> {out_dir}")
    beeswarm(shap_values, X_test_trans, names, lang, out_dir)
    ranking = importance_bar(shap_values, names, lang, out_dir)
    dependence(shap_values, names, dwell_raw, cid_te, "num__dwell_time_s",
               "_dwell_x", "_dwell_y", "_dwell_c", "_dwell_t",
               "shap_dependence_dwell_time.png", lang, out_dir)
    dependence(shap_values, names, ho_raw, dwell_raw, "num__handoff_history_30s",
               "_ho_x", "_ho_y", "_dwell_x", "_ho_t",
               "shap_dependence_handoff_history.png", lang, out_dir)

    top = list(ranking.items())[:10]
    (REPORTS_TEXT / "explainability.json").write_text(
        json.dumps({"base_value": float(explainer.expected_value),
                    "n_rows": int(X_test_trans.shape[0]),
                    "mean_abs_shap_top10": top}, indent=2), encoding="utf-8")
    print("\nTop-10 mean|SHAP|:")
    for k, v in top:
        print(f"   {k:32s} {v:.4f}")
    print("Saved JSON ->", REPORTS_TEXT / "explainability.json")


if __name__ == "__main__":
    main()
