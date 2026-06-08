"""Classic interpretability for the Stage 2 model (complements the SHAP study).

Produces two transparent, model-agnostic views that committees expect:

1. Standardized logistic-regression coefficients (sign + magnitude of each
   feature's linear effect) -> CSV + bar chart.
2. Partial dependence plots (PDP) for the strongest numeric drivers, showing
   how the predicted handoff probability changes with each feature.

Figures are written to ``reports/figures/thesis/``. SHAP figures are
generated separately by scripts/experiment_explainability.py.

Run:
    set PYTHONPATH=D:\\project     (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/interpretability_classic.py
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.config import RANDOM_STATE, REPORTS_FIGURES, REPORTS_TEXT, TEST_SIZE
from src.models.stage2_pipeline import (
    GROUP,
    TARGET,
    build_pipeline,
    feature_columns,
    prepare_dataframe,
    stage2_columns,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

THESIS = REPORTS_FIGURES / "thesis"

PDP_FEATURES = ["dwell_time_s", "handoff_history_30s", "speed_kmh", "rolling_mean_speed_15s"]

NICE = {
    "num__speed_kmh": "Speed (km/h)",
    "num__rolling_mean_speed_15s": "Rolling mean speed (15s)",
    "num__rolling_std_speed_15s": "Speed std (15s)",
    "num__speed_delta_5s": "Speed delta (5s)",
    "num__handoff_history_30s": "Handovers last 30s",
    "num__handoff_history_60s": "Handovers last 60s",
    "num__dwell_time_s": "Dwell time on cell",
    "num__unique_pci_last_60s": "Distinct PCIs (60s)",
    "num__hour_sin": "Hour (sin)", "num__hour_cos": "Hour (cos)",
    "num__minute_sin": "Minute (sin)", "num__minute_cos": "Minute (cos)",
    "num__dow_sin": "Day of week (sin)", "num__dow_cos": "Day of week (cos)",
    "num__frequency": "Frequency", "num__area": "Tracking area",
    "num__is_peak_hour": "Peak hour", "num__is_high_speed": "High speed",
    "te__cid": "Cell CID (target-enc.)", "te__code": "Cell PCI (target-enc.)",
}


def _scaled_preprocessor(numeric, target_enc, categorical):
    transformers = [("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric)]
    if target_enc:
        transformers.append(
            ("te", TargetEncoder(target_type="binary", random_state=RANDOM_STATE), target_enc))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)


def logistic_coefficients(df, X_tr, y_tr):
    numeric, target_enc, categorical = stage2_columns(df)
    preproc = _scaled_preprocessor(numeric, target_enc, categorical)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([("preprocessor", preproc), ("model", lr)])
    pipe.fit(X_tr, y_tr)

    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    table = (pd.DataFrame({"feature": names, "coefficient": coef})
             .assign(abs=lambda d: d["coefficient"].abs())
             .sort_values("abs", ascending=False))
    table.drop(columns="abs").to_csv(REPORTS_TEXT / "logreg_coefficients_stage2.csv", index=False)

    top = table.head(15).iloc[::-1]
    labels = [NICE.get(f, f.split("__")[-1]) for f in top["feature"]]
    colors = ["#c0392b" if c > 0 else "#2980b9" for c in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, top["coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.set_title("Linear effect on imminent-handoff log-odds (Stage 2)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    THESIS.mkdir(parents=True, exist_ok=True)
    plt.savefig(THESIS / "logreg_coefficients.png", dpi=300)
    plt.close()
    print("Saved LR coefficients (CSV + figure).")


def partial_dependence(df, X_tr, y_tr):
    pipe = build_pipeline(df, y_tr)
    pipe.fit(X_tr, y_tr)
    feats = [f for f in PDP_FEATURES if f in X_tr.columns]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    PartialDependenceDisplay.from_estimator(
        pipe, X_tr, features=feats, ax=ax.ravel()[:len(feats)],
        grid_resolution=40, n_jobs=-1)
    fig.suptitle("Partial dependence — XGBoost (Stage 2)")
    plt.tight_layout()
    THESIS.mkdir(parents=True, exist_ok=True)
    plt.savefig(THESIS / "partial_dependence.png", dpi=300)
    plt.close()
    print("Saved partial dependence plots.")


def main():
    df = prepare_dataframe()
    cols = feature_columns(df)
    X, y, groups = df[cols], df[TARGET].astype(int), df[GROUP]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, _ = next(splitter.split(X, y, groups=groups))
    X_tr, y_tr = X.iloc[tr], y.iloc[tr]

    logistic_coefficients(df, X_tr, y_tr)
    partial_dependence(df, X_tr, y_tr)

    print("Done. Figures in reports/figures/thesis/")


if __name__ == "__main__":
    main()
