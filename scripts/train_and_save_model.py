"""Train the Stage 2 XGBoost handoff model and save it as a reusable artifact.

The artifact (joblib) plus a small JSON sidecar (feature columns + per-column
defaults) are consumed by ``scripts/predict.py``.

Run:
    set PYTHONPATH=D:\\project       (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/train_and_save_model.py
"""

from __future__ import annotations

import json

import joblib
from sklearn.model_selection import GroupShuffleSplit

from src.config import BASE_DIR, RANDOM_STATE, TEST_SIZE
from src.models.stage2_pipeline import (
    GROUP,
    TARGET,
    build_pipeline,
    column_defaults,
    feature_columns,
    prepare_dataframe,
)

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "handoff_xgb_stage2.joblib"
META_PATH = MODELS_DIR / "handoff_xgb_stage2.meta.json"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe()
    cols = feature_columns(df)
    X, y, groups = df[cols], df[TARGET].astype(int), df[GROUP]

    # Train on a per-session split so the saved model is validated on unseen
    # sessions; refit on all data afterwards for deployment.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(X, y, groups=groups))

    pipe = build_pipeline(df, y.iloc[tr])
    pipe.fit(X.iloc[tr], y.iloc[tr])

    from sklearn.metrics import average_precision_score, roc_auc_score

    proba = pipe.predict_proba(X.iloc[te])[:, 1]
    holdout = {
        "roc_auc": float(roc_auc_score(y.iloc[te], proba)),
        "pr_auc": float(average_precision_score(y.iloc[te], proba)),
        "n_test_sessions": int(groups.iloc[te].nunique()),
    }
    print("Holdout (unseen sessions):", holdout)

    # Refit on the full dataset for the deployed artifact.
    pipe_full = build_pipeline(df, y)
    pipe_full.fit(X, y)
    joblib.dump(pipe_full, MODEL_PATH)

    meta = {
        "model": "XGBoost (Stage 2 engineered features)",
        "target": TARGET,
        "feature_columns": cols,
        "defaults": column_defaults(df),
        "holdout_metrics": holdout,
        "random_state": RANDOM_STATE,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model:    {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
