import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DATASET_RADIO_GPS,
    HANDOFF_METRICS_JSON,
    RANDOM_STATE,
    REPORTS_FIGURES,
    REPORTS_FIGURES_ML,
    REPORTS_TEXT,
)


def load_dataset():
    df = pd.read_csv(DATASET_RADIO_GPS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "handoff_next" not in df.columns:
        raise ValueError("Coloana handoff_next lipsește din dataset.")

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    if "is_peak_hour" not in df.columns:
        df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    df["is_high_speed"] = (df["speed_kmh"] > 35).astype(int)
    df["cells_seen_in_session"] = df.groupby("session_id")["cid"].transform("nunique")

    return df


def build_features(df):
    numeric_features = [
        "speed_kmh",
        "frequency",
        "area",
        "code",
        "hour",
        "minute",
        "day_of_week",
        "is_peak_hour",
        "is_high_speed",
        "cells_seen_in_session",
        "radio_cell_index",
    ]
    categorical_features = [
        "route_id",
        "transport_mode",
        "operator",
        "technology",
    ]

    existing_numeric = [c for c in numeric_features if c in df.columns]
    existing_categorical = [c for c in categorical_features if c in df.columns]

    X = df[existing_numeric + existing_categorical].copy()
    y = df["handoff_next"].astype(int)
    groups = df["session_id"] if "session_id" in df.columns else df["route_id"]

    return X, y, groups, existing_numeric, existing_categorical


def create_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def score_predictions(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return metrics


def evaluate_model(name, model, X_test, y_test, save_cm=True):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    metrics = score_predictions(y_test, y_pred, y_proba)

    print(f"\n=== {name} ===")
    print(metrics)
    print(classification_report(y_test, y_pred, zero_division=0))

    if save_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        safe = name.lower().replace(" ", "_")
        plt.savefig(REPORTS_FIGURES_ML / f"confusion_matrix_{safe}.png", dpi=300)
        plt.close()

    return metrics, y_proba


def _export_metric_tables(all_metrics: dict):
    tables_dir = REPORTS_TEXT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    holdout_rows, cv_rows = [], []
    for name, block in all_metrics.items():
        if not isinstance(block, dict) or "holdout" not in block:
            continue
        holdout_rows.append({"model": name, **block["holdout"]})
        if block.get("cv_mean"):
            row = {"model": name, **block["cv_mean"]}
            for k, v in (block.get("cv_std") or {}).items():
                row[f"{k}_std"] = v
            cv_rows.append(row)

    if holdout_rows:
        pd.DataFrame(holdout_rows).to_csv(tables_dir / "model_metrics_holdout.csv", index=False)
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(tables_dir / "model_metrics_cv.csv", index=False)


def _plot_roc_curves(roc_data: list, y_test):
    if not roc_data or len(np.unique(y_test)) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_proba in roc_data:
        if y_proba is None:
            continue
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC curves — holdout (sesiuni noi)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_ML / "roc_curves_holdout.png", dpi=300)
    plt.close()


def _plot_model_comparison(all_metrics: dict):
    rows = []
    for name, block in all_metrics.items():
        if isinstance(block, dict) and "holdout" in block and "pr_auc" in block["holdout"]:
            rows.append({"model": name, "pr_auc": block["holdout"]["pr_auc"]})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["model"], df["pr_auc"], color="#2ecc71")
    ax.set_xlabel("PR-AUC (holdout)")
    ax.set_title("Comparație modele — predicție handoff iminent")
    plt.tight_layout()
    thesis_dir = REPORTS_FIGURES / "thesis"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(thesis_dir / "model_comparison_pr_auc.png", dpi=300)
    plt.close()


def cross_validate_model(name, pipeline, X, y, groups, n_splits=5):
    """Return (cv_mean, cv_std) dicts over GroupKFold folds."""
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return {}, {}

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = pipeline.predict(X.iloc[test_idx])
        y_proba = (
            pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )
        fold_metrics.append(score_predictions(y.iloc[test_idx], y_pred, y_proba))

    keys = fold_metrics[0].keys()
    cv_mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    cv_std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return cv_mean, cv_std


def main():
    REPORTS_FIGURES_ML.mkdir(parents=True, exist_ok=True)
    HANDOFF_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print("Dataset shape:", df.shape)
    print("Target distribution:\n", df["handoff_next"].value_counts(normalize=True))

    X, y, groups, num_f, cat_f = build_features(df)
    preprocessor = create_preprocessor(num_f, cat_f)

    test_prevalence = float(y.mean())
    all_metrics = {
        "test_prevalence_handoff_next": test_prevalence,
        "n_sessions": int(groups.nunique()),
        "n_rows": int(len(df)),
    }

    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Baseline (stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Decision Tree (depth 3)": DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    roc_data = []
    for model_name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        holdout, y_proba = evaluate_model(model_name, pipeline, X_test, y_test)
        cv_mean, cv_std = cross_validate_model(model_name, pipeline, X, y, groups)
        all_metrics[model_name] = {"holdout": holdout, "cv_mean": cv_mean, "cv_std": cv_std}
        if y_proba is not None and "Baseline" not in model_name:
            roc_data.append((model_name, y_proba))

    _plot_roc_curves(roc_data, y_test)
    _plot_model_comparison(all_metrics)
    _export_metric_tables(all_metrics)

    # Importanță features (RF)
    rf_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipe.fit(X_train, y_train)
    rf_model = rf_pipe.named_steps["model"]
    feat_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pd.DataFrame({"feature": feat_names, "importance": rf_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(20)
    imp_path = REPORTS_TEXT / "feature_importance_rf.csv"
    imp.to_csv(imp_path, index=False)

    with open(HANDOFF_METRICS_JSON, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    print(f"\nMetrics: {HANDOFF_METRICS_JSON}")
    print(f"Feature importance: {imp_path}")
    print(f"Figures: {REPORTS_FIGURES_ML}")


if __name__ == "__main__":
    main()
