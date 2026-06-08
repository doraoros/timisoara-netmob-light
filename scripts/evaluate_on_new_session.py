"""Leave-one-session-out evaluation CLI.

Trains the Stage 2 XGBoost model on every session except the chosen one, then
reports metrics + figures on that fully unseen session. This is the strongest
practical demonstration of generalization to a new measurement run.

Run:
    set PYTHONPATH=D:\\project     (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/evaluate_on_new_session.py --session S10
    python scripts/evaluate_on_new_session.py --list
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import REPORTS_FIGURES, REPORTS_TEXT
from src.models.stage2_pipeline import (
    GROUP,
    TARGET,
    build_pipeline,
    feature_columns,
    prepare_dataframe,
)

OUT_FIG = REPORTS_FIGURES / "ml" / "per_session_eval"


def main():
    p = argparse.ArgumentParser(description="Evaluate on a single unseen session.")
    p.add_argument("--session", type=str, help="session_id to hold out for evaluation.")
    p.add_argument("--list", action="store_true", help="List available sessions and exit.")
    args = p.parse_args()

    df = prepare_dataframe()
    sessions = sorted(df[GROUP].unique().tolist())

    if args.list or not args.session:
        print("Available sessions:")
        for s in sessions:
            n = int((df[GROUP] == s).sum())
            prev = float(df.loc[df[GROUP] == s, TARGET].mean())
            print(f"  {s:<10} rows={n:<6} handoff_next prevalence={prev:.3f}")
        if not args.session:
            return

    if args.session not in sessions:
        raise SystemExit(f"Unknown session '{args.session}'. Use --list to see options.")

    cols = feature_columns(df)
    is_test = df[GROUP] == args.session
    train, test = df[~is_test], df[is_test]

    y_tr = train[TARGET].astype(int)
    y_te = test[TARGET].astype(int)

    pipe = build_pipeline(df, y_tr)
    pipe.fit(train[cols], y_tr)

    proba = pipe.predict_proba(test[cols])[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "session": args.session,
        "n_rows": int(len(test)),
        "prevalence": float(y_te.mean()),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
    }
    if y_te.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_te, proba))
        metrics["pr_auc"] = float(average_precision_score(y_te, proba))

    print(f"\n=== Unseen session: {args.session} ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_te, pred, zero_division=0))

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    safe = str(args.session).replace(" ", "_")

    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — session {args.session}")
    plt.tight_layout()
    plt.savefig(OUT_FIG / f"confusion_{safe}.png", dpi=200)
    plt.close()

    if y_te.nunique() > 1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax, name=args.session)
        ax.axhline(float(y_te.mean()), color="grey", ls="--", lw=1, label="prevalence")
        ax.set_title(f"Precision-Recall — session {args.session}")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_FIG / f"pr_curve_{safe}.png", dpi=200)
        plt.close()

    metrics_path = REPORTS_TEXT / f"eval_session_{safe}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved figures: {OUT_FIG}")


if __name__ == "__main__":
    main()
