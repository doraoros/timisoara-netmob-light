"""Sequential deep-learning baselines (GRU, Transformer-lite) vs the engineered
XGBoost (Stage 2) model for imminent serving-cell-change prediction.

Scientific question: the Stage 2 tabular model needs hand-crafted rolling
features (dwell time, handover history, rolling speed). Can a sequence model
learn that temporal structure directly from *raw per-timestep* signals, and
does it match or beat the engineered XGBoost on the same held-out sessions?

Design (leakage-safe):
* Per-timestep features are raw/causal only (speed, cyclic time, flags, and the
  observable current handover indicator); NO hand-crafted rolling features.
* For each row t we build the window [t-L+1, t] within the same session
  (left zero-padded at the start) and predict handoff_next(t) (future 15 s).
* Train/test split is by whole sessions (same GroupShuffleSplit as the other
  experiments), so no session leaks across the split.
* Standardization uses training statistics only.
* Class imbalance handled via BCEWithLogitsLoss pos_weight = N_neg / N_pos.

Runs on CPU. Requires torch (and xgboost for the tabular reference).

Run:
    set PYTHONPATH=D:\\project   (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/experiment_sequential.py
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.config import RANDOM_STATE, REPORTS_FIGURES, REPORTS_TEXT
from src.models.train_handoff_models import load_dataset

OUT_FIG = REPORTS_FIGURES / "thesis"
OUT_FIG.mkdir(parents=True, exist_ok=True)
JSON_PATH = REPORTS_TEXT / "sequential_experiment.json"

SEQ_LEN = 30          # last 30 s of context
EPOCHS = 30
BATCH = 256
LR = 1e-3
HIDDEN = 64

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Raw, causal per-timestep features (NO hand-crafted rolling windows).
RAW_NUMERIC = [
    "speed_kmh", "frequency", "area", "is_peak_hour", "is_high_speed",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
    "handoff",
]


def add_cyclic(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def one_hot_static(df, cols):
    return pd.get_dummies(df[cols].astype(str), prefix=cols).astype(np.float32)


def build_sequences(df, feat_cols, seq_len):
    """Return X [N, L, F], y [N], groups [N] with causal left-padded windows."""
    X_list, y_list, g_list = [], [], []
    feats = df[feat_cols].to_numpy(dtype=np.float32)
    target = df["handoff_next"].to_numpy(dtype=np.float32)
    sessions = df["session_id"].to_numpy()
    for sid in pd.unique(sessions):
        idx = np.where(sessions == sid)[0]
        f = feats[idx]
        t = target[idx]
        n, dim = f.shape
        for i in range(n):
            lo = max(0, i - seq_len + 1)
            window = f[lo:i + 1]
            if len(window) < seq_len:
                pad = np.zeros((seq_len - len(window), dim), dtype=np.float32)
                window = np.vstack([pad, window])
            X_list.append(window)
            y_list.append(t[i])
            g_list.append(sid)
    return np.stack(X_list), np.array(y_list, dtype=np.float32), np.array(g_list)


class GRUClassifier(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, n_feat, d_model=HIDDEN, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                         dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


def train_eval(model, Xtr, ytr, Xte, yte, pos_weight, name):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)
    for ep in range(EPOCHS):
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"    [{name}] epoch {ep + 1}/{EPOCHS} loss={tot / len(Xtr):.4f}")
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.from_numpy(Xte))).numpy()
    return {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def main():
    df = load_dataset()
    df = add_cyclic(df)
    static = one_hot_static(df, ["operator", "technology", "transport_mode"])
    df = pd.concat([df.reset_index(drop=True), static.reset_index(drop=True)], axis=1)
    feat_cols = RAW_NUMERIC + list(static.columns)
    print(f"Per-timestep features: {len(feat_cols)} | seq_len={SEQ_LEN}")

    X, y, g = build_sequences(df, feat_cols, SEQ_LEN)
    print("Sequences:", X.shape)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(X, y, groups=g))

    # Standardize with train statistics (flatten over time for per-feature stats).
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xs = (X - mu) / sd
    Xtr, Xte, ytr, yte = Xs[tr], Xs[te], y[tr], y[te]

    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    pw = neg / pos
    print(f"Train pos_weight={pw:.2f} | holdout prevalence={yte.mean():.3f}")

    n_feat = X.shape[-1]
    results = {}
    print("Training GRU ...")
    results["GRU"] = train_eval(GRUClassifier(n_feat), Xtr, ytr, Xte, yte, pw, "GRU")
    print("   GRU:", {k: round(v, 3) for k, v in results["GRU"].items()})
    print("Training Transformer-lite ...")
    results["Transformer"] = train_eval(TransformerClassifier(n_feat), Xtr, ytr, Xte, yte, pw, "Transformer")
    print("   Transformer:", {k: round(v, 3) for k, v in results["Transformer"].items()})

    # Tabular reference (engineered XGBoost, Stage 2) from the prior experiment.
    fe = json.loads((REPORTS_TEXT / "feature_engineering_experiment.json").read_text(encoding="utf-8"))
    xgb_h = fe["stage2"]["results"]["XGBoost"]["holdout"]
    results["XGBoost (engineered, Stage 2)"] = {
        "roc_auc": xgb_h["roc_auc"], "pr_auc": xgb_h["pr_auc"], "n_params": None}
    print("   XGBoost ref:", {k: round(v, 3) for k, v in xgb_h.items() if k in ("roc_auc", "pr_auc")})

    # Comparison figure.
    order = ["XGBoost (engineered, Stage 2)", "GRU", "Transformer"]
    short = {"XGBoost (engineered, Stage 2)": "XGBoost\n(engineered)", "GRU": "GRU\n(raw seq.)",
             "Transformer": "Transformer\n(raw seq.)"}
    roc = [results[m]["roc_auc"] for m in order]
    pr = [results[m]["pr_auc"] for m in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    b1 = ax.bar(x - 0.2, roc, 0.4, label="ROC-AUC", color="#2980b9")
    b2 = ax.bar(x + 0.2, pr, 0.4, label="PR-AUC", color="#e67e22")
    ax.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short[m] for m in order])
    ax.set_ylabel("Holdout score (unseen sessions)")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "sequential_comparison.png", dpi=300)
    plt.close()

    JSON_PATH.write_text(json.dumps({
        "seq_len": SEQ_LEN, "epochs": EPOCHS, "n_per_timestep_features": n_feat,
        "results": results,
    }, indent=2), encoding="utf-8")
    print("\nSaved:", JSON_PATH)
    print("Figure -> sequential_comparison.png")


if __name__ == "__main__":
    main()
