"""Figuri și tabele pentru capitolul de rezultate (disertație)."""

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    DATASET_RADIO_GPS,
    HANDOFF_METRICS_JSON,
    HO_HORIZON_S,
    REPORTS_FIGURES,
    REPORTS_TEXT,
)
from src.utils.validate_dataset import summarize_dataset

sns.set_theme(style="whitegrid", font_scale=1.05)
THESIS_FIG = REPORTS_FIGURES / "thesis"
THESIS_FIG.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = THESIS_FIG / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_methodological_positioning():
    """Fig. 3.1 — Poziționare față de literatură (capitol Metodologie, §3.2.4)."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (1.0, 7.2, 8.0, 1.6, "Studii extinse din literatură",
         "iPerf continuu · RSRP/RSRQ/SINR · modele secvențiale"),
        (1.0, 5.0, 8.0, 1.6, "Adaptare locală — Timișoara",
         "NetMonster (snapshot-uri) + GPS 1 Hz + Speedtest punctual"),
        (1.0, 2.8, 8.0, 1.6, "Dataset tabular reproductibil",
         "context radio + mobilitate + timp · pipeline open-source"),
        (1.0, 0.6, 8.0, 1.6, "Predicție handoff iminent",
         "clasificare binară · validare pe sesiuni noi (GroupCV)"),
    ]
    colors = ["#ecf0f1", "#d6eaf8", "#d5f5e3", "#fdebd0"]
    edge = "#2c3e50"

    for (x, y, w, h, title, subtitle), fill in zip(boxes, colors):
        rect = plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=9, color="#34495e")

    arrow_x = 5.0
    for y_from, y_to in [(7.2, 6.6), (5.0, 4.4), (2.8, 2.2)]:
        ax.annotate(
            "",
            xy=(arrow_x, y_to + 1.6),
            xytext=(arrow_x, y_from),
            arrowprops=dict(arrowstyle="->", color=edge, lw=2),
        )

    ax.set_title(
        "Poziționarea metodologică a studiului față de cercetările existente",
        fontsize=12,
        fontweight="bold",
        pad=16,
    )
    _save(fig, "methodological_positioning.png")


def plot_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    steps = [
        "notes/\nNetMonster",
        "gps_simulated/\nTrasee 1Hz",
        "speedtest/\nThroughput",
        "build_dataset\n+ etichete HO",
        "ML\nGroupCV",
        "reports/\nDisertație",
    ]
    x = 0.05
    for i, label in enumerate(steps):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4fc", edgecolor="#2980b9"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.14, 0.5), xytext=(x + 0.08, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#34495e"))
        x += 0.16
    ax.set_title("Pipeline Timișoara NetMob Light", fontsize=12, fontweight="bold")
    _save(fig, "pipeline_overview.png")


def plot_operator_handoff_rates(df: pd.DataFrame):
    agg = (
        df.groupby("operator", observed=True)
        .agg(handoff_next_rate=("handoff_next", "mean"), n=("handoff_next", "count"))
        .reset_index()
        .sort_values("handoff_next_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=agg, x="operator", y="handoff_next_rate", hue="operator", legend=False, ax=ax, palette="viridis")
    ax.set_ylabel(f"Rată handoff_next (orizont {HO_HORIZON_S}s)")
    ax.set_xlabel("Operator")
    ax.set_title("Probabilitate handoff iminent pe operator (Timișoara)")
    for i, row in agg.iterrows():
        ax.text(i, row["handoff_next_rate"] + 0.01, f"n={int(row['n'])}", ha="center", fontsize=9)
    _save(fig, "handoff_rate_by_operator.png")
    return agg


def plot_route_mobility(df: pd.DataFrame):
    agg = (
        df.groupby(["route_id", "transport_mode"], observed=True)
        .agg(
            mean_speed=("speed_kmh", "mean"),
            handoff_next_rate=("handoff_next", "mean"),
            n=("timestamp", "count"),
        )
        .reset_index()
        .sort_values("mean_speed", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=agg, x="route_id", y="mean_speed", hue="transport_mode", ax=ax)
    ax.set_title("Viteză medie pe rută și mod de transport")
    ax.set_xlabel("Rută")
    ax.set_ylabel("Viteză medie (km/h)")
    plt.xticks(rotation=30, ha="right")
    _save(fig, "mean_speed_by_route.png")
    return agg


def plot_session_heatmap(df: pd.DataFrame):
    agg = (
        df.groupby(["session_id", "operator"], observed=True)["handoff_next"]
        .mean()
        .reset_index()
        .pivot(index="session_id", columns="operator", values="handoff_next")
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(agg, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Rată handoff iminent pe sesiune și operator")
    _save(fig, "session_handoff_heatmap.png")
    return agg.reset_index()


def plot_feature_importance():
    path = REPORTS_TEXT / "feature_importance_rf.csv"
    if not path.exists():
        return None
    imp = pd.read_csv(path).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp = imp.sort_values("importance")
    ax.barh(imp["feature"].str.replace("num__", "").str.replace("cat__", ""), imp["importance"], color="#3498db")
    ax.set_xlabel("Importanță")
    ax.set_title("Top 15 predictori — Random Forest")
    _save(fig, "feature_importance_top15.png")
    return imp


def plot_speedtest_throughput(df: pd.DataFrame):
    sub = df[df["speedtest_matched"] == 1].copy()
    if sub.empty or sub["dl_mbps"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=sub, x="operator", y="dl_mbps", hue="operator", legend=False, ax=ax, palette="Set2")
    ax.set_title("Throughput măsurat (speedtest) — puncte sincronizate")
    ax.set_ylabel("Download (Mbps)")
    _save(fig, "throughput_by_operator_matched.png")
    return sub.groupby("operator")["dl_mbps"].agg(["count", "mean", "median"]).reset_index()


def plot_handoff_vs_throughput(df: pd.DataFrame):
    sub = df[(df["speedtest_matched"] == 1) & df["dl_mbps"].notna()].copy()
    if len(sub) < 10:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=sub, x="handoff_next", y="dl_mbps", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fără HO iminent", f"HO iminent ({HO_HORIZON_S}s)"])
    ax.set_title("Throughput vs. etichetă handoff iminent")
    ax.set_ylabel("Download (Mbps)")
    _save(fig, "throughput_vs_handoff_next.png")
    return sub.groupby("handoff_next")["dl_mbps"].agg(["count", "mean"]).reset_index()


def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["handoff"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Eveniment handoff (cid/PCI)")
    axes[0].set_xticklabels(["Nu", "Da"], rotation=0)
    df["handoff_next"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color=["#55A868", "#C44E52"])
    axes[1].set_title(f"Handoff iminent ({HO_HORIZON_S}s)")
    axes[1].set_xticklabels(["Nu", "Da"], rotation=0)
    _save(fig, "target_distribution.png")


def export_tables(df: pd.DataFrame, extras: dict):
    tables_dir = REPORTS_TEXT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in extras.items():
        if frame is not None and hasattr(frame, "to_csv"):
            frame.to_csv(tables_dir / f"{name}.csv", index=False)
    with open(REPORTS_TEXT / "thesis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summarize_dataset(df), f, indent=2, ensure_ascii=False)


def main():
    df = pd.read_csv(DATASET_RADIO_GPS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    plot_methodological_positioning()
    plot_pipeline_overview()
    extras = {
        "operator_handoff": plot_operator_handoff_rates(df),
        "route_mobility": plot_route_mobility(df),
        "session_heatmap": plot_session_heatmap(df),
        "speedtest_operator": plot_speedtest_throughput(df),
        "throughput_handoff": plot_handoff_vs_throughput(df),
        "feature_importance": plot_feature_importance(),
    }
    plot_target_distribution(df)
    export_tables(df, extras)
    print(f"Thesis figures saved to: {THESIS_FIG}")


if __name__ == "__main__":
    main()
