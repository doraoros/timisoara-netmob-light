"""Static map of the measurement routes (PNG + PDF).

Colour = route_id; one polyline per session_id (no artificial joins between
sessions on the same route). Dark styling for thesis figures; fully
offline (no map tiles).

Run:
    set PYTHONPATH=D:\\project     (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/generate_route_map.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from src.config import CITY_CENTER, DATASET_RADIO_GPS, REPORTS_FIGURES

BG_FIG = "#111111"
BG_AX = "#181818"
GRID_ALPHA = 0.12


def _load() -> pd.DataFrame:
    cols = ["timestamp", "latitude", "longitude", "route_id", "session_id"]
    df = pd.read_csv(DATASET_RADIO_GPS, usecols=cols)
    df = df.dropna(subset=["latitude", "longitude", "route_id", "session_id"])
    df = df[(df["latitude"].abs() < 1e6) & (df["longitude"].abs() < 1e6)]
    return df


def _style_dark(ax, fig):
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors="white", labelsize=10)
    ax.grid(color="white", alpha=GRID_ALPHA, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _plot(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, legend_title: str, stem):
    routes = sorted(df["route_id"].unique().tolist())
    colors = plt.cm.tab10.colors if len(routes) <= 10 else plt.cm.tab20.colors
    route_color = {r: colors[i % len(colors)] for i, r in enumerate(routes)}

    fig, ax = plt.subplots(figsize=(12, 10))
    _style_dark(ax, fig)

    for (route, _sid), seg in df.groupby(["route_id", "session_id"], sort=False):
        seg = seg.sort_values("timestamp")
        c = route_color[route]
        ax.plot(
            seg["longitude"], seg["latitude"],
            linewidth=2.8, color=c, alpha=0.9, zorder=2,
        )
        ax.scatter(
            seg["longitude"].iloc[0], seg["latitude"].iloc[0],
            s=45, color=c, edgecolor="white", linewidth=0.8,
            marker="o", zorder=5,
        )
        ax.scatter(
            seg["longitude"].iloc[-1], seg["latitude"].iloc[-1],
            s=55, color=c, edgecolor="white", linewidth=0.8,
            marker="s", zorder=5,
        )

    ax.scatter(
        CITY_CENTER[1], CITY_CENTER[0],
        s=280, marker="*", color="#FFD700",
        edgecolor="black", linewidth=1.2, zorder=10,
    )

    n_sessions = df["session_id"].nunique()
    ax.set_title(title, fontsize=18, pad=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")

    handles = [Line2D([0], [0], color=route_color[r], lw=2.8, label=r) for r in routes]
    handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
               markeredgecolor="black", markersize=12, linestyle="None",
               label="Timișoara" if "Timi" in title else "Timisoara")
    )
    legend = ax.legend(
        handles=handles,
        title=f"{legend_title} · {len(routes)} rute, {n_sessions} sesiuni"
        if legend_title == "Rută"
        else f"{legend_title} · {len(routes)} routes, {n_sessions} sessions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )
    legend.get_frame().set_facecolor("#222222")
    legend.get_frame().set_edgecolor("#555555")
    legend.get_title().set_color("white")
    for text in legend.get_texts():
        text.set_color("white")

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print("Saved:", out)
    plt.close(fig)


def main():
    df = _load()

    _plot(
        df,
        title="Rute de măsurare — Timișoara",
        xlabel="Longitudine",
        ylabel="Latitudine",
        legend_title="Rută",
        stem=REPORTS_FIGURES / "thesis" / "route_map",
    )


if __name__ == "__main__":
    main()
