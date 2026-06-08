"""Interactive route map (HTML) for demo and presentation.

Each session is drawn as its own animated path; colour follows route_id
(same logic as the static PNG in generate_route_map.py).

Run:
    set PYTHONPATH=D:\\project     (PowerShell: $env:PYTHONPATH="D:\\project")
    python scripts/generate_interactive_route_map.py
"""

from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import AntPath, Fullscreen, MiniMap, MeasureControl

from src.config import CITY_CENTER, DATASET_RADIO_GPS, REPORTS_FIGURES

COLORS = [
    "#00E5FF",
    "#FF4081",
    "#7CFF00",
    "#FFD740",
    "#B388FF",
    "#FF6E40",
    "#69F0AE",
    "#40C4FF",
    "#FF5252",
    "#18FFFF",
]


def _load() -> pd.DataFrame:
    cols = ["timestamp", "latitude", "longitude", "route_id", "session_id"]
    df = pd.read_csv(DATASET_RADIO_GPS, usecols=cols)
    df = df.dropna(subset=["latitude", "longitude", "route_id", "session_id"])
    df = df[(df["latitude"].abs() < 1e6) & (df["longitude"].abs() < 1e6)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values(["route_id", "session_id", "timestamp"])


def generate_interactive_map() -> None:
    df = _load()
    routes = sorted(df["route_id"].unique().tolist())
    route_color = {r: COLORS[i % len(COLORS)] for i, r in enumerate(routes)}

    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

    out_dir = REPORTS_FIGURES / "interactive"
    out_dir.mkdir(parents=True, exist_ok=True)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    MeasureControl(position="topleft").add_to(m)

    route_layers: dict[str, folium.FeatureGroup] = {}
    for route in routes:
        fg = folium.FeatureGroup(name=str(route), show=True)
        fg.add_to(m)
        route_layers[route] = fg

    for (route, session_id), seg in df.groupby(["route_id", "session_id"], sort=False):
        points = seg[["latitude", "longitude"]].values.tolist()
        if len(points) < 2:
            continue

        color = route_color[route]
        label = f"{route} · {session_id}"
        layer = route_layers[route]

        AntPath(
            locations=points,
            color=color,
            weight=4,
            opacity=0.85,
            delay=800,
            dash_array=[10, 20],
            pulse_color="#FFFFFF",
        ).add_to(layer)

        start, end = points[0], points[-1]

        folium.CircleMarker(
            location=start,
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"Start: {label}",
        ).add_to(layer)

        folium.CircleMarker(
            location=end,
            radius=6,
            color="#FFFFFF",
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"End: {label}",
        ).add_to(layer)

    folium.Marker(
        location=[CITY_CENTER[0], CITY_CENTER[1]],
        tooltip="Timișoara",
        icon=folium.Icon(color="orange", icon="star"),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    n_sessions = df["session_id"].nunique()
    title_html = f"""
    <div style="
        position: fixed;
        top: 20px;
        left: 50px;
        z-index: 9999;
        background: rgba(0, 0, 0, 0.75);
        color: white;
        padding: 14px 18px;
        border-radius: 12px;
        font-family: Arial, sans-serif;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    ">
        <div style="font-size: 20px; font-weight: bold;">
            Timișoara NetMob Light
        </div>
        <div style="font-size: 13px; margin-top: 4px;">
            {len(routes)} routes · {n_sessions} sessions · synthetic GPS traces
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    output_path = out_dir / "interactive_route_map.html"
    m.save(str(output_path))
    print(f"Saved: {output_path}")


def main() -> None:
    generate_interactive_map()


if __name__ == "__main__":
    main()
