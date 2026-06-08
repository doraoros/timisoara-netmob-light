import folium
import pandas as pd

def color_scale_rsrp(x):
    if x is None or pd.isna(x): return "gray"
    return "green" if x>-95 else ("orange" if x>-110 else "red")

def color_scale_dl(x):
    if x is None or pd.isna(x): return "gray"
    return "red" if x<10 else ("orange" if x<50 else "green")

def _with_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "lat" not in d.columns and "latitude" in d.columns:
        d["lat"] = d["latitude"]
    if "lon" not in d.columns and "longitude" in d.columns:
        d["lon"] = d["longitude"]
    return d


def map_points(df: pd.DataFrame, value_col: str, out_html: str, zoom=13):
    dfm = _with_lat_lon(df).dropna(subset=["lat", "lon"])
    if dfm.empty:
        return None
    center = [dfm["lat"].mean(), dfm["lon"].mean()]
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")
    for _, r in dfm.iterrows():
        v = r.get(value_col)
        color = color_scale_rsrp(v) if value_col.lower().startswith("rsrp") else color_scale_dl(v)
        folium.CircleMarker([r["lat"], r["lon"]], radius=3, color=color, fill=True, fill_opacity=0.7).add_to(m)
    m.save(out_html)
    return out_html
