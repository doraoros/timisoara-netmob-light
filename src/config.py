from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_yaml_config():
    """Load config.yaml from the project root if PyYAML is available.

    Returns an empty dict on any failure so that the hardcoded defaults below
    always remain valid (non-breaking).
    """
    cfg_path = BASE_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # optional dependency
    except Exception:
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


_CFG = _load_yaml_config()

# Date brute
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_RAW_NOTES = DATA_RAW / "notes"          # export NetMonster (.txt JSON)
DATA_RAW_GPS = DATA_RAW / "gps_simulated"    # trasee GPS sintetice
DATA_RAW_NETMONSTER = DATA_RAW / "netmonster"  # export NetMonster app (log.json / .csv)
DATA_RAW_SPEEDTEST = DATA_RAW / "speedtest"

# Artefacte intermediare / finale
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

SESSIONS_CSV = DATA_INTERIM / "sessions.csv"
DATASET_RADIO_GPS = DATA_PROCESSED / "dataset_radio_gps.csv"
SPEEDTEST_ALL = DATA_PROCESSED / "speedtest_all.csv"
MERGED_SESSIONS_PARQUET = DATA_INTERIM / "merged_sessions.parquet"
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"

DISSERTATION_DIR = BASE_DIR / "dissertation"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_FIGURES = REPORTS_DIR / "figures"
REPORTS_FIGURES_ML = REPORTS_FIGURES / "ml"
REPORTS_TEXT = REPORTS_DIR / "text"
HANDOFF_METRICS_JSON = REPORTS_TEXT / "handoff_model_metrics.json"

_tol = _CFG.get("join_tolerances", {})
TOLERANCE_GPS = _tol.get("gps", "2s")
TOLERANCE_SPEEDTEST = _tol.get("speedtest", "10s")

_ho = _CFG.get("handoff", {})
HO_HORIZON_S = _ho.get("horizon_s", 15)
PEAK_HOURS = _ho.get("peak_hours", [7, 8, 9, 16, 17, 18, 19])
HIGH_SPEED_KMH = _ho.get("high_speed_kmh", 35)

_val = _CFG.get("validation", {})
TEST_SIZE = _val.get("test_size", 0.25)
CV_FOLDS = _val.get("cv_folds", 5)

CITY_CENTER = tuple(_CFG.get("city_center", [45.7542, 21.2260]))  # Timișoara (aprox.)
RANDOM_STATE = _CFG.get("seed", 42)

OPERATOR_MAP = {
    1: "Orange",
    3: "Telekom",
    5: "Digi",
    10: "Vodafone",
}
