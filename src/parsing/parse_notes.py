"""Parsează fișierele JSON NetMonster salvate manual în data/raw/notes/."""

import json
from pathlib import Path

import pandas as pd

from src.config import OPERATOR_MAP


def load_json_cells(json_path: str | Path) -> pd.DataFrame:
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    data = json.loads(content)
    if isinstance(data, dict):
        data = [data]

    rows = []
    for cell in data:
        network = cell.get("network", {})
        mnc = network.get("mnc")
        rows.append(
            {
                "mcc": network.get("mcc"),
                "mnc": mnc,
                "operator": OPERATOR_MAP.get(mnc, "Unknown"),
                "technology": cell.get("technology"),
                "cid": cell.get("cid"),
                "area": cell.get("area"),
                "code": cell.get("code"),
                "frequency": cell.get("frequency"),
                "netmonster_latitude": cell.get("latitude"),
                "netmonster_longitude": cell.get("longitude"),
                "netmonster_location": cell.get("location"),
            }
        )

    return pd.DataFrame(rows)
