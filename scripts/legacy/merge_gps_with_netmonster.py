import json
import pandas as pd
from pathlib import Path


GPS_FILE = "data/raw/gps_simulated/gps_01_victoriei_michelangelo_complex.csv"
NETMONSTER_FOLDER = "data/raw/notes"
OUTPUT_FILE = "data/processed/dataset_01.csv"


def load_netmonster_json_files(folder_path):
    rows = []

    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"Cannot read JSON from: {file_path}")
            continue

        if isinstance(data, dict):
            data = [data]

        for cell in data:
            if not isinstance(cell, dict):
                continue

            network = cell.get("network", {})

            rows.append({
                "mcc": network.get("mcc"),
                "mnc": network.get("mnc"),
                "technology": cell.get("technology"),
                "cid": cell.get("cid"),
                "area": cell.get("area"),
                "code": cell.get("code"),
                "frequency": cell.get("frequency"),
                "radio_latitude": cell.get("latitude"),
                "radio_longitude": cell.get("longitude"),
                "radio_location": cell.get("location"),
                "source_file": file_path.name
            })

    return pd.DataFrame(rows)


def main():
    gps_df = pd.read_csv(GPS_FILE)
    radio_df = load_netmonster_json_files(NETMONSTER_FOLDER)

    print("GPS columns:")
    print(gps_df.columns)

    print("Radio columns:")
    print(radio_df.columns)

    if radio_df.empty:
        raise ValueError("Nu s-au gasit date NetMonster. Verifica folderul data/raw/notes.")

    if "cid" not in radio_df.columns:
        raise ValueError("Coloana cid nu exista in datele NetMonster.")

    n = min(len(gps_df), len(radio_df))

    gps_df = gps_df.iloc[:n].reset_index(drop=True)
    radio_df = radio_df.iloc[:n].reset_index(drop=True)

    final_df = pd.concat([gps_df, radio_df], axis=1)

    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
    final_df["hour"] = final_df["timestamp"].dt.hour

    final_df["operator"] = final_df["mnc"].map({
        1: "Vodafone",
        3: "Telekom",
        5: "Digi",
        10: "Orange"
    }).fillna("Unknown")

    final_df["handoff"] = final_df["cid"].ne(final_df["cid"].shift()).astype(int)
    final_df.loc[0, "handoff"] = 0

    final_df["handoff_next"] = final_df["handoff"].shift(-1).fillna(0).astype(int)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset generated successfully: {OUTPUT_FILE}")
    print(final_df.head())


if __name__ == "__main__":
    main()