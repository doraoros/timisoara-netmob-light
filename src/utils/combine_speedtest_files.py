import pandas as pd
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_RAW_SPEEDTEST, SPEEDTEST_ALL


INPUT_DIR = DATA_RAW_SPEEDTEST
OUTPUT_FILE = SPEEDTEST_ALL


def clean_speedtest_file(file_path):
    df = pd.read_csv(file_path)

    df["source_file"] = file_path.name

    df = df.rename(columns={
        "ConnType": "connection_type",
        "Lat": "latitude",
        "Lon": "longitude",
        "Download Speed": "download_mbps",
        "Download Size": "download_size",
        "Upload Speed": "upload_mbps",
        "Upload Size": "upload_size",
        "Latency": "latency_ms",
        "Server": "server",
        "Internal IP": "internal_ip",
        "External IP": "external_ip",
        "URL": "url"
    })

    if "Date" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
            dayfirst=False
        )

        missing_dates = df["timestamp"].isna()

        if missing_dates.any():
            df.loc[missing_dates, "timestamp"] = pd.to_datetime(
                df.loc[missing_dates, "Date"],
                errors="coerce",
                dayfirst=True
            )
    else:
        df["timestamp"] = pd.NaT

    important_columns = [
        "Date",
        "timestamp",
        "connection_type",
        "latitude",
        "longitude",
        "download_mbps",
        "download_size",
        "upload_mbps",
        "upload_size",
        "latency_ms",
        "server",
        "source_file",
        "url"
    ]

    existing_columns = [col for col in important_columns if col in df.columns]
    df = df[existing_columns]

    numeric_columns = [
        "latitude",
        "longitude",
        "download_mbps",
        "download_size",
        "upload_mbps",
        "upload_size",
        "latency_ms"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[
        "latitude",
        "longitude",
        "download_mbps",
        "upload_mbps",
        "latency_ms"
    ])

    df["source"] = "real_speedtest"

    return df


def main():
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("Nu exista fisiere CSV in data/raw/speedtest")

    all_dfs = []

    for file_path in csv_files:
        df = clean_speedtest_file(file_path)
        all_dfs.append(df)
        print(f"Loaded {file_path.name}: {len(df)} rows")

    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df = final_df.sort_values(
        by=["timestamp", "source_file"],
        na_position="last"
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total Speedtest rows: {len(final_df)}")
    print(final_df.head())


if __name__ == "__main__":
    main()