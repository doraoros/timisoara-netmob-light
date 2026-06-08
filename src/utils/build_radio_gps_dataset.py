import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RAW_GPS,
    DATA_RAW_NOTES,
    DATASET_RADIO_GPS,
    HO_HORIZON_S,
    SESSIONS_CSV,
    SPEEDTEST_ALL,
)
from src.features.handoff_labels import label_handoff_events, label_handoff_next
from src.parsing.parse_notes import load_json_cells
from src.utils.merge_speedtest import attach_speedtest, load_speedtest_table


def expand_radio_to_gps(radio_df, gps_df):
    """
    NetMonster: puține celule per fișier; GPS: serie 1 Hz.
    Distribuim celulele uniform pe durata traseului.
    """
    if radio_df.empty or gps_df.empty:
        return pd.DataFrame()

    total_gps_rows = len(gps_df)
    total_radio_rows = len(radio_df)
    repeated_rows = []
    segment_size = max(30, total_gps_rows // max(total_radio_rows, 1))
    current_radio_index = 0

    for i in range(total_gps_rows):
        if i > 0 and i % segment_size == 0:
            current_radio_index = min(current_radio_index + 1, total_radio_rows - 1)
        row = radio_df.iloc[current_radio_index].to_dict()
        row["radio_cell_index"] = current_radio_index
        repeated_rows.append(row)

    return pd.DataFrame(repeated_rows)


def _label_session(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp").reset_index(drop=True)
    group["handoff"] = label_handoff_events(group)
    group["handoff_next"] = label_handoff_next(group, horizon_s=HO_HORIZON_S)
    return group


def main():
    if not SPEEDTEST_ALL.exists():
        from src.utils.combine_speedtest_files import main as combine_st

        try:
            combine_st()
        except FileNotFoundError:
            pass

    sessions_df = pd.read_csv(SESSIONS_CSV)
    all_sessions = []

    for _, session in sessions_df.iterrows():
        session_id = session["session_id"]
        json_file = session["json_file"]
        gps_file = session["gps_file"]

        json_path = DATA_RAW_NOTES / json_file
        gps_path = DATA_RAW_GPS / gps_file

        if not json_path.exists():
            print(f"Missing JSON file: {json_path}")
            continue
        if not gps_path.exists():
            print(f"Missing GPS file: {gps_path}")
            continue

        radio_df = load_json_cells(json_path)
        gps_df = pd.read_csv(gps_path)
        radio_expanded = expand_radio_to_gps(radio_df, gps_df)

        combined_df = pd.concat(
            [gps_df.reset_index(drop=True), radio_expanded.reset_index(drop=True)],
            axis=1,
        )
        combined_df["session_id"] = session_id
        combined_df["json_file"] = json_file
        combined_df["gps_file"] = gps_file
        all_sessions.append(combined_df)

        print(f"Processed {json_file} + {gps_file} -> {len(combined_df)} rows")

    if not all_sessions:
        raise RuntimeError("Nicio sesiune procesată. Verifică sessions.csv și fișierele din data/raw/.")

    final_df = pd.concat(all_sessions, ignore_index=True)
    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
    final_df["hour"] = final_df["timestamp"].dt.hour
    final_df["day"] = final_df["timestamp"].dt.date

    labeled_parts = []
    for _, group in final_df.groupby("session_id", sort=False):
        labeled_parts.append(_label_session(group))
    final_df = pd.concat(labeled_parts, ignore_index=True)

    speedtest = load_speedtest_table()
    final_df = attach_speedtest(final_df, speedtest)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(DATASET_RADIO_GPS, index=False)

    print()
    print(f"Final dataset saved to: {DATASET_RADIO_GPS}")
    print(f"Total rows: {len(final_df)}")
    print(f"handoff rate: {final_df['handoff'].mean():.4f}")
    print(f"handoff_next rate ({HO_HORIZON_S}s): {final_df['handoff_next'].mean():.4f}")
    print(f"speedtest matched: {final_df['speedtest_matched'].mean():.4f}")
    print(final_df.head())


if __name__ == "__main__":
    main()
