"""Nodes for the data_ingestion pipeline.

Loads raw CSV files from flight directories, merges them into a single
time-aligned DataFrame via merge_asof, and filters noise/metadata columns.
"""
import glob
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Metadata keywords that identify non-informative columns
NOISE_KEYWORDS = [
    "covariance",
    "header.",
    "checksum",
    "magic",
    "seq",
    "stamp",
    "frame_id",
    "sysid",
    "compid",
    "msgid",
    "payload",
    "incompat_flags",
    "compat_flags",
    "framing_status",
    "time_ref",
]

# Topic-level sources to discard entirely
TOPIC_BLACKLIST = ["diagnostics", "mavlink-from"]

# Field-level terms to discard (applied across all topics)
FIELD_BLACKLIST = [
    # Battery metadata (static during flight)
    "design_capacity", "location", "present", "serial_number",
    "technology", "power_supply_health", "power_supply_technology",
    "power_supply_status", "capacity", "charge",
    # GPS absolute coordinates (use local position instead)
    "latitude", "longitude",
    # Atmospheric and thermal noise
    "fluid_pressure",
    # Setpoint fields redundant with nav_info
    "coordinate_frame", "type_mask", "acceleration_or_force",
    "setpoint_raw-local_field.yaw_rate",
    "setpoint_raw-local_field.yaw",
    "setpoint_raw-local_field.velocity.z",
    "setpoint_raw-local_field.position.x",
    "setpoint_raw-local_field.position.y",
    "setpoint_raw-local_field.position.z",
    # RC radio channels (manual override channels, not useful for autonomy)
    *[f"channels{i}" for i in range(16)],
    "rssi",
    # MAVLink state flags (binary, constant during autonomous flight)
    "armed_field", "connected_field", "guided_field", "system_status_field",
    # Angular velocity from wind/state topics (negligible signal)
    "twist.angular.x", "twist.angular.y", "twist.angular.z",
]


def merge_flight_csvs(flight_dir: str | Path) -> pd.DataFrame:
    """Load all CSV files from a single flight directory and merge into one DataFrame.

    Each CSV file corresponds to a ROS topic. Columns are prefixed with the
    topic name to avoid collisions across sources. Time is normalized to
    seconds from a common absolute origin across all topics.

    The merge strategy is merge_asof (backward fill): for each timestamp in
    the densest topic, the most recent value from every other topic is used.
    This preserves the temporal density of the primary source while filling
    in lower-frequency sensors without creating NaN gaps.

    Args:
        flight_dir: Path to the flight directory containing .csv files.

    Returns:
        Single merged DataFrame with '%time' column in seconds.
    """
    flight_dir = Path(flight_dir)
    all_files = sorted(glob.glob(str(flight_dir / "*.csv")))

    if not all_files:
        raise ValueError(f"No CSV files found in {flight_dir}")

    dfs = []
    for filename in all_files:
        topic_name = os.path.basename(filename).replace(".csv", "")
        temp_df = pd.read_csv(filename)

        # Keep only numeric columns (ROS topics mix types)
        cols = temp_df.select_dtypes(include=["number"]).columns.tolist()
        if "%time" not in cols:
            temp_df["%time"] = pd.to_numeric(temp_df["%time"], errors="coerce")
            cols.append("%time")

        temp_df = temp_df[cols].sort_values("%time")

        # Prefix each column with the topic name to avoid collisions
        new_names = {c: f"{topic_name}_{c}" for c in temp_df.columns if c != "%time"}
        temp_df.rename(columns=new_names, inplace=True)
        dfs.append(temp_df)

    # Normalize time: find absolute origin across all topics, convert to seconds
    min_time = min(df["%time"].min() for df in dfs)
    processed = []
    for df in dfs:
        tmp = df.copy()
        tmp["%time"] = (tmp["%time"] - min_time) / 1e9
        processed.append(tmp.sort_values("%time"))

    # Use the densest topic as the base for merge_asof
    processed.sort(key=len, reverse=True)
    merged = processed[0]

    for next_df in processed[1:]:
        new_cols = list(dict.fromkeys(
            next_df.columns.difference(merged.columns).tolist() + ["%time"]
        ))
        merged = pd.merge_asof(
            merged,
            next_df[new_cols],
            on="%time",
            direction="backward",
        )

    merged.fillna(0, inplace=True)
    log.info("Merged %d topics → shape %s | flight: %s", len(dfs), merged.shape, flight_dir.name)
    return merged


def filter_noise_columns(
    df: pd.DataFrame,
    imu_source_to_discard: str = "imu-data_raw",
) -> pd.DataFrame:
    """Remove metadata, covariance, and irrelevant columns from the merged DataFrame.

    Filtering criteria (applied in order):
    1. NOISE_KEYWORDS — ROS headers, covariance matrices, checksums
    2. TOPIC_BLACKLIST — diagnostics, mavlink-from
    3. imu_source_to_discard — raw IMU (prefer pre-processed IMU)
    4. FIELD_BLACKLIST — battery metadata, absolute GPS, RC channels, etc.

    Args:
        df: Raw merged DataFrame from merge_flight_csvs.
        imu_source_to_discard: Topic string for raw IMU to exclude.

    Returns:
        DataFrame with only informative columns, sorted alphabetically.
    """
    all_noise = NOISE_KEYWORDS + FIELD_BLACKLIST
    useful = ["%time"]

    for col in df.columns:
        if col == "%time":
            continue
        col_lower = col.lower()
        if any(kw in col_lower for kw in all_noise):
            continue
        if any(topic in col for topic in TOPIC_BLACKLIST):
            continue
        if imu_source_to_discard and imu_source_to_discard in col:
            continue
        useful.append(col)

    filtered = df[useful].copy()
    sorted_cols = ["%time"] + sorted(c for c in filtered.columns if c != "%time")
    log.info("Filtered columns: %d → %d", len(df.columns), len(sorted_cols))
    return filtered[sorted_cols]


def load_all_flights(
    raw_data_dir: str,
    imu_source_to_discard: str = "imu-data_raw",
) -> dict[str, pd.DataFrame]:
    """Load, merge, and filter all flights from the raw data directory.

    This is the single entry-point node for the data_ingestion pipeline.
    Iterates over subdirectories (one per flight), applies merge_flight_csvs
    and filter_noise_columns, and returns a dict keyed by flight name.

    The output is consumed by a PartitionedDataset in the catalog, which
    saves each flight as a separate CSV in data/02_intermediate/preprocessed/.

    Args:
        raw_data_dir: Path to directory where each subdirectory is one flight.
        imu_source_to_discard: Raw IMU topic to exclude from every flight.

    Returns:
        Dict mapping flight_name → filtered DataFrame (ready for data_preparation).
    """
    raw_path = Path(raw_data_dir)
    result: dict[str, pd.DataFrame] = {}

    flight_dirs = sorted(p for p in raw_path.iterdir() if p.is_dir())
    log.info("Found %d flight directories in '%s'", len(flight_dirs), raw_data_dir)

    for flight_dir in flight_dirs:
        flight_name = flight_dir.name
        log.info("Processing: %s", flight_name)
        df = merge_flight_csvs(flight_dir)
        df = filter_noise_columns(df, imu_source_to_discard)
        result[flight_name] = df

    log.info("data_ingestion complete: %d flights processed", len(result))
    return result
