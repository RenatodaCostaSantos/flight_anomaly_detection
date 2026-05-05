"""Nodes for the fft_data_preparation pipeline.

Applies the same preparation steps as data_preparation, but skips
create_error_features — the FFT dataset does not include the attitude
command/measured columns (pitch_cmd, roll_cmd, etc.) required by that step.

Steps applied:
    1. rename_columns             — ROS names → human-readable names
    2. remove_redundant_columns   — drop constant/redundant columns (no-op for FFT subset)
    3. cut_initial_seconds        — remove sensor spin-up artifacts
    4. rename_final_columns       — final standardized naming
    5. filter_low_variance_columns — drop columns with std ≤ min_std_threshold
"""
import logging

import pandas as pd

from aeroespacial_2.pipelines.data_preparation.nodes import (
    cut_initial_seconds,
    detrend_magnetometer,
    filter_low_variance_columns,
    remove_redundant_columns,
    rename_columns,
    rename_final_columns,
)

log = logging.getLogger(__name__)


def prepare_fft_flight(
    df: pd.DataFrame,
    flight_name: str,
    cut_seconds: float = 1.0,
    min_std_threshold: float = 0.0,
    detrend_seconds: float = 2.0,
) -> pd.DataFrame:
    """Prepare a single FFT-selected flight DataFrame.

    Mirrors prepare_flight() from data_preparation, omitting create_error_features
    because the FFT subset does not contain attitude command/measured columns.

    Steps applied:
    1. rename_columns             — ROS names → human-readable names
    2. remove_redundant_columns   — drop constant/redundant columns
    3. cut_initial_seconds        — remove sensor spin-up artifacts
    4. rename_final_columns       — final standardized naming
    5. detrend_magnetometer       — remove per-flight calibration offset from mag_x/y/z
    6. filter_low_variance_columns — drop columns with std ≤ min_std_threshold

    Args:
        df: FFT-selected DataFrame from fft_ingestion.
        flight_name: Flight identifier (used to strip ROS column prefixes).
        cut_seconds: Seconds to remove from the start of the flight.
        min_std_threshold: Columns with std ≤ this value are dropped.
        detrend_seconds: Seconds of stable baseline used to estimate and
            remove the magnetometer calibration offset. Set to 0.0 to skip.

    Returns:
        Prepared DataFrame ready for FFT feature engineering.
    """
    df = rename_columns(df, flight_name)
    df = remove_redundant_columns(df)
    df = cut_initial_seconds(df, cut_seconds)
    df = rename_final_columns(df)
    if "target_fault" not in df.columns:
        df["target_fault"] = 0
    df = detrend_magnetometer(df, detrend_seconds)
    df = filter_low_variance_columns(df, min_std_threshold)
    log.info("Prepared FFT flight '%s' → shape %s", flight_name, df.shape)
    return df


def prepare_all_fft_flights(
    fft_ready_flights: dict,
    cut_seconds: float = 1.0,
    min_std_threshold: float = 0.0,
    detrend_seconds: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Apply the FFT preparation pipeline to all fft_ready flights.

    Args:
        fft_ready_flights: Dict from fft_ingestion (flight_name → df or loader callable).
        cut_seconds: Seconds to cut from the start of each flight.
        min_std_threshold: Columns with std ≤ this value are dropped per flight.
        detrend_seconds: Seconds of stable baseline used to estimate and
            remove the per-flight magnetometer calibration offset.

    Returns:
        Dict mapping flight names → prepared DataFrames.
    """
    result = {}
    for flight_name, loader in fft_ready_flights.items():
        df = loader() if callable(loader) else loader
        result[flight_name] = prepare_fft_flight(
            df, flight_name, cut_seconds, min_std_threshold, detrend_seconds
        )

    log.info("fft_data_preparation complete: %d flights prepared", len(result))
    return result
