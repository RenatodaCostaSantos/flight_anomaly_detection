"""Nodes for the fft_feature_engineering pipeline.

Engineers spectral and rolling features from the FFT-prepared dataset (output of
fft_data_preparation). Reuses compute_fft_features and compute_rolling_features
from the main feature_engineering pipeline — no duplicated implementation.

Signals available in fft_prepared_flights (after fft_data_preparation):
    imu_accel_x/y/z  — structural vibration at motor rotation frequency
    mag_x/y/z        — motor winding EM field + propeller rotation
    aspd_meas        — airspeed: thrust-ripple oscillations before full failure

Features engineered:
    Rolling statistics (mean, std, slope) on all 7 signals — captures temporal
    amplitude trends invisible in instantaneous values.

    FFT spectral features (peak_power, entropy, high_ratio) on all 7 signals —
    captures frequency-domain changes at motor rotation frequency.
"""
import logging

import pandas as pd

from aeroespacial_2.pipelines.feature_engineering.nodes import (
    compute_fft_features,
    compute_rolling_features,
)

log = logging.getLogger(__name__)

# All signals in the FFT prepared dataset.
# Used for both rolling statistics and spectral analysis — each provides
# complementary information: rolling captures amplitude trends, FFT captures
# spectral structure.
FFT_PIPELINE_SIGNALS: list[str] = [
    # Vibration signals — best for detecting motor/propeller oscillation changes
    "imu_accel_x",
    "imu_accel_y",
    "imu_accel_z",
    # Magnetometer — EM field from motor windings and propeller rotation
    "mag_x",
    "mag_y",
    "mag_z",
    # Airspeed — thrust-ripple proxy; rolling mean tracks thrust-level trend
    "aspd_meas",
]


def engineer_fft_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    fft_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Apply FFT feature engineering to a single prepared FFT flight.

    Computes two feature families from the FFT-prepared signal subset:
        1. Rolling statistics (mean, std, slope) on all available FFT signals.
           Captures amplitude and trend changes over configurable time windows.
        2. FFT spectral features (peak_power, entropy, high_ratio) on all
           available FFT signals. Captures frequency-domain structure changes
           induced by motor degradation or failure.

    Args:
        df: Prepared FFT flight DataFrame (output of fft_data_preparation).
            Expected columns: timestamp, target_fault, imu_accel_x/y/z,
            mag_x/y/z, aspd_meas.
        rolling_windows: Window sizes in samples for rolling statistics.
        fft_windows: Window sizes in samples for FFT spectral features.
            Defaults to rolling_windows when None.

    Returns:
        DataFrame with all engineered features appended. Column naming:
            {signal}_mean_{w}, {signal}_std_{w}, {signal}_slope_{w}
            fft_peak_power_{signal}_{w}, fft_entropy_{signal}_{w},
            fft_high_ratio_{signal}_{w}
    """
    fft_wins = fft_windows if fft_windows is not None else rolling_windows

    signals = [f for f in FFT_PIPELINE_SIGNALS if f in df.columns]

    df = compute_rolling_features(df, rolling_windows, signals)
    df = compute_fft_features(df, fft_wins, signals)

    log.info(
        "engineer_fft_features: shape %s | rolling windows %s | fft windows %s",
        df.shape,
        rolling_windows,
        fft_wins,
    )
    return df


def engineer_fft_features_for_all_flights(
    fft_prepared_flights: dict,
    rolling_windows: list[int],
    fft_windows: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply FFT feature engineering to every prepared FFT flight.

    Handles both direct DataFrames and lazy callables returned by Kedro's
    PartitionedDataset.

    Args:
        fft_prepared_flights: Dict mapping flight_name → DataFrame or loader callable.
            Produced by the fft_data_preparation pipeline.
        rolling_windows: Window sizes in samples for rolling statistics.
        fft_windows: Window sizes in samples for FFT spectral features.
            Defaults to rolling_windows when None.

    Returns:
        Dict mapping flight_name → feature-engineered DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for flight_name, loader in fft_prepared_flights.items():
        df = loader() if callable(loader) else loader
        result[flight_name] = engineer_fft_features(df, rolling_windows, fft_windows)
        log.info("Engineered FFT: %s → shape %s", flight_name, result[flight_name].shape)

    log.info("fft_feature_engineering complete: %d flights processed", len(result))
    return result
