"""Nodes for the fft_ingestion pipeline.

Loads raw CSV files from flight directories, merges them into a single
time-aligned DataFrame, filters noise/metadata columns, and then retains
only the signals whose spectral (FFT) characteristics are genuinely relevant
for motor anomaly detection.

Selection rationale
-------------------
The criterion is strict: a signal is included only if there is a direct
physical mechanism by which the motor creates a periodic component in that
signal. Derived features (energy_specific, glide_ratio, speed_horizontal,
energy_rate) and their ingredient signals (altitude, GPS velocity, vertical
velocity) are intentionally excluded — they are strong time-domain indicators
and useful as rolling statistics, but they do not carry spectral content
linked to the motor's rotation frequency. Keeping them here would conflate
"useful feature" with "useful FFT feature".

IMU linear acceleration:
    The motor and propeller induce mechanical vibrations in the airframe at
    the motor's rotation frequency and its harmonics. These appear as clear
    spectral peaks in the accelerometer data. Both peak magnitude and
    dominant frequency shift at failure or degradation.

Magnetometer:
    Motor windings and rotating propeller blades create a periodic EM field
    at the motor's rotation frequency. The amplitude and frequency of this
    EM signature change markedly at failure.

Airspeed (measured):
    Thrust ripple from the motor creates low-amplitude periodic oscillations
    in measured airspeed (0.5–5 Hz range). The oscillation character changes
    at partial or full motor failure.

Altitude:
    A time-domain indicator included as the only non-vibration feature.
    Altitude changes monotonically during the failure event (the aircraft
    loses thrust and begins to descend), providing a complementary signal
    that can help the model identify motor failure even when spectral
    changes are subtle.
"""
import logging
from pathlib import Path

import pandas as pd

from aeroespacial_2.pipelines.data_ingestion.nodes import (
    filter_noise_columns,
    merge_flight_csvs,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Substring patterns that identify FFT-relevant columns in the preprocessed
# (merged + noise-filtered) ROS dataset.  Each pattern is matched as a
# substring of the full column name, which still contains the ROS topic path.
# ---------------------------------------------------------------------------
FFT_RELEVANT_PATTERNS: list[str] = [
    # ── Structural vibration (primary FFT targets) ──────────────────────────
    "imu-data_field.linear_acceleration",  # IMU accel x/y/z — mechanical vibration
                                           #   at motor rotation frequency.
    # ── Magnetic field (EM signature of motor + propeller) ──────────────────
    "mag_field.magnetic",                  # Magnetometer x/y/z — periodic EM field
                                           #   at motor rotation frequency.
    # ── Airspeed (thrust ripple) ────────────────────────────────────────────
    "airspeed_field.measured",             # Airspeed oscillations from thrust ripple;
                                           #   frequency and amplitude shift at failure.
    # ── Altitude (única feature time-domain; complementa as features espectrais) ──
    "global_field.altitude",               # Altitude monotonicamente decrescente na falha;
                                           #   única feature não-vibração incluída.
    # ── Failure label ───────────────────────────────────────────────────────
    "failure_status",                      # Target: binary motor failure flag.
]


def select_fft_features(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Retain only columns whose name contains at least one FFT-relevant pattern.

    Always keeps '%time' (timestamp).  Every other column is kept only if its
    name contains at least one substring from *patterns*.

    Args:
        df: Preprocessed (merged + noise-filtered) flight DataFrame.
        patterns: Substring patterns to match against column names.
            Defaults to FFT_RELEVANT_PATTERNS when None.

    Returns:
        DataFrame with only FFT-relevant columns, preserving row order.
    """
    if patterns is None:
        patterns = FFT_RELEVANT_PATTERNS

    selected = ["%time"]
    for col in df.columns:
        if col == "%time":
            continue
        if any(pat in col for pat in patterns):
            selected.append(col)

    result = df[selected].copy()
    n_dropped = len(df.columns) - len(result.columns)
    log.info(
        "FFT feature selection: %d → %d columns kept | %d dropped",
        len(df.columns),
        len(result.columns),
        n_dropped,
    )
    return result


def load_fft_flights(
    raw_data_dir: str,
    imu_source_to_discard: str = "imu-data_raw",
    flight_keywords: list[str] | None = None,
    fft_relevant_patterns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load, merge, filter, and FFT-select all flights from the raw data directory.

    Entry-point node for the fft_ingestion pipeline.  Extends the
    data_ingestion pipeline with an additional step that retains only the
    columns whose spectral characteristics are relevant for motor anomaly
    detection.

    Args:
        raw_data_dir: Path to directory where each subdirectory is one flight.
        imu_source_to_discard: Raw IMU topic to exclude (prefer processed IMU).
        flight_keywords: If provided, only load flights whose directory name
            contains at least one keyword.  None → load all.
        fft_relevant_patterns: Substring patterns for column selection.
            Defaults to FFT_RELEVANT_PATTERNS when None.

    Returns:
        Dict mapping flight_name → FFT-selected DataFrame.
    """
    raw_path = Path(raw_data_dir)
    result: dict[str, pd.DataFrame] = {}

    all_dirs = sorted(p for p in raw_path.iterdir() if p.is_dir())
    if flight_keywords:
        flight_dirs = [p for p in all_dirs if any(kw in p.name for kw in flight_keywords)]
        log.info(
            "Filtered %d/%d flight directories using keywords %s",
            len(flight_dirs), len(all_dirs), flight_keywords,
        )
    else:
        flight_dirs = all_dirs
        log.info("Found %d flight directories in '%s'", len(flight_dirs), raw_data_dir)

    for flight_dir in flight_dirs:
        flight_name = flight_dir.name
        log.info("Processing: %s", flight_name)
        df = merge_flight_csvs(flight_dir)
        df = filter_noise_columns(df, imu_source_to_discard)
        df = select_fft_features(df, fft_relevant_patterns)
        result[flight_name] = df
        log.info("  → shape %s", df.shape)

    log.info("fft_ingestion complete: %d flights processed", len(result))
    return result
