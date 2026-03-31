"""Nodes for the feature_engineering pipeline.

Engineers physics-informed features for motor failure detection:
  - Specific mechanical energy: h + v²/(2g), the primary leading indicator
  - Energy rate of change: captures the energy drain at motor failure
  - Velocity magnitudes: horizontal and total speed
  - Glide ratio: converges to the aerodynamic L/D ratio when the motor fails
  - Control effort: composite tracking-error metric
  - Rolling statistics: temporal trends over configurable windows
  - FFT spectral features: frequency-domain descriptors over rolling windows
"""
import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

log = logging.getLogger(__name__)

G: float = 9.81  # gravitational acceleration [m/s²]

PROTECTED_COLS: frozenset[str] = frozenset({"timestamp", "target_fault"})

# Features for which rolling statistics (mean, std, slope) are computed.
# Order matters: compute_specific_energy and compute_control_effort must run first.
ROLLING_TARGET_FEATURES: list[str] = [
    "energy_specific",
    "alt_global",
    "aspd_meas",
    "control_effort",
    "err_vel_z",
    "glide_ratio",
    "hud_throttle",  # autopilot throttle: stays high while motor is off — strong step signal
]

# Signals analysed in the frequency domain via rolling FFT windows.
# Prioritised by physical proximity to motor state:
#   batt_current  — direct motor power: drops to zero at failure (most direct)
#   imu_accel_*   — structural vibration from motor/propeller (needs large windows)
#   mag_*         — magnetometer detects motor winding field and propeller rotation
#   aspd_meas     — airspeed oscillations from thrust ripple before full failure
#   energy_specific / alt_global — slower, flight-dynamics signals (need large windows)
FFT_TARGET_FEATURES: list[str] = [
    # Vibration signals — best for detecting motor/propeller oscillations
    "imu_accel_x",
    "imu_accel_y",
    "imu_accel_z",
    # Magnetometer — detects motor winding field and propeller rotation changes
    "mag_x",
    "mag_y",
    "mag_z",
    # Flight-dynamics signals — useful with large windows (500–2000 samples)
    "aspd_meas",
    "energy_specific",
    "alt_global",
]


# ---------------------------------------------------------------------------
# Individual feature transformations
# ---------------------------------------------------------------------------


def compute_specific_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Add specific mechanical energy: h + v²/(2g)  [meters].

    In powered flight the motor continuously compensates for aerodynamic drag,
    keeping total specific energy roughly constant. At motor failure, drag
    dissipates energy monotonically and no thrust replaces it.

    Crucially, the energy drop begins *immediately* at failure while the
    altitude drop is delayed — the aircraft may initially trade airspeed for
    altitude (or vice versa). This makes energy_specific a *leading*
    indicator, whereas altitude alone is a *lagging* indicator.

    Uses aspd_meas when available; falls back to groundspeed_hud for flights
    that do not record the nav_info airspeed field.

    New column:
        energy_specific  [m]
    """
    df = df.copy()
    if "aspd_meas" in df.columns:
        airspeed = df["aspd_meas"]
    elif "groundspeed_hud" in df.columns:
        log.warning("aspd_meas not found — falling back to groundspeed_hud for energy_specific")
        airspeed = df["groundspeed_hud"]
    else:
        raise KeyError("Neither 'aspd_meas' nor 'groundspeed_hud' found in DataFrame")
    df["energy_specific"] = df["alt_global"] + airspeed ** 2 / (2 * G)
    return df


def compute_energy_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add the time-derivative of specific energy  [m/s].

    Computed as a finite difference normalised by the elapsed time between
    consecutive samples — robust to non-uniform sampling.

    Sign convention:
        > 0  →  gaining energy (engine thrust exceeds drag, or altitude gain)
        ≈ 0  →  steady powered cruise
        < 0  →  losing energy (drag > thrust, or engine off)

    New column:
        energy_rate  [m/s]
    """
    df = df.copy()
    dt = df["timestamp"].diff().clip(lower=1e-6)
    df["energy_rate"] = df["energy_specific"].diff() / dt
    df["energy_rate"] = df["energy_rate"].fillna(0)
    return df


def compute_velocity_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add horizontal and total speed from GPS velocity components  [m/s].

    These are used downstream to compute glide_ratio and enrich the feature
    set with speed information independent of individual axis components.

    New columns:
        speed_horizontal  [m/s]  = sqrt(vel_x_gps² + vel_y_gps²)
        speed_total       [m/s]  = sqrt(vel_x_gps² + vel_y_gps² + vel_z_meas²)
    """
    df = df.copy()
    df["speed_horizontal"] = np.sqrt(df["vel_x_gps"] ** 2 + df["vel_y_gps"] ** 2)
    df["speed_total"] = np.sqrt(
        df["vel_x_gps"] ** 2 + df["vel_y_gps"] ** 2 + df["vel_z_meas"] ** 2
    )
    return df


def compute_glide_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add the aerodynamic glide ratio: horizontal speed / sink rate.

    In powered cruise this varies erratically. When the motor fails and the
    aircraft settles into an unpowered glide, the ratio converges to the
    airframe's aerodynamic L/D (~8 for the CarbonZ). The transition from
    erratic to stable is itself a failure signature.

    Sink rate is clipped to 0.1 m/s to avoid division-by-zero during level
    flight; extreme ratios are capped at 50 to limit outliers.

    Requires: speed_horizontal (from compute_velocity_magnitudes).

    New column:
        glide_ratio  [dimensionless]
    """
    df = df.copy()
    sink_rate = (-df["vel_z_meas"]).clip(lower=0.1)
    df["glide_ratio"] = (df["speed_horizontal"] / sink_rate).clip(upper=50.0)
    return df


def compute_control_effort(df: pd.DataFrame) -> pd.DataFrame:
    """Add a composite control tracking-error metric.

    When the motor fails the autopilot keeps issuing normal commands while
    the aircraft diverges from the set-point — all error signals grow
    simultaneously. Squaring and summing them into a single scalar amplifies
    the failure signature while remaining interpretable.

    New column:
        control_effort  = err_pitch² + err_roll² + err_vel_z²
    """
    df = df.copy()
    df["control_effort"] = (
        df["err_pitch"] ** 2 + df["err_roll"] ** 2 + df["err_vel_z"] ** 2
    )
    return df


def compute_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    features: list[str],
) -> pd.DataFrame:
    """Add rolling mean, standard deviation, and slope for selected features.

    Rolling statistics convert point-in-time measurements into temporal
    summaries that capture gradual degradation patterns invisible in a single
    timestep:
        mean:  noise-smoothed signal level over the window
        std:   local variability (typically spikes at fault onset)
        slope: average rate of change over the window [units/s]

    The slope is computed as the total change over the window divided by the
    window's elapsed time, giving a physically meaningful rate in units-per-second
    regardless of the sampling rate.

    Args:
        df: DataFrame with all base features already present.
        windows: Rolling window sizes in samples (e.g. [50, 100, 200]).
        features: Feature column names to process.

    Returns:
        DataFrame with 3 × len(windows) × len(features) new columns appended.
        Column naming convention: {feature}_{stat}_{window}
        Example: energy_specific_slope_100
    """
    df = df.copy()
    dt_median = df["timestamp"].diff().median()

    for feat in features:
        if feat not in df.columns:
            log.warning("Rolling feature '%s' not found in DataFrame — skipping.", feat)
            continue
        for w in windows:
            roll = df[feat].rolling(w, min_periods=1)
            df[f"{feat}_mean_{w}"] = roll.mean()
            df[f"{feat}_std_{w}"] = roll.std().fillna(0)
            # slope = Δx / Δt  →  change per second over the window
            window_duration = w * dt_median if dt_median > 1e-9 else float(w)
            df[f"{feat}_slope_{w}"] = (df[feat].diff(w) / window_duration).fillna(0)

    return df


def compute_fft_features(
    df: pd.DataFrame,
    windows: list[int],
    features: list[str],
) -> pd.DataFrame:
    """Add FFT-based spectral features over rolling windows.

    For each (feature, window) pair, three descriptors are extracted from the
    magnitude spectrum of a sliding window:

        fft_peak_power_{feat}_{w}:
            Magnitude of the dominant non-DC frequency component.
            A failing motor introduces periodic oscillations — thrust ripple,
            propeller imbalance — that concentrate spectral energy at specific
            frequencies and raise this value even before the energy envelope drops.

        fft_entropy_{feat}_{w}:
            Shannon entropy of the normalised power spectrum, scaled to [0, 1].
            Low entropy → energy concentrated at a few frequencies (structured
            oscillation). High entropy → energy spread across all bins (noise /
            chaos). Engine degradation tends to shift a signal from structured
            (powered cruise) to chaotic (partial motor failure).

        fft_high_ratio_{feat}_{w}:
            Fraction of spectral power in the upper half of the frequency band.
            Normal manoeuvres are dominated by low-frequency components (slow
            attitude changes). Motor-induced vibration and turbulence inject
            high-frequency energy, raising this ratio before the fault is
            detectable in the time domain.

    Implementation uses vectorised batch FFT via sliding_window_view, which
    avoids a Python loop over rows and is substantially faster than
    rolling().apply() for large DataFrames.

    Args:
        df: DataFrame with feature columns already present.
        windows: Rolling window sizes in samples (e.g. [50, 100, 200]).
        features: Columns to analyse in the frequency domain.

    Returns:
        DataFrame with 3 × len(windows) × len(features) new columns appended.
        Naming: fft_peak_power_{feat}_{w}, fft_entropy_{feat}_{w},
                fft_high_ratio_{feat}_{w}.
    """
    df = df.copy()
    n = len(df)

    for feat in features:
        if feat not in df.columns:
            log.warning("FFT feature '%s' not found in DataFrame — skipping.", feat)
            continue

        signal = pd.Series(df[feat].values.astype(float)).ffill().fillna(0).values

        for w in windows:
            peak_power = np.zeros(n)
            entropy = np.zeros(n)
            high_ratio = np.zeros(n)

            if n >= w:
                # Batch FFT over all windows at once: shape (n-w+1, w//2+1)
                wins = sliding_window_view(signal, window_shape=w)
                mags = np.abs(np.fft.rfft(wins, axis=1))

                # Exclude DC component (index 0)
                ac = mags[:, 1:]                        # (n-w+1, w//2)
                power = ac ** 2
                total = power.sum(axis=1, keepdims=True).clip(min=1e-12)

                # Peak magnitude of dominant AC frequency
                peak = ac.max(axis=1)

                # Normalised Shannon entropy of power spectrum
                p_norm = power / total
                n_bins = ac.shape[1]
                ent = -np.sum(p_norm * np.log(p_norm + 1e-12), axis=1) / np.log(n_bins + 1)

                # High-frequency band power ratio (upper half / total)
                mid = n_bins // 2
                ratio = (power[:, mid:].sum(axis=1) / total[:, 0]).clip(0.0, 1.0)

                # Align output: pad leading rows with the first valid value
                offset = w - 1
                peak_power[offset:] = peak
                entropy[offset:] = ent
                high_ratio[offset:] = ratio
                peak_power[:offset] = peak[0]
                entropy[:offset] = ent[0]
                high_ratio[:offset] = ratio[0]

            df[f"fft_peak_power_{feat}_{w}"] = peak_power
            df[f"fft_entropy_{feat}_{w}"] = entropy
            df[f"fft_high_ratio_{feat}_{w}"] = high_ratio

    return df


# ---------------------------------------------------------------------------
# Orchestration nodes
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    fft_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Apply the complete feature engineering pipeline to a single flight.

    Transformation order is fixed — downstream steps depend on upstream outputs:
        1. compute_specific_energy     — requires alt_global, aspd_meas
        2. compute_energy_rate         — requires energy_specific, timestamp
        3. compute_velocity_magnitudes — requires vel_x_gps, vel_y_gps, vel_z_meas
        4. compute_glide_ratio         — requires speed_horizontal, vel_z_meas
        5. compute_control_effort      — requires err_pitch, err_roll, err_vel_z
        6. compute_rolling_features    — operates on features created above
        7. compute_fft_features        — frequency-domain descriptors (if fft_windows given)

    Args:
        df: Fully prepared flight DataFrame (output of data_preparation).
        rolling_windows: Window sizes in samples for rolling statistics.
        fft_windows: Window sizes in samples for FFT spectral features.
            Defaults to rolling_windows when None.

    Returns:
        DataFrame with all engineered features appended.
    """
    df = compute_specific_energy(df)
    df = compute_energy_rate(df)
    df = compute_velocity_magnitudes(df)
    df = compute_glide_ratio(df)
    df = compute_control_effort(df)

    rolling_feats = [f for f in ROLLING_TARGET_FEATURES if f in df.columns]
    df = compute_rolling_features(df, rolling_windows, rolling_feats)

    fft_wins = fft_windows if fft_windows is not None else rolling_windows
    fft_feats = [f for f in FFT_TARGET_FEATURES if f in df.columns]
    df = compute_fft_features(df, fft_wins, fft_feats)

    log.info(
        "engineer_features: shape %s | rolling windows %s | fft windows %s",
        df.shape,
        rolling_windows,
        fft_wins,
    )
    return df


def engineer_features_for_all_flights(
    prepared_flights: dict,
    rolling_windows: list[int],
    fft_windows: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply feature engineering to every prepared flight.

    Handles both direct DataFrames and lazy callables returned by Kedro's
    PartitionedDataset.

    Args:
        prepared_flights: Dict mapping flight_name → DataFrame or loader callable.
        rolling_windows: Window sizes in samples for rolling statistics.
        fft_windows: Window sizes in samples for FFT spectral features.
            Defaults to rolling_windows when None.

    Returns:
        Dict mapping flight_name → feature-engineered DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for flight_name, loader in prepared_flights.items():
        df = loader() if callable(loader) else loader
        result[flight_name] = engineer_features(df, rolling_windows, fft_windows)
        log.info("Engineered: %s → shape %s", flight_name, result[flight_name].shape)

    log.info("feature_engineering complete: %d flights processed", len(result))
    return result
