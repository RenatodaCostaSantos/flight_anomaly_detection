"""Nodes for the data_preparation pipeline.

Renames ROS topic columns to human-readable names, removes constant/redundant
columns identified during EDA, cuts sensor initialization artifacts from the
start of each flight, and engineers tracking-error features.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column rename mappings
# ---------------------------------------------------------------------------

# Explicit mapping: partial column name (without flight prefix) → clean name.
# Applied first via rename_columns(); everything else goes through clean_col().
SPECIFIC_RENAME: dict[str, str] = {
    "%time": "timestamp",
    "failure_status-engines_field.data": "fault_status",
    # MAVCTRL — path deviation and attitude commands
    "mavctrl-path_dev_field.x": "path_dev_x",
    "mavctrl-path_dev_field.y": "path_dev_y",
    "mavctrl-path_dev_field.z": "path_dev_z",
    "mavctrl-rpy_field.x": "ctrl_roll",
    "mavctrl-rpy_field.y": "ctrl_pitch",
    "mavctrl-rpy_field.z": "ctrl_yaw",
    # Battery
    "mavros-battery_field.current": "batt_current",
    "mavros-battery_field.percentage": "batt_pct",
    "mavros-battery_field.voltage": "batt_voltage",
    # IMU (processed)
    "mavros-imu-data_field.angular_velocity.x": "imu_gyro_x",
    "mavros-imu-data_field.angular_velocity.y": "imu_gyro_y",
    "mavros-imu-data_field.angular_velocity.z": "imu_gyro_z",
    "mavros-imu-data_field.linear_acceleration.x": "imu_accel_x",
    "mavros-imu-data_field.linear_acceleration.y": "imu_accel_y",
    "mavros-imu-data_field.linear_acceleration.z": "imu_accel_z",
    # VFR HUD (consolidated flight instruments)
    "mavros-vfr_hud_field.airspeed": "hud_airspeed",
    "mavros-vfr_hud_field.altitude": "hud_altitude",
    "mavros-vfr_hud_field.climb": "hud_climb_rate",
    "mavros-vfr_hud_field.throttle": "hud_throttle",
    # Navigation errors
    "mavros-nav_info-errors_field.aspd_error": "err_airspeed",
    "mavros-nav_info-errors_field.alt_error": "err_altitude",
    "mavros-nav_info-errors_field.xtrack_error": "err_xtrack",
    # Nav info: commanded vs measured attitude
    "mavros-nav_info-pitch_field.commanded": "pitch_cmd",
    "mavros-nav_info-pitch_field.measured": "pitch_meas",
    "mavros-nav_info-roll_field.commanded": "roll_cmd",
    "mavros-nav_info-roll_field.measured": "roll_meas",
    # Wind estimation
    "mavros-wind_estimation_field.twist.linear.x": "wind_x",
    "mavros-wind_estimation_field.twist.linear.y": "wind_y",
    "mavros-wind_estimation_field.twist.linear.z": "wind_z",
}

# Second-pass rename: intermediate names (after clean_col) → final standardized names
FINAL_RENAME: dict[str, str] = {
    "fault_status": "target_fault",
    "path_dev_y": "dev_path_y",
    "path_dev_z": "dev_path_z",
    "ctrl_yaw": "ctrl_yaw_rate",
    "compass_hdg": "nav_heading",
    "global.altitude": "alt_global",
    "fix.altitude": "alt_gps_fix",
    "errors.wp_dist": "nav_dist_wp",
    "err_xtrack": "nav_error_xtrack",
    "local.twist.twist.linear.z": "vel_z_local",
    "gps_vel.twist.linear.x": "vel_x_gps",
    "gps_vel.twist.linear.y": "vel_y_gps",
    "odom.twist.twist.linear.x": "vel_x_odom",
    "odom.twist.twist.linear.y": "vel_y_odom",
    "odom.twist.twist.linear.z": "vel_z_odom",
    "velocity.twist.linear.x": "vel_x_twist",
    "velocity.twist.linear.y": "vel_y_twist",
    "velocity.twist.linear.z": "vel_z_twist",
    "position.x": "pos_x_local",
    "position.y": "pos_y_local",
    "position.z": "pos_z_local",
    "airspeed.commanded": "aspd_cmd",
    "airspeed.measured": "aspd_meas",
    "err_airspeed": "aspd_error",
    "hud_airspeed": "aspd_hud",
    "err_altitude": "alt_error",
    "yaw.commanded": "yaw_cmd",
    "yaw.measured": "yaw_meas",
    "target_global.yaw_rate": "yaw_rate_target",
    "velocity.des_x": "vel_x_des",
    "velocity.des_y": "vel_y_des",
    "velocity.des_z": "vel_z_des",
    "velocity.meas_x": "vel_x_meas",
    "velocity.meas_y": "vel_y_meas",
    "velocity.meas_z": "vel_z_meas",
    "hud_altitude": "alt_hud",
    "vfr_hud.groundspeed": "groundspeed_hud",
    "mag.magnetic.x": "mag_x",
    "mag.magnetic.y": "mag_y",
    "mag.magnetic.z": "mag_z",
}

# ---------------------------------------------------------------------------
# Columns to drop after rename_columns() — determined during EDA
# ---------------------------------------------------------------------------

COLUMNS_TO_DROP: list[str] = [
    # Constant during flight (battery readings, global status flags)
    "batt_current", "batt_pct", "battery.power_supply_status", "batt_voltage",
    "global.status.status",
    "local.twist.twist.angular.x", "local.twist.twist.angular.y", "local.twist.twist.angular.z",
    "target_global.yaw", "path_dev_x",
    # Step-change / discontinuous signals (IMU orientation quaternions, angular rates)
    "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
    "imu_accel_x", "imu_accel_y", "imu_accel_z",
    "data.orientation.w", "data.orientation.x", "data.orientation.y", "data.orientation.z",
    "odom.orientation.w", "odom.orientation.x", "odom.orientation.y", "odom.orientation.z",
    "odom.twist.twist.angular.x", "odom.twist.twist.angular.y", "odom.twist.twist.angular.z",
    "orientation.w", "orientation.x", "orientation.y", "orientation.z",
    "velocity.twist.angular.x", "velocity.twist.angular.y", "velocity.twist.angular.z",
    "hud_climb_rate", "vfr_hud.heading", "hud_throttle", "ctrl_roll", "ctrl_pitch",
    "local.orientation.w", "local.orientation.x", "local.orientation.y", "local.orientation.z",
    # Redundant: duplicated position/velocity across odometry sources
    "target_global.velocity.x", "target_global.velocity.y",
    "local.velocity.x", "local.velocity.y",
    "odom.position.x", "odom.position.y", "odom.position.z",
    "local.position.x", "local.position.y", "local.position.z",
    "local.twist.twist.linear.x", "local.twist.twist.linear.y",
    "rel_alt",
]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def rename_columns(df: pd.DataFrame, flight_name: str) -> pd.DataFrame:
    """Rename columns from raw ROS topic names to clean, human-readable names.

    Two-step process:
    1. Explicit mapping via SPECIFIC_RENAME (with flight_name prefix applied).
    2. Automated cleaning via clean_col() for everything not in the mapping:
       strip the flight prefix, remove ROS field suffixes (_field, .data),
       and take the last dash-separated segment.

    Args:
        df: Merged DataFrame with columns prefixed by flight_name.
        flight_name: Flight identifier, e.g. 'carbonZ_2018-07-18-15-53-31_1_engine_failure'.

    Returns:
        DataFrame with intermediate human-readable column names.
    """
    prefix = flight_name + "-"
    full_rename = {f"{prefix}{k}": v for k, v in SPECIFIC_RENAME.items()}
    full_rename["%time"] = "timestamp"

    def clean_col(col: str) -> str:
        if col in full_rename:
            return full_rename[col]
        if col.startswith(prefix):
            col = col[len(prefix):]
        col = col.replace("_field", "").replace(".data", "").replace("pose.pose.", "")
        return col.split("-")[-1]

    result = df.copy()
    result.columns = [clean_col(c) for c in result.columns]
    return result


def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant, step-change, and redundant columns identified during EDA.

    These columns carry no useful anomaly signal:
    - Constant: battery readings that don't change during flight
    - Step-change: IMU orientation quaternions (discontinuous due to gimbal lock)
    - Redundant: position/velocity duplicated across odometry sources

    Args:
        df: DataFrame after rename_columns().

    Returns:
        Reduced DataFrame.
    """
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=to_drop)


def cut_initial_seconds(df: pd.DataFrame, seconds: float = 1.0) -> pd.DataFrame:
    """Remove the first N seconds of flight data and reset timestamp to zero.

    The first second contains sensor initialization artifacts: zero-valued
    readings, missing GPS lock, and unstable IMU estimates. Removing it
    produces cleaner baselines for anomaly detection.

    Args:
        df: DataFrame with 'timestamp' column in seconds.
        seconds: Duration to cut from the beginning.

    Returns:
        Trimmed DataFrame with timestamp reset to start at 0.
    """
    df = df[df["timestamp"] >= seconds].copy()
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df.reset_index(drop=True, inplace=True)
    return df


def create_error_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer tracking-error features as (commanded − measured).

    These are the most informative signals for detecting motor failures:
    when the motor stops, the flight controller keeps commanding normal
    values while measured values diverge — creating large error spikes.

    New columns:
        err_pitch: pitch_cmd − pitch_meas
        err_roll:  roll_cmd − roll_meas
        err_vel_x/y/z: velocity.des_* − velocity.meas_*
        err_yaw:   yaw.commanded − yaw.measured

    Note: uses intermediate column names (after rename_columns, before rename_final_columns).

    Args:
        df: DataFrame with intermediate column names.

    Returns:
        DataFrame with 6 additional error columns appended.
    """
    df = df.copy()
    df["err_pitch"] = df["pitch_cmd"] - df["pitch_meas"]
    df["err_roll"] = df["roll_cmd"] - df["roll_meas"]
    df["err_vel_x"] = df["velocity.des_x"] - df["velocity.meas_x"]
    df["err_vel_y"] = df["velocity.des_y"] - df["velocity.meas_y"]
    df["err_vel_z"] = df["velocity.des_z"] - df["velocity.meas_z"]
    df["err_yaw"] = df["yaw.commanded"] - df["yaw.measured"]
    return df


def rename_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply FINAL_RENAME to standardize all remaining column names."""
    return df.rename(columns={k: v for k, v in FINAL_RENAME.items() if k in df.columns})


def prepare_flight(
    df: pd.DataFrame,
    flight_name: str,
    cut_seconds: float = 1.0,
) -> pd.DataFrame:
    """Complete preparation pipeline for a single preprocessed flight.

    Applies all preparation steps in the correct order:
    1. rename_columns        — ROS names → intermediate human-readable names
    2. remove_redundant_columns — drop constant/step/duplicate columns
    3. cut_initial_seconds   — remove sensor spin-up artifacts
    4. create_error_features — add (commanded − measured) error signals
    5. rename_final_columns  — final standardized naming convention

    Args:
        df: Preprocessed DataFrame from data_ingestion.
        flight_name: Flight identifier (used to strip ROS column prefixes).
        cut_seconds: Seconds to remove from the start of flight.

    Returns:
        Fully prepared DataFrame ready for model training.
    """
    df = rename_columns(df, flight_name)
    df = remove_redundant_columns(df)
    df = cut_initial_seconds(df, cut_seconds)
    df = create_error_features(df)
    df = rename_final_columns(df)
    log.info("Prepared flight '%s' → shape %s", flight_name, df.shape)
    return df


def prepare_all_flights(
    preprocessed_flights: dict,
    cut_seconds: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Apply the full preparation pipeline to all preprocessed flights.

    Handles both direct DataFrames and lazy callables (PartitionedDataset
    returns callables when used as node input in Kedro).

    Args:
        preprocessed_flights: Dict from data_ingestion (flight_name → df or loader callable).
        cut_seconds: Seconds to cut from the start of each flight.

    Returns:
        Dict mapping flight names → fully prepared DataFrames.
    """
    result = {}
    for flight_name, loader in preprocessed_flights.items():
        df = loader() if callable(loader) else loader
        result[flight_name] = prepare_flight(df, flight_name, cut_seconds)

    log.info("data_preparation complete: %d flights prepared", len(result))
    return result
