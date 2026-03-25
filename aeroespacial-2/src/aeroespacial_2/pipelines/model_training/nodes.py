"""Nodes for the model_training pipeline.

Creates sliding-window feature matrices, selects the most informative features
with a Random Forest, trains an Isolation Forest anomaly detector, and evaluates
detection performance with fault latency metrics.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)


def create_windows(
    df: pd.DataFrame,
    window_size: int,
    features: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window feature matrices for time-series anomaly detection.

    Each sample i contains a flattened view of the last `window_size` rows
    of `features`, labelled by `target_col[i]`. This gives the model temporal
    context without requiring recurrent architectures — critical for detecting
    faults that manifest as gradual signal deviations over ~0.2 seconds.

    Args:
        df: Prepared DataFrame (output of data_preparation).
        window_size: Number of timesteps per window. At 100 Hz, 20 → 0.2 s.
        features: Feature column names to include in the window.
        target_col: Binary label column (0 = normal, 1 = fault).

    Returns:
        X: Array of shape (n_samples, window_size × n_features)
        y: Array of shape (n_samples,) with binary labels
    """
    data = df[features].values
    labels = df[target_col].values
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(data[i - window_size : i].flatten())
        y.append(labels[i])
    return np.array(X), np.array(y)


def select_top_features_rf(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_top: int = 10,
) -> list[str]:
    """Rank features by Random Forest importance and return the top N.

    A supervised Random Forest is used here purely as a feature ranker —
    it exploits the labeled data to identify which signals change most
    discriminatively at fault onset, reducing the window dimension and
    improving Isolation Forest's unsupervised detection.

    Args:
        df: Prepared DataFrame.
        features: Candidate feature column names.
        target_col: Binary fault label column.
        n_top: Number of top features to return.

    Returns:
        List of the N most important feature names.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(df[features], df[target_col])
    importances = pd.Series(rf.feature_importances_, index=features)
    top = importances.nlargest(n_top).index.tolist()
    log.info("Top %d features selected: %s", n_top, top)
    return top


def split_windows(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temporal train/test split — preserves time order, no shuffling.

    Shuffling would leak future fault information into training, making
    evaluation artificially optimistic. The last (1 - train_ratio) fraction
    contains the fault event and is used exclusively for evaluation.

    Args:
        X: Feature matrix.
        y: Labels.
        train_ratio: Fraction of samples for training.

    Returns:
        X_train, X_test, y_train, y_test
    """
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


def fit_and_scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """Fit a RobustScaler on training data and transform both splits.

    RobustScaler uses median and IQR instead of mean and std, so anomalous
    windows present in the training set do not distort the scaling centre.
    A StandardScaler would shift "normal" toward the anomalies; RobustScaler
    anchors on the median, which represents normal flight.

    The scaler is fitted exclusively on X_train. Applying it to X_test
    simulates real deployment where future data is unseen at fit time —
    fitting on the full dataset would leak test-period anomaly statistics
    into the model's reference frame.

    The fitted scaler is saved to the catalog so it can be reused at
    inference time without refitting.

    Args:
        X_train: Training feature matrix (n_samples, n_features).
        X_test:  Test feature matrix  (n_samples, n_features).

    Returns:
        X_train_scaled, X_test_scaled, fitted RobustScaler.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log.info(
        "RobustScaler fitted on X_train=%s | X_test=%s transformed",
        X_train.shape,
        X_test.shape,
    )
    return X_train_scaled, X_test_scaled, scaler


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.06,
    n_estimators: int = 100,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest anomaly detector on normal flight windows.

    The `contamination` parameter sets the fraction of the training data
    the model assumes are anomalies, which determines the decision threshold.
    Too low → misses the fault; too high → triggers false alarms during normal flight.
    Tuning this is the main lever for balancing precision vs. recall (see 03_01).

    Args:
        X_train: Sliding-window training matrix (should contain mostly normal flight).
        contamination: Estimated fraction of anomalies. Typical range: 0.01 – 0.10.
        n_estimators: Number of isolation trees. More → more stable, slower to train.
        random_state: Seed for reproducibility.

    Returns:
        Fitted IsolationForest model.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    log.info(
        "IsolationForest trained | contamination=%.3f | n_estimators=%d | X_train=%s",
        contamination, n_estimators, X_train.shape,
    )
    return model


def evaluate_model(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_timestamps: np.ndarray,
) -> dict:
    """Evaluate the model and compute fault detection latency.

    Detection latency = time from real fault onset to first model alert.
    This is the primary metric for this application: we want the model to
    raise the alarm as quickly as possible after the motor stops.

    Args:
        model: Fitted IsolationForest.
        X_test: Test sliding-window matrix.
        y_test: True binary labels.
        test_timestamps: Timestamps aligned to X_test rows (for latency calc).

    Returns:
        Metrics dict containing:
        - classification_report: precision/recall/F1 per class
        - detection_latency_s: seconds from fault to first detection (None if undetected)
        - real_fault_time_s: timestamp of actual fault onset
        - predicted_fault_time_s: timestamp of first model alert
    """
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)  # IsolationForest: -1=anomaly, 1=normal

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics: dict = {"classification_report": report}

    fault_indices = np.where(y_test == 1)[0]
    if len(fault_indices) > 0:
        real_idx = fault_indices[0]
        post_fault_preds = np.where(
            (y_pred == 1) & (np.arange(len(y_pred)) >= real_idx)
        )[0]
        metrics["real_fault_time_s"] = float(test_timestamps[real_idx])

        if len(post_fault_preds) > 0:
            pred_idx = post_fault_preds[0]
            metrics["detection_latency_s"] = float(
                test_timestamps[pred_idx] - test_timestamps[real_idx]
            )
            metrics["predicted_fault_time_s"] = float(test_timestamps[pred_idx])
            log.info(
                "Fault at %.2fs | Detected at %.2fs | Latency: %.2fs",
                metrics["real_fault_time_s"],
                metrics["predicted_fault_time_s"],
                metrics["detection_latency_s"],
            )
        else:
            metrics["detection_latency_s"] = None
            metrics["predicted_fault_time_s"] = None
            log.warning("Fault present in test set but model never detected it.")
    else:
        log.info("No fault events in test set.")

    return metrics


def build_training_data(
    prepared_flights: dict,
    window_size: int,
    n_top_features: int,
    train_ratio: float,
    target_col: str,
    timestamp_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Combine all prepared flights, select features, create windows, and split.

    This is the first node in the model_training pipeline. It aggregates all
    flights into a single dataset to train and evaluate a single global model
    (as opposed to per-flight models).

    Pipeline:
    1. Concatenate all flights.
    2. Select top N features via Random Forest importance.
    3. Create sliding-window feature matrix per flight (avoids cross-flight boundary windows).
    4. Temporal train/test split.

    Args:
        prepared_flights: Dict from data_preparation (may contain callables from PartitionedDataset).
        window_size: Sliding window size in timesteps.
        n_top_features: Number of features to select.
        train_ratio: Temporal train/test split fraction.
        target_col: Binary fault label column name.
        timestamp_col: Timestamp column name.

    Returns:
        X_train, X_test, y_train, y_test, test_timestamps, selected_features
    """
    dfs = []
    for name, loader in prepared_flights.items():
        df = loader() if callable(loader) else loader
        dfs.append(df)

    combined = pd.concat(dfs, join="inner", ignore_index=True)
    log.info("Combined dataset shape: %s", combined.shape)

    features = [c for c in combined.columns if c not in [target_col, timestamp_col]]
    top_features = select_top_features_rf(combined, features, target_col, n_top_features)

    all_X, all_y, all_ts = [], [], []
    for df in dfs:
        Xi, yi = create_windows(df, window_size, top_features, target_col)
        all_X.append(Xi)
        all_y.append(yi)
        all_ts.append(df[timestamp_col].values[window_size:])
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    timestamps = np.concatenate(all_ts)

    X_train, X_test, y_train, y_test = split_windows(X, y, train_ratio)
    split_idx = int(len(timestamps) * train_ratio)
    test_timestamps = timestamps[split_idx:]

    log.info("X_train=%s | X_test=%s | features=%d", X_train.shape, X_test.shape, len(top_features))
    return X_train, X_test, y_train, y_test, test_timestamps, top_features
