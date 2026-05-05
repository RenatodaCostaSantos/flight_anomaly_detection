"""Nodes for the model_training pipeline.

Creates sliding-window feature matrices, selects the most informative features
with a Random Forest, trains an Isolation Forest anomaly detector, and evaluates
detection performance with fault latency metrics.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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


def select_top_features_effect_size(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_top: int = 10,
) -> list[str]:
    """Rank features by Cohen's d effect size and return the top N.

    For each feature, computes how strongly its distribution differs between
    normal (target=0) and fault (target=1) windows:

        Cohen's d = |mean_fault − mean_normal| / pooled_std

    This directly answers the question the Isolation Forest needs answered:
    "which features have the most different distribution in fault vs normal
    flight?" — regardless of *when* within the fault period they change.

    Advantages over Random Forest importance:
    - Non-parametric in spirit: does not assume linear or tree-based structure
    - Robust to class imbalance (normalises by within-group spread)
    - Does not reward temporal autocorrelation or spurious correlations with
      the *timing* of faults (a problem with lookahead-shifted labels)
    - Computationally lightweight: O(n × p) vs O(n × p × n_estimators)

    Features with zero pooled standard deviation (constant columns that
    survived earlier filtering) receive score 0 and sink to the bottom.

    Args:
        df: Combined DataFrame with all flights concatenated.
        features: Candidate feature column names.
        target_col: Binary fault label column (0 = normal, 1 = fault).
        n_top: Number of top features to return.

    Returns:
        List of the N features with the largest Cohen's d effect size.
    """
    normal = df.loc[df[target_col] == 0, features]
    fault  = df.loc[df[target_col] == 1, features]

    mean_diff = (fault.mean() - normal.mean()).abs()

    # Pooled standard deviation: sqrt(((n0-1)*s0² + (n1-1)*s1²) / (n0+n1-2))
    n0, n1 = len(normal), len(fault)
    pooled_std = np.sqrt(
        ((n0 - 1) * normal.std() ** 2 + (n1 - 1) * fault.std() ** 2) / (n0 + n1 - 2)
    ).replace(0, np.nan)

    scores = (mean_diff / pooled_std).fillna(0)
    top = scores.nlargest(n_top).index.tolist()
    log.info("Top %d features by Cohen's d: %s", n_top, top)
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
    max_samples: int | str = 128,
    max_features: float | int = 1.0,
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
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    log.info(
        "IsolationForest trained | contamination=%.3f | n_estimators=%d | max_samples=%s | max_features=%s | X_train=%s",
        contamination, n_estimators, max_samples, max_features, X_train.shape,
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
    skip_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Combine prepared flights, select features, create windows, and split by flight type.

    This is the first node in the model_training pipeline. It implements pure
    novelty detection: the Isolation Forest trains exclusively on normal (no_failure)
    flights, learning only what healthy flight looks like. Failure flights and a
    held-out portion of normal flights form the test set.

    Pipeline:
    1. Load all flights; skip initial transient phase (skip_seconds) per flight.
    2. Sort flights alphabetically — filenames encode the recording date, so
       alphabetical order is a reliable chronological proxy.
    3. Feature selection: Cohen's d is computed on ALL flights combined (needs
       both fault and normal labels to rank discriminative power).
    4. Temporal split of no_failure flights: first train_ratio fraction → training;
       remainder → test (validates false-positive rate on unseen normal data).
    5. All failure flights → test only (never seen during training).
    6. Build sliding-window matrices per flight to avoid cross-flight boundary artefacts.

    **Why train only on no_failure?**
    Isolation Forest is a novelty detector: it learns the density of its training
    distribution and flags points far from it. Training on contaminated data
    (normal + fault windows mixed) shifts the learned "normal" distribution toward
    the fault signature, degrading detection sensitivity. Training on pure normal
    data means any deviation from healthy flight is correctly scored as anomalous.

    **Why split no_failure flights temporally?**
    A row-level split within a concatenated dataset does not guarantee that the
    test set represents genuinely unseen flight conditions. A flight-level split
    ensures the model is evaluated on complete flights recorded after its training
    window — a more honest proxy for deployment generalisation.

    Args:
        prepared_flights: Dict from data_preparation (may contain callables from PartitionedDataset).
        window_size: Sliding window size in timesteps.
        n_top_features: Number of features to select.
        train_ratio: Fraction of no_failure flights used for training. The remaining
            no_failure flights plus all failure flights form the test set.
        target_col: Binary fault label column name.
        timestamp_col: Timestamp column name.
        skip_seconds: Seconds to skip from the start of each flight before building
            windows. Applied to all flights.

    Returns:
        X_train, X_test, y_train, y_test, test_timestamps, selected_features
    """
    # 1. Load all flights sorted alphabetically (chronological proxy via filename date)
    sorted_names = sorted(prepared_flights.keys())
    dfs_by_name: dict[str, pd.DataFrame] = {}
    for name in sorted_names:
        loader = prepared_flights[name]
        df = loader() if callable(loader) else loader
        if skip_seconds > 0.0:
            df = df[df[timestamp_col] >= skip_seconds].copy()
        dfs_by_name[name] = df

    # 2. Feature selection on all flights combined — needs fault labels for Cohen's d
    combined = pd.concat(list(dfs_by_name.values()), join="inner", ignore_index=True)
    log.info("Combined dataset shape (after skip): %s", combined.shape)
    features = [c for c in combined.columns if c not in [target_col, timestamp_col]]
    top_features = select_top_features_effect_size(combined, features, target_col, n_top_features)

    # 3. Separate flight names by type
    no_failure_names = [n for n in sorted_names if "no_failure" in n]
    failure_names    = [n for n in sorted_names if "no_failure" not in n]

    if not no_failure_names:
        raise ValueError("No no_failure flights found. Cannot train a novelty detector.")
    if not failure_names:
        log.warning("No failure flights found — test set contains only normal data.")

    # 4. Temporal split of no_failure flights
    n_train = max(1, int(len(no_failure_names) * train_ratio))
    train_names = no_failure_names[:n_train]
    val_names   = no_failure_names[n_train:]   # held-out normal flights (FP validation)
    test_names  = val_names + failure_names     # full test set (ordered chronologically)

    log.info(
        "Flight split — train (no_failure): %d | val (no_failure): %d | test (failure): %d",
        len(train_names), len(val_names), len(failure_names),
    )
    log.info("Train flights: %s", train_names)
    log.info("Test  flights: %s", test_names)

    # 5. Build training windows — no_failure only
    train_X_list, train_y_list = [], []
    for name in train_names:
        Xi, yi = create_windows(dfs_by_name[name], window_size, top_features, target_col)
        train_X_list.append(Xi)
        train_y_list.append(yi)
    X_train = np.concatenate(train_X_list)
    y_train = np.concatenate(train_y_list)

    # 6. Build test windows — held-out normal + all failure flights
    test_X_list, test_y_list, test_ts_list = [], [], []
    for name in test_names:
        df = dfs_by_name[name]
        Xi, yi = create_windows(df, window_size, top_features, target_col)
        test_X_list.append(Xi)
        test_y_list.append(yi)
        test_ts_list.append(df[timestamp_col].values[window_size:])
    X_test         = np.concatenate(test_X_list)
    y_test         = np.concatenate(test_y_list)
    test_timestamps = np.concatenate(test_ts_list)

    log.info(
        "X_train=%s (no_failure only) | X_test=%s | features=%d",
        X_train.shape, X_test.shape, len(top_features),
    )
    return X_train, X_test, y_train, y_test, test_timestamps, top_features
