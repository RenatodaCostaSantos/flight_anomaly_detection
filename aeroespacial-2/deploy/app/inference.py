import json
import joblib
import numpy as np
import os
import pandas as pd
import re
from pathlib import Path
from typing import List
from numpy.lib.stride_tricks import sliding_window_view

# Em desenvolvimento: resolve relativo ao arquivo (aeroespacial-2/data/06_models)
# Em Docker: sobrescrito pela variável de ambiente MODELS_DIR
_default_models_dir = Path(__file__).parent.parent.parent / "data" / "06_models"
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(_default_models_dir)))

G: float = 9.81


class AllFeaturesInferencePipeline:
    """
    Replica o pipeline de feature engineering do treinamento para inferência.

    Artefatos: isolation_forest.pkl, feature_scaler.pkl, selected_features.json
    Modelo treinado com F1 = 0.86 no conjunto de teste.
    """

    WINDOW_SIZE = 20       # janela deslizante de timesteps
    SKIP_SECONDS = 20.0    # ignorar início do voo (spin-up dos sensores)

    def __init__(self):
        self._load_artifacts()

    def _load_artifacts(self):
        """Carrega os artefatos treinados uma única vez na inicialização."""
        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        model_path = MODELS_DIR / "isolation_forest.pkl"
        features_path = MODELS_DIR / "selected_features.json"

        if not all(p.exists() for p in [scaler_path, model_path, features_path]):
            raise FileNotFoundError(
                f"Artefatos de modelo não encontrados em {MODELS_DIR}. "
                "Execute o pipeline Kedro de treinamento primeiro."
            )

        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)

        with open(features_path) as f:
            self.selected_features: List[str] = json.load(f)

        # Deriva as janelas necessárias diretamente dos features selecionados.
        # Garante que retreinamentos com janelas diferentes funcionem automaticamente.
        self.ROLLING_WINDOWS = self._extract_windows(r"_(?:mean|std|slope)_(\d+)$")
        self.FFT_WINDOWS = self._extract_windows(r"^fft_(?:peak_power|entropy|high_ratio)_.+_(\d+)$")

    def _extract_windows(self, pattern: str) -> list[int]:
        """Extrai os tamanhos de janela únicos dos nomes de features selecionadas."""
        windows = set()
        for feature in self.selected_features:
            m = re.search(pattern, feature)
            if m:
                windows.add(int(m.group(1)))
        return sorted(windows)

    def _compute_specific_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """energy_specific = alt_global + aspd_meas² / (2g). aspd_meas é opcional."""
        df = df.copy()
        if "aspd_meas" in df.columns and not df["aspd_meas"].isna().all():
            airspeed = df["aspd_meas"].fillna(0.0)
        else:
            airspeed = pd.Series(0.0, index=df.index)
        df["energy_specific"] = df["alt_global"] + airspeed ** 2 / (2 * G)
        return df

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean, std e slope para os sinais base das features selecionadas."""
        base_signals: set[str] = set()
        for feat in self.selected_features:
            m = re.match(r"^(.+?)_(?:mean|std|slope)_\d+$", feat)
            if m:
                base_signals.add(m.group(1))

        dt_median = df["timestamp"].diff().median()
        if pd.isna(dt_median) or dt_median <= 1e-9:
            dt_median = 0.01  # fallback para 100 Hz

        for signal in base_signals:
            if signal not in df.columns:
                continue
            for window in self.ROLLING_WINDOWS:
                roll = df[signal].rolling(window, min_periods=1)
                df[f"{signal}_mean_{window}"] = roll.mean()
                df[f"{signal}_std_{window}"] = roll.std().fillna(0)
                window_duration = window * dt_median
                df[f"{signal}_slope_{window}"] = (
                    df[signal].diff(window) / window_duration
                ).fillna(0)

        return df

    def _compute_fft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """FFT spectral features com sliding_window_view (idêntico ao Kedro)."""
        fft_signals: set[str] = set()
        for feat in self.selected_features:
            m = re.match(r"^fft_(?:peak_power|entropy|high_ratio)_(.+)_\d+$", feat)
            if m:
                fft_signals.add(m.group(1))

        n = len(df)
        for signal in fft_signals:
            if signal not in df.columns:
                continue
            values = pd.Series(df[signal].values.astype(float)).ffill().fillna(0).values

            for window in self.FFT_WINDOWS:
                peak_power = np.zeros(n)
                entropy = np.zeros(n)
                high_ratio = np.zeros(n)

                if n >= window:
                    wins = sliding_window_view(values, window_shape=window)
                    mags = np.abs(np.fft.rfft(wins, axis=1))

                    ac = mags[:, 1:]  # exclui componente DC
                    power = ac ** 2
                    total = power.sum(axis=1, keepdims=True).clip(min=1e-12)

                    peak = ac.max(axis=1)

                    p_norm = power / total
                    n_bins = ac.shape[1]
                    ent = (
                        -np.sum(p_norm * np.log(p_norm + 1e-12), axis=1)
                        / np.log(n_bins + 1)
                    )

                    mid = n_bins // 2
                    ratio = (power[:, mid:].sum(axis=1) / total[:, 0]).clip(0.0, 1.0)

                    # Alinha saída: preenche linhas iniciais com o primeiro valor válido
                    offset = window - 1
                    peak_power[offset:] = peak
                    entropy[offset:] = ent
                    high_ratio[offset:] = ratio
                    peak_power[:offset] = peak[0]
                    entropy[:offset] = ent[0]
                    high_ratio[:offset] = ratio[0]

                df[f"fft_peak_power_{signal}_{window}"] = peak_power
                df[f"fft_entropy_{signal}_{window}"] = entropy
                df[f"fft_high_ratio_{signal}_{window}"] = high_ratio

        return df

    def _build_sliding_window(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cria janelas deslizantes de WINDOW_SIZE timesteps (igual ao Kedro)."""
        feature_matrix = df[self.selected_features].values
        timestamps = df["timestamp"].values
        X, ts = [], []
        for i in range(self.WINDOW_SIZE, len(feature_matrix)):
            X.append(feature_matrix[i - self.WINDOW_SIZE: i].flatten())
            ts.append(timestamps[i])
        return np.array(X), np.array(ts)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de inferência.

        Args:
            df: DataFrame com colunas [timestamp, hud_throttle, err_vel_z,
                alt_global, alt_gps_fix, vel_z_meas, vel_z_local, pos_z_local,
                vel_z_twist] e opcionalmente [aspd_meas].

        Returns:
            DataFrame com colunas [timestamp, score, is_anomaly].
        """
        df = df.copy().reset_index(drop=True)

        # Skip adaptativo: se o voo for curto, usa as últimas min_required linhas
        min_required = (self.FFT_WINDOWS[-1] if self.FFT_WINDOWS else 0) + self.WINDOW_SIZE
        if len(df) < min_required:
            raise ValueError(
                f"Dados insuficientes: {len(df)} linhas (mínimo {min_required})."
            )

        t0 = df["timestamp"].iloc[0]
        rows_after_skip = (df["timestamp"] >= t0 + self.SKIP_SECONDS).sum()
        if rows_after_skip >= min_required:
            df = df[df["timestamp"] >= t0 + self.SKIP_SECONDS].reset_index(drop=True)
        else:
            df = df.tail(min_required).reset_index(drop=True)

        # Feature engineering — ordem idêntica ao pipeline Kedro de treinamento
        df = self._compute_specific_energy(df)
        df = self._compute_rolling_features(df)
        df = self._compute_fft_features(df)

        df = df.dropna(subset=self.selected_features).reset_index(drop=True)

        X, timestamps = self._build_sliding_window(df)
        X_scaled = self.scaler.transform(X)
        raw_preds = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        return pd.DataFrame({
            "timestamp": timestamps,
            "score": scores,
            "is_anomaly": raw_preds == -1,
        })
