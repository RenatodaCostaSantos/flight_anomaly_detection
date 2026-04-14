import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# Caminhos relativos ao arquivo de modelos treinados
MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "06_models"



class FFTInferencePipeline:
    """
    Replica o pipeline de feature engineering do treinamento para inferência.
    
    Esta classe encapsula tudo que o modelo precisa para fazer uma predição:
    1. Os artefatos treinados (scaler, modelo, lista de features)
    2. A lógica de feature engineering (deve ser idêntica ao treinamento)
    """

    # Janelas usadas no treinamento (devem ser idênticas às do Kedro)
    ROLLING_WINDOWS = [50,100,200]
    FFT_WINDOWS = [500,1000,2000]
    WINDOW_SIZE = 20        # janela deslizante para o modelo
    SKIP_SECONDS = 20.0     # ignorar início do voo (spin-up)

    # Sinais usados na feature engineering FFT
    FFT_SIGNALS = [
       "imu_accel_x", "imu_accel_y", "imu_accel_z",
        "mag_x", "mag_y", "mag_z",
        "aspd_meas", "alt_global"
    ]

    def __init__(self):
        self._load_artifacts()

    def _load_artifacts(self):
        """Carrega os artefatos treinados uma única vez na inicialização."""
        scaler_path = MODELS_DIR / "fft_feature_scaler.pkl"
        model_path = MODELS_DIR / "fft_isolation_forest.pkl"
        features_path = MODELS_DIR / "fft_selected_features.json"

        if not all(p.exists() for p in [scaler_path, model_path, features_path]):
            raise FileNotFoundError(
                f"Artefatos de modelo não encontrados em {MODELS_DIR}. "
                "Execute o pipeline Kedro de treinamento primeiro."
            )
        
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)

        with open(features_path) as f:
            self.selected_features: List[str] = json.load(f)

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computa rolling mean para cada sinal e janela.
        Réplica exata do nó feature_engineering do Kedro.
        """

        signals = [s for s in self.FFT_SIGNALS if s != "alt_global"]

        for signal in signals:
            if signal not in df.columns:
                continue
            for window in self.ROLLING_WINDOWS:
                df[f"{signal}_mean_{window}"] = (
                    df[signal].rolling(window=window, min_periods=1).mean()
                )
        return df
    
    def _compute_fft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computa features espectrais (FFT) para cada sinal e janela.
        
        Para cada ponto no tempo, olha para trás `window` amostras
        e computa 3 descritores do espectro de frequência.
        """
        for signal in self.FFT_SIGNALS:
            if signal not in df.columns:
                continue
            for window in self.FFT_WINDOWS:
                peak_power_col = f"fft_peak_power_{signal}_{window}"
                entropy_col = f"fft_entropy_{signal}_{window}"
                high_ratio_col = f"fft_high_ratio_{signal}_{window}"

                # Inciializa com NaN

                df[peak_power_col] = np.nan
                df[entropy_col] = np.nan
                df[high_ratio_col] = np.nan

                values = df[signal].values

                # Computa FFT em cada janela (computacionalmente intensivo)
                for i in range(window, len(values)):
                    segment = values[i - window: i]
                    fft_vals = np.abs(np.fft.rfft(segment)) ** 2 

                    total_power = fft_vals.sum()
                    if total_power == 0:
                        continue

                    # Peak power: pico máximo do espectro 
                    df.at[i, peak_power_col] = fft_vals.max()

                    # Entropia espectral: uniformidade da distribuição de frequencias
                    psd_norm = fft_vals / total_power
                    psd_norm = psd_norm[psd_norm > 0]
                    df.at[i, entropy_col] = -np.sum(psd_norm * np.log(psd_norm))

                    #High frequency ratio: proporção de energia em altas frequências
                    mid = len(fft_vals) // 2
                    df.at[i, high_ratio_col] = fft_vals[mid:].sum() / total_power

        return df
    
    def _build_sliding_window(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cria janelas deslizantes de WINDOW_SIZE timesteps.
        Idêntico ao nó build_training_data do Kedro.
        
        Retorna:
            X: array (n_samples, WINDOW_SIZE * n_features)
            timestamps: array (n_samples,)
        """

        feature_matrix = df[self.selected_features].values
        timestamps = df["timestamp"].values

        X, ts = [], []
        for i in range(self.WINDOW_SIZE, len(feature_matrix)):
            window = feature_matrix[i -self.WINDOW_SIZE: i].flatten()
            X.append(window)
            ts.append(timestamps[i])
        
        return np.array(X), np.array(ts)
    
    def predict(self, df: pd.Dataframe) -> pd.Datraframe:
        """
        Pipeline completo de inferência.
        
        Recebe um DataFrame com colunas de sensor bruto e retorna
        um DataFrame com scores de anomalia por timestamp.
        
        Args:
            df: DataFrame com colunas [timestamp, imu_accel_x, ..., alt_global]
            
        Returns:
            DataFrame com colunas [timestamp, score, is_anomaly]
        """

        df = df.copy().reset_index(drop=True)

        # 1. Filtrar início do voo (spin-up do sensor)
        t0 = df["timestamp"].iloc[0]
        df = df[df["timestamp"] >= t0 + self.SKIP_SECONDS].reset_index(drop=True)

        if len(df) < self.FFT_WINDOWS[-1] + self.WINDOW_SIZE:
            raise ValueError(
                f"Dados insuficientes após filtrar os primeiros {self.SKIP_SECONDS}s. "
                f"Necessário: {self.FFT_WINDOWS[-1] + self.WINDOW_SIZE} amostras."
            )

        # 2. Feature engineering (deve ser idêntico ao treinamento)
        df = self._compute_rolling_features(df)
        df = self._compute_fft_features(df)

        # 3. Remover linhas com NaN (janelas incompletas no início)
        df = df.dropna(subset=self.selected_features).reset_index(drop=True)

         # 4. Janelas deslizantes
        X, timestamps = self._build_sliding_windows(df)

        # 5. Normalização (com o scaler treinado — nunca fit novamente!)
        X_scaled = self.scaler.transform(X)

         # 6. Predição: -1 = anomalia, 1 = normal
        raw_preds = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)  # quanto mais negativo, mais anômalo

        return pd.DataFrame({
            "timestamp": timestamps,
            "score": scores,
            "is_anomaly": raw_preds == -1
        })



        

