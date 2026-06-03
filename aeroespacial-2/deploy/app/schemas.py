from pydantic import BaseModel, Field
from typing import List


class SensorReading(BaseModel):
    """Uma linha de telemetria do voo (nomes padronizados pós data_preparation)."""
    timestamp: float
    hud_throttle: float
    err_vel_z: float
    alt_global: float
    alt_gps_fix: float
    vel_z_meas: float
    vel_z_local: float
    pos_z_local: float
    vel_z_twist: float
    aspd_meas: float | None = None  # opcional: usado em energy_specific


class PredictRequest(BaseModel):
    """Corpo da requisição POST /predict."""
    flight_id: str = Field(..., description="Identificador do voo")
    readings: List[SensorReading] = Field(
        ...,
        min_length=2020,  # mínimo: janela FFT de 2000 + sliding window de 20
        description="Leituras de sensores em ordem cronológica",
    )


class AnomalyEvent(BaseModel):
    """Um evento de anomalia detectado."""
    timestamp: float
    score: float       # quanto mais negativo, mais anômalo
    is_anomaly: bool


class PredictResponse(BaseModel):
    """Corpo da resposta do /predict."""
    flight_id: str
    total_readings: int
    anomalies_detected: int
    first_anomaly_at: float | None  # timestamp do primeiro alerta
    events: List[AnomalyEvent]
