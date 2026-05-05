from pydantic import BaseModel, Field
from typing import List


class SensorReading(BaseModel):
    """Uma linha de telemetria do voo."""
    timestamp: float
    imu_accel_x: float
    imu_accel_y: float
    imu_accel_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    aspd_meas: float | None = None   # opcional: não está entre as features selecionadas
    alt_global: float

class PredictRequest(BaseModel):
    """Corpo da requisição POST /predict."""
    flight_id: str = Field(...,description='Identificador do voo')
    readings: List[SensorReading] = Field(
        ...,
        min_length=2001, # mínimo para janela FFT de 2000 + window de 20
        description='Leituras de sensores em ordem cronológica'
    )

class AnomalyEvent(BaseModel):
    """Um evento de anomalia detectado."""
    timestamp: float
    score: float            # quanto mais negativo, mais anômalo
    is_anomaly: bool

class PredictResponse(BaseModel):
    """Corpo da resposta do /predict."""
    flight_id: str
    total_readings: int
    anomalies_detected: int
    first_anomaly_at: float | None   # timestamp do primeiro alerta
    events: List[AnomalyEvent]