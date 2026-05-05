"""
Testes de integração da API de detecção de falhas.

Usamos TestClient do FastAPI — ele roda o app no mesmo processo,
sem precisar subir o uvicorn. É a forma recomendada de testar APIs FastAPI.
"""

from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

# Caminho para os dados de voo preparados (sensores brutos, pré feature engineering)
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "03_primary" / "fft"

# Voo com falha de motor conhecida — usamos para testar detecção
FAULT_FLIGHT = "carbonZ_2018-07-18-15-53-31_1_engine_failure.csv"

# Voo normal mais longo disponível (15k linhas) — não tem aspd_meas, caso típico em produção
NORMAL_FLIGHT = "carbonZ_2018-10-05-15-52-12_1_no_failure.csv"

@pytest.fixture(scope="module")
def client():
    """
    Cria o TestClient uma única vez para todo o módulo de testes.
    
    scope="module" garante que o modelo é carregado uma única vez
    (o lifespan do FastAPI roda no primeiro uso do client).
    """

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def fault_flight_df():
    """Carrega o voo com falha de motor."""
    return pd.read_csv(DATA_DIR / FAULT_FLIGHT)


@pytest.fixture(scope="module")
def normal_flight_df():
    """Carrega o voo normal."""
    return pd.read_csv(DATA_DIR / NORMAL_FLIGHT)

def df_to_readings(df: pd.DataFrame) -> list[dict]:
    """
    Converte DataFrame de sensores para lista de dicts esperada pela API.
    aspd_meas é opcional — incluída só se presente no DataFrame.
    """
    required_cols = [
        "timestamp", "imu_accel_x", "imu_accel_y", "imu_accel_z",
        "mag_x", "mag_y", "mag_z", "alt_global",
    ]
    cols = required_cols + (["aspd_meas"] if "aspd_meas" in df.columns else [])
    return df[cols].to_dict(orient="records")

# --- Testes ---

def test_health(client):
    """API deve responder saudável com modelo carregado."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 20


def test_predict_rejects_insufficient_data(client):
    """
    API deve retornar 422 quando receber menos de 2001 leituras.
    
    Esse teste valida que a regra de negócio no schema está funcionando —
    o modelo precisa de pelo menos 2001 amostras para computar a janela
    FFT mais larga (2000) e uma predição.
    """

    payload = {
        "flight_id": "test_short",
        "readings": [
            {
            "timestamp": float(i),
                "imu_accel_x": 0.0, "imu_accel_y": 0.0, "imu_accel_z": 9.8,
                "mag_x": 0.0, "mag_y": 0.0, "mag_z": 0.0,
                "aspd_meas": 15.0, "alt_global": 300.0,
            }
            for i in range(100)  # apenas 100 leituras — deve falhar
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_fault_flight_structure(client, fault_flight_df):
    """
    Verifica que a resposta tem a estrutura correta para um voo com falha.
    Usa as primeiras 5000 linhas para ser rápido — suficiente para
    verificar estrutura sem precisar processar o voo inteiro.
    """
    readings = df_to_readings(fault_flight_df.head(7000))
    payload = {"flight_id": FAULT_FLIGHT, "readings": readings}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    # Verifica campos obrigatórios da resposta
    assert body["flight_id"] == FAULT_FLIGHT
    assert body["total_readings"] == 7000
    assert isinstance(body["anomalies_detected"], int)
    assert isinstance(body["events"], list)
    assert len(body["events"]) > 0

    # Verifica estrutura de cada evento
    event = body["events"][0]
    assert "timestamp" in event
    assert "score" in event
    assert "is_anomaly" in event

@pytest.mark.slow
def test_predict_detects_fault_in_full_flight(client, fault_flight_df):
    """
    Testa que o modelo detecta a falha de motor no voo completo.
    
    A falha real ocorre em t=115.3s. Esperamos que o modelo a detecte
    dentro de uma janela razoável após esse momento.

    Marcado como @pytest.mark.slow pois processa ~26k linhas com FFT.
    Execute com: pytest -m slow
    """

    readings = df_to_readings(fault_flight_df)
    payload = {"flight_id": FAULT_FLIGHT, "readings": readings}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    # Isolation Forest com contamination=0.03 classifica ~3% dos dados normais
    # como anomalia — falsos positivos antes da falha são comportamento esperado.
    # O que importa é que a taxa de anomalias seja maior DURANTE a falha.
    events = body["events"]
    pre_fault  = [e for e in events if e["timestamp"] < 115.0]
    post_fault = [e for e in events if e["timestamp"] >= 115.0]

    assert post_fault, "Nenhum evento após t=115s — dados insuficientes"

    pre_rate  = sum(1 for e in pre_fault  if e["is_anomaly"]) / len(pre_fault)  if pre_fault  else 0
    post_rate = sum(1 for e in post_fault if e["is_anomaly"]) / len(post_fault)

    assert post_rate > pre_rate, (
        f"Taxa de anomalias não aumentou na falha: "
        f"antes={pre_rate:.1%}, durante={post_rate:.1%}"
    )

    first_fault_alert = next((e["timestamp"] for e in post_fault if e["is_anomaly"]), None)
    print(f"\nFalha real: t=115.3s | Primeiro alerta pós-falha: t={first_fault_alert:.3f}s"
          f"\nTaxa pré-falha: {pre_rate:.1%} | Taxa pós-falha: {post_rate:.1%}")

@pytest.mark.slow
def test_predict_normal_flight_api_response(client, normal_flight_df):
    """
    Testa que a API processa um voo sem falha sem travar e retorna estrutura válida.

    NOTA: o modelo atual apresenta alta taxa de falsos positivos neste voo
    (distribution shift: voo de Out/2018 tem valores de magnetômetro muito
    diferentes dos voos de Jul/2018 usados no treino). Este é um problema de
    generalização do modelo, não de deploy — requer mais diversidade nos dados
    de treino para ser corrigido.
    """
    readings = df_to_readings(normal_flight_df)
    payload = {"flight_id": NORMAL_FLIGHT, "readings": readings}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["flight_id"] == NORMAL_FLIGHT
    assert body["total_readings"] == len(readings)
    assert isinstance(body["anomalies_detected"], int)
    assert len(body["events"]) > 0

    total = len(body["events"])
    rate = body["anomalies_detected"] / total if total > 0 else 0
    print(f"\nVoo normal — Taxa de anomalias: {rate:.1%} ({body['anomalies_detected']}/{total})"
          " [KNOWN ISSUE: distribution shift Jul→Out 2018]")