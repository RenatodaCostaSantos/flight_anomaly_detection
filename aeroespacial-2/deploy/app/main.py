from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import pandas as pd

from .inference import FFTInferencePipeline
from .schemas import PredictRequest, PredictResponse, AnomalyEvent


# Variável global para o pipeline (carregado uma vez na inicialização)
pipeline: FFTInferencePipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carrega os artefatos de modelo quando o servidor inicia.
    
    Por que lifespan e não no topo do arquivo?
    Porque o modelo demora para carregar (~1s). Com lifespan,
    você garante que está pronto antes de aceitar requisições.
    """
    global pipeline
    print("Carregando artefatos do modelo FFT...")
    pipeline = FFTInferencePipeline()
    print(f"Modelo carregado. Features: {pipeline.selected_features[:3]}...")
    yield
    #Cleanup ao desligar (se necessário)
    pipeline = None

app = FastAPI(
    title = "Motor Failure Detection API",
    description="Detecta falhas de motor  em UAVs usando features espectrais (FFT).",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    """Endpoint de saúde - usado por load balancers e Docker healthcheck"""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "n_features": len(pipeline.selected_features) if pipeline else 0,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Recebe leituras de sensores de um voo e retorna eventos de anomalia.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    
    #Converte as leituras Pydantic para DataFrame
    df = pd.DataFrame([r.model_dump() for r in request.readings])

    try:
        results = pipeline.predict(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    #Monta a resposta
    events = [
        AnomalyEvent(
            timestamp=float(row["timestamp"]),
            score=float(row["score"]),
            is_anomaly=bool(row["is_anomaly"]),
        )
        for _,row in results.iterrows()
    ]

    anomaly_events = [e for e in events if e.is_anomaly]
    first_anomaly