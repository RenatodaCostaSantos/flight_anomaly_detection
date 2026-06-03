import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .inference import AllFeaturesInferencePipeline
from .schemas import AnomalyEvent, PredictRequest, PredictResponse

STATIC_DIR = Path(__file__).parent / "static"


# Variável global para o pipeline (carregado uma vez na inicialização)
pipeline: AllFeaturesInferencePipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os artefatos de modelo quando o servidor inicia."""
    global pipeline
    print("Carregando artefatos do modelo all_features...")
    pipeline = AllFeaturesInferencePipeline()
    print(f"Modelo carregado. Features: {pipeline.selected_features[:3]}...")
    yield
    pipeline = None

app = FastAPI(
    title="Motor Failure Detection API",
    description="Detecta falhas de motor em UAVs (all_features model, F1=0.86).",
    version="2.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

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
    first_anomaly = anomaly_events[0].timestamp if anomaly_events else None

    return PredictResponse(
        flight_id = request.flight_id,
        total_readings = len(request.readings),
        anomalies_detected = len(anomaly_events),
        first_anomaly_at = first_anomaly,
        events = events,
    )


@app.post("/stream")
async def stream_predict(
    file: UploadFile = File(...),
    speed: float = Form(10.0),
):
    """
    Aceita um CSV de voo e retorna predições via Server-Sent Events.

    Protocolo de eventos (todos JSON no campo `data:`):
      {type: "processing"}           — inferência em andamento
      {type: "meta", ...}            — metadados do voo após inferência
      {type: "batch", points: [...]} — lote de predições
      {type: "done", ...}            — resumo final
      {type: "error", message: "…"}  — erro durante inferência
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    has_fault = "target_fault" in df.columns and (df["target_fault"] == 1).any()
    fault_start = (
        float(df.loc[df["target_fault"] == 1, "timestamp"].min()) if has_fault else None
    )

    async def event_generator():
        # Sinaliza imediatamente que a inferência começou (pode levar alguns segundos)
        yield f"data: {json.dumps({'type': 'processing'})}\n\n"

        # pipeline.predict() é CPU-bound — roda em thread separada para não bloquear o event loop
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(None, pipeline.predict, df)
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'meta', 'flight_id': file.filename or 'unknown', 'duration': float(df['timestamp'].max() - df['timestamp'].min()), 'n_predictions': len(results), 'fault_start': fault_start, 'threshold': float(pipeline.model.offset_)})}\n\n"

        # Envia lotes de ~50ms de tempo real independente da velocidade → ~20fps no gráfico
        REAL_BATCH_S = 0.05
        flight_batch_s = REAL_BATCH_S * speed  # quantos segundos de voo cabem num lote

        rows = list(results.itertuples(index=False))
        i = 0
        anomalies = 0
        first_anomaly = None

        while i < len(rows):
            batch_end_ts = rows[i].timestamp + flight_batch_s
            batch = []

            while i < len(rows) and rows[i].timestamp < batch_end_ts:
                row = rows[i]
                is_anomaly = bool(row.is_anomaly)
                if is_anomaly:
                    anomalies += 1
                    if first_anomaly is None:
                        first_anomaly = float(row.timestamp)
                batch.append({
                    "timestamp": float(row.timestamp),
                    "score": float(row.score),
                    "is_anomaly": is_anomaly,
                })
                i += 1

            yield f"data: {json.dumps({'type': 'batch', 'points': batch})}\n\n"
            await asyncio.sleep(REAL_BATCH_S)

        yield f"data: {json.dumps({'type': 'done', 'anomalies_detected': anomalies, 'first_anomaly_at': first_anomaly})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )