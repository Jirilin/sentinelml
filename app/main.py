import time
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
from app.api.schemas import PredictionRequest, PredictionResponse
from app.core.config import RISK_THRESHOLD
from app.core.telemetry import LATENCY, log_prediction
from app.ml.explain import explain
from app.ml.model import load_bundle, predict_probability

app = FastAPI(title="SentinelML Service Risk API", version="1.0.0", description="Production-style SLA risk prediction service")
app.mount("/metrics", make_asgi_app())

@app.get("/health")
def health():
    try:
        _, metadata = load_bundle()
        return {"status": "ok", "model_version": metadata["model_version"]}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = time.perf_counter()
    payload = request.model_dump()
    probability, metadata = predict_probability(payload)
    prediction = "sla_risk" if probability >= RISK_THRESHOLD else "on_time"
    latency = time.perf_counter() - start
    LATENCY.observe(latency)
    log_prediction(payload, probability, prediction, latency)
    return PredictionResponse(
        prediction=prediction,
        risk_probability=round(probability, 4),
        threshold=RISK_THRESHOLD,
        model_version=metadata["model_version"],
        top_risk_factors=explain(payload),
    )
