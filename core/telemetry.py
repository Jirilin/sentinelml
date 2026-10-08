import json
import time
from pathlib import Path
from threading import Lock
from prometheus_client import Counter, Histogram
from app.core.config import INFERENCE_LOG

REQUESTS = Counter("sentinelml_predictions_total", "Prediction requests", ["prediction"])
LATENCY = Histogram("sentinelml_prediction_latency_seconds", "Prediction latency")
_lock = Lock()

def log_prediction(features: dict, probability: float, prediction: str, latency: float) -> None:
    REQUESTS.labels(prediction=prediction).inc()
    record = {"timestamp": time.time(), "features": features, "risk_probability": probability, "prediction": prediction, "latency_seconds": latency}
    Path(INFERENCE_LOG).parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(INFERENCE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
