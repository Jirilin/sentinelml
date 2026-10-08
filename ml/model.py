import json
from functools import lru_cache
import joblib
import pandas as pd
from app.core.config import METADATA_PATH, MODEL_PATH
from app.ml.features import ALL_FEATURES

@lru_cache(maxsize=1)
def load_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run: python scripts/train.py")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata

def predict_probability(payload: dict) -> tuple[float, dict]:
    model, metadata = load_bundle()
    frame = pd.DataFrame([{k: payload[k] for k in ALL_FEATURES}])
    probability = float(model.predict_proba(frame)[0, 1])
    return probability, metadata
