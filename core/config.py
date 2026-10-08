from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "service_risk.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
REFERENCE_PATH = ROOT / "data" / "monitoring" / "reference.csv"
INFERENCE_LOG = ROOT / "data" / "monitoring" / "inference_log.jsonl"
MODEL_VERSION = "1.0.0"
RISK_THRESHOLD = 0.50
