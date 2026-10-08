import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.core.config import INFERENCE_LOG, REFERENCE_PATH
from app.ml.features import NUMERIC_FEATURES

def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0,1,bins+1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a-e)*np.log(a/e)))

def main():
    if not Path(INFERENCE_LOG).exists(): raise SystemExit("No inference log yet. Call /predict first.")
    ref = pd.read_csv(REFERENCE_PATH)
    rows = [json.loads(line)["features"] for line in Path(INFERENCE_LOG).read_text().splitlines() if line.strip()]
    cur = pd.DataFrame(rows)
    if len(cur) < 20: print("Warning: fewer than 20 production observations; drift estimate is unstable.")
    scores = {c: round(psi(ref[c].astype(float).to_numpy(), cur[c].astype(float).to_numpy()), 4) for c in NUMERIC_FEATURES}
    print(json.dumps({"psi": scores, "interpretation":"<0.10 little change; 0.10-0.25 investigate; >0.25 material shift (heuristic thresholds)"}, indent=2))
if __name__ == "__main__": main()
