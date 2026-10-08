from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "service_requests.csv"

def sigmoid(x): return 1 / (1 + np.exp(-x))

def main(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    priority = rng.choice(["low", "medium", "high", "critical"], n, p=[.25,.45,.22,.08])
    weekend = rng.random(n) < 2/7
    df = pd.DataFrame({
        "queue_minutes": np.clip(rng.gamma(2.2, 18, n), 0, 300),
        "estimated_work_minutes": np.clip(rng.lognormal(4.3, .55, n), 10, 600),
        "worker_load": rng.beta(3, 2, n),
        "worker_experience_months": np.clip(rng.gamma(2.5, 16, n), 0, 180).astype(int),
        "distance_km": np.clip(rng.gamma(2, 5, n), 0, 80),
        "priority": priority,
        "request_hour": rng.integers(0, 24, n),
        "is_weekend": weekend,
        "recent_worker_completion_rate": rng.beta(8, 2, n),
        "recent_category_delay_rate": rng.beta(2, 6, n),
    })
    pmap = {"low": -.35, "medium": 0, "high": .35, "critical": .75}
    z = (-3.2 + .012*df.queue_minutes + .004*df.estimated_work_minutes + 2.0*df.worker_load
         - .012*df.worker_experience_months + .035*df.distance_km
         + df.priority.map(pmap) + .45*df.is_weekend.astype(int)
         - 2.0*df.recent_worker_completion_rate + 3.2*df.recent_category_delay_rate
         + .35*((df.request_hour < 7) | (df.request_hour > 20)).astype(int))
    prob = sigmoid(z)
    df["sla_breached"] = rng.binomial(1, prob)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows to {OUT}; breach rate={df.sla_breached.mean():.3f}")
if __name__ == "__main__": main()
