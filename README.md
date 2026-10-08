# SentinelML — Adaptive Service-Risk Prediction Platform

A production-style ML/API/MLOps portfolio project that predicts SLA breach risk for incoming service requests and exposes the model through FastAPI. It includes reproducible synthetic data, preprocessing + model pipeline, MLflow experiment tracking, strict API validation, Prometheus metrics, inference logging, PSI drift checks, tests, Docker and GitHub Actions CI.

## Architecture
Client -> FastAPI -> Pydantic validation -> sklearn Pipeline -> probability/decision -> telemetry/logging -> drift check
Training data -> preprocessing/training -> evaluation -> MLflow + versioned artifact -> API

## Local quick start
Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py
pytest -q
uvicorn app.main:app --reload --port 8000
```

Open API docs at `http://127.0.0.1:8000/docs`, health at `/health`, and Prometheus metrics at `/metrics`.

## Example prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"queue_minutes":85,"estimated_work_minutes":180,"worker_load":0.88,"worker_experience_months":5,"distance_km":22,"priority":"critical","request_hour":21,"is_weekend":true,"recent_worker_completion_rate":0.66,"recent_category_delay_rate":0.48}'
```

## MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then open `http://127.0.0.1:5000`.

## Drift
After at least several prediction calls:
```bash
python scripts/check_drift.py
```
PSI is used as a lightweight portfolio-friendly drift signal, not as proof that a model must be retrained. Investigate shifts with adequate sample sizes and business/model-performance evidence.

## Docker
```bash
docker compose up --build
```

## Security notes
- Request schema rejects unexpected fields and bounds numeric inputs.
- Container runs as a non-root user.
- Never load untrusted pickle/joblib artifacts; these formats can execute arbitrary code.
- In a public deployment add TLS at the ingress, authentication/authorization, rate limiting, secrets management, dependency/image scanning and centralised immutable logs.

## Portfolio extensions
Add PostgreSQL-backed MLflow Registry, champion/challenger aliases, cloud object storage, Evidently reports, Grafana dashboards, scheduled retraining, canary deployment, feature store, and Kubernetes only after the core system is understood and tested.
