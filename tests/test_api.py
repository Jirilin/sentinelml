from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
VALID = {"queue_minutes":20,"estimated_work_minutes":90,"worker_load":0.5,"worker_experience_months":24,"distance_km":5,"priority":"medium","request_hour":14,"is_weekend":False,"recent_worker_completion_rate":0.9,"recent_category_delay_rate":0.1}

def test_health():
    r=client.get('/health'); assert r.status_code==200; assert r.json()['status']=='ok'
def test_predict_contract():
    r=client.post('/predict', json=VALID); assert r.status_code==200; body=r.json(); assert 0<=body['risk_probability']<=1; assert body['prediction'] in ['on_time','sla_risk']
def test_validation_rejects_bad_load():
    bad={**VALID,'worker_load':1.5}; assert client.post('/predict',json=bad).status_code==422
