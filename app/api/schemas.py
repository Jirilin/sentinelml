from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queue_minutes: float = Field(ge=0, le=1440)
    estimated_work_minutes: float = Field(gt=0, le=2880)
    worker_load: float = Field(ge=0, le=1)
    worker_experience_months: int = Field(ge=0, le=600)
    distance_km: float = Field(ge=0, le=500)
    priority: Literal["low", "medium", "high", "critical"]
    request_hour: int = Field(ge=0, le=23)
    is_weekend: bool
    recent_worker_completion_rate: float = Field(ge=0, le=1)
    recent_category_delay_rate: float = Field(ge=0, le=1)

class PredictionResponse(BaseModel):
    prediction: Literal["on_time", "sla_risk"]
    risk_probability: float
    threshold: float
    model_version: str
    top_risk_factors: list[str]
