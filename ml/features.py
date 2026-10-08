NUMERIC_FEATURES = [
    "queue_minutes", "estimated_work_minutes", "worker_load",
    "worker_experience_months", "distance_km", "request_hour",
    "recent_worker_completion_rate", "recent_category_delay_rate",
]
CATEGORICAL_FEATURES = ["priority", "is_weekend"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
