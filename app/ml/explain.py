def explain(row: dict) -> list[str]:
    scored: list[tuple[float, str]] = []
    rules = [
        (row["queue_minutes"] / 120, "long queue time"),
        (row["estimated_work_minutes"] / 240, "large estimated workload"),
        (row["worker_load"], "high worker utilisation"),
        (row["distance_km"] / 40, "long service distance"),
        (row["recent_category_delay_rate"], "category has recent delays"),
        (1 - row["recent_worker_completion_rate"], "worker's recent completion rate is lower"),
        (max(0, 12 - row["worker_experience_months"]) / 12, "limited worker experience"),
        (0.75 if row["priority"] == "critical" else 0.4 if row["priority"] == "high" else 0, "high-priority request"),
        (0.3 if row["is_weekend"] else 0, "weekend request"),
    ]
    for score, label in rules:
        if score > 0.25:
            scored.append((float(score), label))
    return [label for _, label in sorted(scored, reverse=True)[:3]] or ["no dominant heuristic risk factor"]
