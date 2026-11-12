from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from app.ml.features import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[1]

SOURCE_MODEL = (
    ROOT
    / "reports/model_comparison"
    / "logistic_regression_unweighted.joblib"
)

SOURCE_METRICS = (
    ROOT
    / "reports/candidate_evaluation"
    / "candidate_metrics.json"
)

RELEASE_DIR = (
    ROOT
    / "models/releases/sentinelml-v2-rc1"
)

RELEASE_MODEL = RELEASE_DIR / "model.joblib"
RELEASE_MANIFEST = RELEASE_DIR / "manifest.json"

VERSION = "2.0.0-rc1"
THRESHOLD = 0.20


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    RELEASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(SOURCE_MODEL)

    if not SOURCE_METRICS.exists():
        raise FileNotFoundError(SOURCE_METRICS)

    with open(
        SOURCE_METRICS,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    if metrics["model"] != "logistic_regression_unweighted":
        raise ValueError("Unexpected candidate model.")

    if not np.isclose(metrics["threshold"], THRESHOLD):
        raise ValueError("Threshold mismatch.")

    # Verify that the model can be loaded.
    model = joblib.load(SOURCE_MODEL)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model does not support probability prediction."
        )

    # Validate the expected input feature names.
    if hasattr(model, "feature_names_in_"):
        actual_features = list(model.feature_names_in_)

        if actual_features != list(ALL_FEATURES):
            raise ValueError(
                "Model input features do not match "
                "the application feature definition."
            )

    # Preserve the exact trained model.
    shutil.copy2(SOURCE_MODEL, RELEASE_MODEL)

    manifest = {
        "model_name": "sentinelml-service-risk",
        "version": VERSION,
        "release_status": "candidate",
        "algorithm": "LogisticRegression",
        "class_weight": None,
        "threshold": THRESHOLD,
        "model_file": "model.joblib",
        "sha256": sha256_file(RELEASE_MODEL),
        "features": list(ALL_FEATURES),
        "evaluation_dataset": (
            "previously_inspected_reserved_set"
        ),
        "evaluation_metrics": metrics,
        "data_origin": "synthetic",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        RELEASE_MANIFEST,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\nSENTINELML RELEASE CANDIDATE")
    print("=" * 55)

    print("Version:", VERSION)
    print("Algorithm:", manifest["algorithm"])
    print("Threshold:", THRESHOLD)
    print("SHA-256:", manifest["sha256"])
    print("Model:", RELEASE_MODEL)
    print("Manifest:", RELEASE_MANIFEST)

    print("\nRelease candidate prepared successfully.")
    print("The existing FastAPI model was not modified.")


if __name__ == "__main__":
    main()