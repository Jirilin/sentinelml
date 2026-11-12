from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from app.ml.features import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[1]

RELEASE_DIR = (
    ROOT / "models/releases/sentinelml-v2-rc1"
)

MANIFEST_PATH = RELEASE_DIR / "manifest.json"


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
    print("\nSENTINELML — RELEASE VERIFICATION")
    print("=" * 60)

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    model_path = RELEASE_DIR / manifest["model_file"]

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    # Verify file integrity before deserialisation.
    actual_hash = sha256_file(model_path)

    if actual_hash != manifest["sha256"]:
        raise ValueError(
            "Model checksum does not match the manifest."
        )

    print("Checksum: PASS")

    if manifest["version"] != "2.0.0-rc1":
        raise ValueError("Unexpected model version.")

    threshold = float(manifest["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid decision threshold.")

    if manifest["features"] != list(ALL_FEATURES):
        raise ValueError(
            "Feature schema does not match the application."
        )

    print("Manifest: PASS")

    # Only load trusted, locally generated model artefacts.
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model does not support predict_proba."
        )

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != list(ALL_FEATURES):
            raise ValueError(
                "Model feature names do not match."
            )

    print("Model loading: PASS")

    # Use an actual input row from the project dataset.
    data_path = ROOT / "data/raw/service_requests.csv"

    df = pd.read_csv(data_path)

    example = df.loc[:, ALL_FEATURES].head(1).copy()

    probabilities = np.asarray(
        model.predict_proba(example)
    )

    if probabilities.shape != (1, 2):
        raise ValueError(
            "Unexpected probability output shape."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite prediction probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
    ):
        raise ValueError(
            "Class probabilities do not sum to 1."
        )

    # Explicitly check the positive-class mapping.
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Positive class 1 is missing."
        )

    positive_index = classes.index(1)

    risk_probability = float(
        probabilities[0, positive_index]
    )

    prediction = (
        "sla_risk"
        if risk_probability >= threshold
        else "no_sla_risk"
    )

    print("Prediction: PASS")

    print("\nRELEASE DETAILS")
    print("=" * 60)

    print("Version:", manifest["version"])
    print("Threshold:", threshold)
    print("Risk probability:", round(risk_probability, 4))
    print("Prediction:", prediction)

    print("\nALL RELEASE CHECKS PASSED")


if __name__ == "__main__":
    main()