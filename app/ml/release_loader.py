from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np

from app.ml.features import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[2]

RELEASES_DIR = ROOT / "models" / "releases"

SUPPORTED_RELEASES = {
    "v2-rc1": "sentinelml-v2-rc1",
}


@dataclass
class LoadedRelease:
    model: object
    version: str
    threshold: float
    features: list


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_release(version: str) -> LoadedRelease:

    if version not in SUPPORTED_RELEASES:
        raise ValueError(
            f"Unsupported release version: {version}"
        )

    release_dir = (
        RELEASES_DIR / SUPPORTED_RELEASES[version]
    )

    manifest_path = release_dir / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    model_filename = manifest["model_file"]

    if Path(model_filename).name != model_filename:
        raise ValueError(
            "Model filename must not contain a path."
        )

    model_path = release_dir / model_filename

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if sha256_file(model_path) != manifest["sha256"]:
        raise ValueError(
            "Release model checksum mismatch."
        )

    if manifest["version"] != "2.0.0-rc1":
        raise ValueError(
            "Unexpected release version."
        )

    threshold = float(manifest["threshold"])

    if not np.isfinite(threshold):
        raise ValueError(
            "Threshold must be finite."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if manifest["features"] != list(ALL_FEATURES):
        raise ValueError(
            "Release feature schema mismatch."
        )

    # Load only trusted, locally generated artefacts.
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Release model does not support predict_proba."
        )

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != list(ALL_FEATURES):
            raise ValueError(
                "Model feature names do not match."
            )

    return LoadedRelease(
        model=model,
        version=manifest["version"],
        threshold=threshold,
        features=list(ALL_FEATURES),
    )