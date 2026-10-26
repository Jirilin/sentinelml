from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
)

from app.ml.features import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/service_requests.csv"

MODEL_PATH = (
    ROOT
    / "reports/model_comparison"
    / "logistic_regression_unweighted.joblib"
)

OUTPUT_DIR = ROOT / "reports/candidate_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
THRESHOLD = 0.20


def main():

    print("\nSENTINELML — CANDIDATE EVALUATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    X = df[ALL_FEATURES]
    y = df["sla_breached"].astype(int)

    # Reconstruct the original 80/20 split.
    # The second portion is the previously inspected
    # reserved dataset.
    _, X_reserved, _, y_reserved = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    print("Reserved observations:", len(X_reserved))

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(
        X_reserved
    )[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model produced non-finite probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    metrics = {
        "model": "logistic_regression_unweighted",
        "threshold": THRESHOLD,
        "dataset": "previously_inspected_reserved_set",
        "observations": int(len(y_reserved)),
        "accuracy": float(
            accuracy_score(y_reserved, predictions)
        ),
        "precision": float(
            precision_score(
                y_reserved,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_reserved,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_reserved,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_reserved,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_reserved,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_reserved,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_reserved,
                probabilities,
                labels=[0, 1],
            )
        ),
    }

    # Constant-probability baseline.
    # Use the training/development prevalence rather
    # than learning the baseline from reserved labels.
    X_dev, _, y_dev, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    baseline_probability = float(y_dev.mean())

    baseline_probs = np.full(
        len(y_reserved),
        baseline_probability,
    )

    metrics["baseline_brier_score"] = float(
        brier_score_loss(
            y_reserved,
            baseline_probs,
        )
    )

    metrics["baseline_log_loss"] = float(
        log_loss(
            y_reserved,
            baseline_probs,
            labels=[0, 1],
        )
    )

    cm = confusion_matrix(
        y_reserved,
        predictions,
        labels=[0, 1],
    )

    print("\nEVALUATION METRICS")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nCONFUSION MATRIX")
    print(cm)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_reserved,
            predictions,
            zero_division=0,
        )
    )

    with open(
        OUTPUT_DIR / "candidate_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame(
        cm,
        index=["actual_no_breach", "actual_breach"],
        columns=[
            "predicted_no_breach",
            "predicted_breach",
        ],
    ).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print("\nReports saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()