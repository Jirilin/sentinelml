from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from app.ml.features import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/service_requests.csv"

MODEL_DIR = ROOT / "reports/model_comparison"

OUTPUT_DIR = MODEL_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def main():

    df = pd.read_csv(DATA_PATH)

    X = df[ALL_FEATURES]
    y = df["sla_breached"].astype(int)

    # Reproduce the development split.
    X_dev, _, y_dev, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    _, X_val, _, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=SEED,
    )

    models = {
        "Logistic Regression": (
            MODEL_DIR / "logistic_regression.joblib"
        ),
        "Random Forest": (
            MODEL_DIR / "random_forest.joblib"
        ),
        "Gradient Boosting": (
            MODEL_DIR / "hist_gradient_boosting.joblib"
        ),
    }

    results = []

    plt.figure(figsize=(10, 7))

    # Perfect calibration reference.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    for name, path in models.items():

        model = joblib.load(path)

        probabilities = model.predict_proba(
            X_val
        )[:, 1]

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"{name} produced non-finite probabilities"
            )

        fraction_positive, mean_predicted = (
            calibration_curve(
                y_val,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=name,
        )

        results.append({
            "model": name,
            "brier_score": brier_score_loss(
                y_val,
                probabilities,
            ),
            "log_loss": log_loss(
                y_val,
                probabilities,
            ),
            "roc_auc": roc_auc_score(
                y_val,
                probabilities,
            ),
            "average_precision": (
                average_precision_score(
                    y_val,
                    probabilities,
                )
            ),
        })

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed SLA Breach Frequency")

    plt.title(
        "SentinelML — Probability Calibration"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "calibration_curve.png",
        dpi=200,
    )

    plt.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "calibration_metrics.csv",
        index=False,
    )

    print("\nPROBABILITY CALIBRATION RESULTS")
    print("=" * 70)

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print("\nCalibration reports saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()