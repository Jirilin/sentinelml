from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import train_test_split

from app.ml.features import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/service_requests.csv"
MODEL_PATH = ROOT / "models/service_risk.joblib"
OUTPUT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df[ALL_FEATURES]
    y = df["sla_breached"]

    # Reproduce the original test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.05)

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    results_df = pd.DataFrame(results)

    print("\nTHRESHOLD ANALYSIS")
    print("=" * 60)

    print(results_df.to_string(index=False))

    # Diagnostic only: do not select a production
    # threshold using the final test set.
    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\nHIGHEST OBSERVED TEST-SET F1")
    print("=" * 60)

    print(
        f"Threshold: {best_row['threshold']:.2f}"
    )

    print(
        f"Precision: {best_row['precision']:.4f}"
    )

    print(
        f"Recall: {best_row['recall']:.4f}"
    )

    print(
        f"F1: {best_row['f1']:.4f}"
    )

    # Save results
    results_df.to_csv(
        OUTPUT_DIR / "threshold_analysis.csv",
        index=False,
    )

    # Plot results
    plt.figure(figsize=(10, 6))

    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        label="Precision",
        marker="o",
    )

    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        label="Recall",
        marker="s",
    )

    plt.plot(
        results_df["threshold"],
        results_df["f1"],
        label="F1",
        marker="^",
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")

    plt.title(
        "SentinelML — Threshold Analysis"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "threshold_analysis.png",
        dpi=200,
    )

    plt.close()

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()