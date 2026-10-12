from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from sklearn.model_selection import train_test_split

from app.ml.features import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/raw/service_requests.csv"
MODEL_PATH = ROOT / "models/service_risk.joblib"
OUTPUT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df[ALL_FEATURES]
    y = df["sla_breached"]

    # Reproduce original train/test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\nMODEL EVALUATION")
    print("=" * 50)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    # Save metrics
    with open(
        OUTPUT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Breach", "SLA Breach"],
    )

    plt.title("SentinelML — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
    )

    plt.title("SentinelML — ROC Curve")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=200,
    )
    plt.close()

    # Precision-recall curve
    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_prob,
    )

    plt.title("SentinelML — Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "precision_recall_curve.png",
        dpi=200,
    )
    plt.close()

    print("\nEvaluation reports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()