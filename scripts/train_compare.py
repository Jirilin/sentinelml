
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from app.ml.features import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/service_requests.csv"
OUT = ROOT / "reports/model_comparison"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def make_pipeline(classifier):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def main():
    if not DATA.exists():
        raise SystemExit("Dataset missing: run python scripts/generate_data.py")
    df = pd.read_csv(DATA)
    X, y = df[ALL_FEATURES], df["sla_breached"].astype(int)
    # This is a development split, NOT a fresh untouched benchmark.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED
    )
    print(f"Train={len(X_train)} Validation={len(X_val)} Reserved={len(X_test)}")
    print("Reserved set is not scored here; prior v1 experiments already inspected it.")

    candidates = {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),

        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=20,
            class_weight="balanced",
            random_state=SEED,
        ),
}
    thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
    rows = []
    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)
        started = time.perf_counter()
        print(f"\nTraining model: {name}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)

            pipeline.fit(X_train, y_train)

        print(f"Runtime warnings during fitting: {len(caught)}")

        for warning in caught:
            print(
                f"{warning.category.__name__}: "
                f"{warning.message}"
    )
        fit_seconds = time.perf_counter() - started
        prob = pipeline.predict_proba(X_val)[:, 1]
        # Select threshold on validation data only.
        options = [(float(t), metrics(y_val, prob, float(t))) for t in thresholds]
        threshold, scores = max(options, key=lambda pair: (pair[1]["f1"], -pair[0]))
        row = {"model": name, "threshold": threshold, "fit_seconds": fit_seconds, **scores}
        rows.append(row)
        joblib.dump(pipeline, OUT / f"{name}.joblib")
        print(f"{name}: F1={scores['f1']:.4f}, AP={scores['average_precision']:.4f}, threshold={threshold:.2f}")

    results = pd.DataFrame(rows).sort_values(
        ["average_precision", "f1"], ascending=False
    )
    results.to_csv(OUT / "validation_results.csv", index=False)
    champion = results.iloc[0].to_dict()
    (OUT / "candidate_selection.json").write_text(json.dumps({
        "selection_rule": "highest validation average_precision; F1 tie-breaker",
        "note": "Development-only candidate. No deployment or final test evaluation performed.",
        "selected": champion,
        "splits": {"train": len(X_train), "validation": len(X_val), "reserved": len(X_test)},
    }, indent=2))
    print("\nVALIDATION RESULTS (not final test performance)")
    print(results.round(4).to_string(index=False))
    print(f"\nCandidate: {champion['model']}; API model remains unchanged.")
    print(f"Saved to: {OUT}")


if __name__ == "__main__":
    main()
