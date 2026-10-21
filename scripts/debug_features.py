import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from scripts.train_compare import (
    DATA,
    SEED,
    make_pipeline,
)

from app.ml.features import ALL_FEATURES

from sklearn.linear_model import LogisticRegression


def main():

    df = pd.read_csv(DATA)

    X = df[ALL_FEATURES]
    y = df["sla_breached"].astype(int)

    X_dev, _, y_dev, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    X_train, _, _, _ = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=SEED,
    )

    pipeline = make_pipeline(
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )
    )

    preprocessor = pipeline.named_steps["preprocess"]

    X_transformed = preprocessor.fit_transform(X_train)

    X_transformed = np.asarray(
        X_transformed,
        dtype=np.float64,
    )

    print("\nPREPROCESSED FEATURE MATRIX")
    print("=" * 50)

    print("Shape:", X_transformed.shape)

    print(
        "Contains NaN:",
        np.isnan(X_transformed).any(),
    )

    print(
        "Contains infinity:",
        np.isinf(X_transformed).any(),
    )

    finite_values = X_transformed[
        np.isfinite(X_transformed)
    ]

    if finite_values.size:
        print(
            "Minimum finite value:",
            finite_values.min(),
        )

        print(
            "Maximum finite value:",
            finite_values.max(),
        )

    print(
        "Matrix dtype:",
        X_transformed.dtype,
    )

    print("\nDEBUGGING COMPLETE")


if __name__ == "__main__":
    main()