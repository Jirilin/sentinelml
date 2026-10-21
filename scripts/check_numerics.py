import warnings

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


def main():
    print("\nNUMERICAL ENVIRONMENT")
    print("=" * 55)

    print("NumPy:", np.__version__)
    print("BLAS configuration:")
    print(threadpool_info())

    rng = np.random.default_rng(42)

    X = rng.normal(size=(7200, 14)).astype(np.float64)
    weights = rng.normal(size=14).astype(np.float64)

    print("\nMATRIX MULTIPLICATION TEST")
    print("=" * 55)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)

        with threadpool_limits(limits=1):
            result = X @ weights

    print("Result shape:", result.shape)
    print("All finite:", np.isfinite(result).all())
    print("Runtime warnings:", len(caught))

    for warning in caught:
        print(type(warning.message).__name__, warning.message)

    print("\nDIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()