import pytest

from app.ml.release_loader import load_release


def test_v2_release_loads():

    release = load_release("v2-rc1")

    assert release.version == "2.0.0-rc1"

    assert release.threshold == pytest.approx(0.20)

    assert hasattr(
        release.model,
        "predict_proba",
    )


def test_unsupported_release_rejected():

    with pytest.raises(ValueError):
        load_release("v99")