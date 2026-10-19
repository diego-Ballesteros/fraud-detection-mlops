"""Smoke tests for the project scaffolding."""

import fraud_detection
from fraud_detection import settings


def test_package_imports() -> None:
    assert fraud_detection.__version__
    assert fraud_detection.settings is settings


def test_random_state_is_42() -> None:
    assert settings.RANDOM_STATE == 42


def test_data_directories_exist() -> None:
    data_dirs = [settings.RAW_DATA_DIR, settings.PROCESSED_DATA_DIR]
    missing = [path for path in data_dirs if not path.is_dir()]
    assert not missing, f"Data directories missing on disk: {missing}"


def test_temporal_cut_is_fixed_and_contiguous() -> None:
    # Contiguity guards against desynced constants; the literals against drifting from the spec.
    assert settings.PRODUCTION_FIRST_STEP == settings.REFERENCE_LAST_STEP + 1
    assert settings.REFERENCE_LAST_STEP == 400
    assert settings.PRODUCTION_FIRST_STEP == 401
