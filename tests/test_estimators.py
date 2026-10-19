"""Tests of fraud_detection.models.estimators: the factories of phase 2A."""

import subprocess
import sys

import pytest

from fraud_detection import settings
from fraud_detection.models import estimators
from fraud_detection.models.baseline import SignatureRule
from fraud_detection.models.estimators import MODEL_SPECS


def test_the_five_models_in_report_order() -> None:
    assert list(MODEL_SPECS) == [
        "majority_class",
        "signature_rule",
        "logistic_regression",
        "random_forest",
        "lightgbm",
    ]
    learners = [name for name, spec in MODEL_SPECS.items() if spec.learns_from_labels]
    assert learners == ["logistic_regression", "random_forest", "lightgbm"]


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_every_factory_builds_a_new_object(name: str) -> None:
    spec = MODEL_SPECS[name]
    assert spec.model() is not spec.model()
    assert spec.preprocessor() is not spec.preprocessor()


def test_seeds_come_from_settings_and_every_core_is_used() -> None:
    assert estimators.logistic_regression().random_state == settings.RANDOM_STATE
    forest = estimators.random_forest()
    assert (forest.random_state, forest.n_jobs) == (settings.RANDOM_STATE, -1)
    booster = estimators.lightgbm_classifier()
    assert (booster.random_state, booster.n_jobs) == (settings.RANDOM_STATE, -1)


def test_hyperparameters_are_the_library_defaults() -> None:
    # Only the seed, n_jobs and LightGBM's log level are set.
    forest = estimators.random_forest().get_params()
    assert (forest["n_estimators"], forest["max_depth"], forest["class_weight"]) == (
        100,
        None,
        None,
    )
    logistic = estimators.logistic_regression().get_params()
    assert (logistic["C"], logistic["solver"], logistic["max_iter"]) == (1.0, "lbfgs", 100)
    booster = estimators.lightgbm_classifier().get_params()
    assert (booster["n_estimators"], booster["num_leaves"], booster["learning_rate"]) == (
        100,
        31,
        0.1,
    )
    assert estimators.majority_class().strategy == "most_frequent"
    assert isinstance(estimators.signature_rule(), SignatureRule)


def test_the_lightgbm_factory_works_in_a_clean_interpreter() -> None:
    # On a Windows machine without the VC++ Redistributable this fails if lightgbm is
    # imported before scikit-learn (module docstring of estimators).
    code = (
        "from fraud_detection.models.estimators import lightgbm_classifier\n"
        "print(type(lightgbm_classifier()).__name__)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "LGBMClassifier"
