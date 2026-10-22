"""Tests of fraud_detection.models.estimators: the factories of phase 2A."""

import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraud_detection import settings
from fraud_detection.models import estimators
from fraud_detection.models.baseline import SignatureRule
from fraud_detection.models.estimators import MODEL_SPECS, UniformRandomScores, splits_on
from fraud_detection.models.evaluation import derived_seeds


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


def test_the_ablation_preprocessor_drops_the_signature_and_nothing_else() -> None:
    assert estimators.no_signature_preprocessor() is not estimators.no_signature_preprocessor()
    rows = pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["TRANSFER", "CASH_OUT"],
            "amount": [100.0, 50.0],
            "oldbalanceOrg": [100.0, 80.0],
            "oldbalanceDest": [0.0, 10.0],
        }
    )
    full = set(estimators.full_preprocessor().fit(rows).get_feature_names_out())
    ablated = set(estimators.no_signature_preprocessor().fit(rows).get_feature_names_out())
    # Exactly the two columns that carry the signature go, the flag and the balance it is
    # computed from; amount stays, because the flag needs both.
    assert full - ablated == {"amount_equals_origin_balance", "oldbalanceOrg"}
    assert ablated < full


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


WINDOWS_ONLY = (
    "Windows only: it catches lightgbm imported before scikit-learn, which fails to load "
    "lib_lightgbm.dll on a Windows machine without the VC++ Redistributable. On Linux the "
    "import works in any order, so there this test cannot fail and its pass means nothing."
)


@pytest.mark.skipif(sys.platform != "win32", reason=WINDOWS_ONLY)
def test_the_lightgbm_factory_works_in_a_clean_interpreter() -> None:
    # The factory must build the model in an interpreter where nothing was imported first
    # (module docstring of estimators).
    code = (
        "from fraud_detection.models.estimators import lightgbm_classifier\n"
        "print(type(lightgbm_classifier()).__name__)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "LGBMClassifier"


def test_the_random_classifier_draws_uniform_scores_seeded_and_learns_nothing() -> None:
    rows = pd.DataFrame({"amount": np.arange(1000.0)})
    first, second = derived_seeds(2)
    model = UniformRandomScores(random_state=first)
    state = dict(vars(model))
    model.fit(rows, np.zeros(1000))
    assert vars(model) == state
    scores = model.predict_proba(rows)[:, 1]
    same = UniformRandomScores(random_state=first).predict_proba(rows)[:, 1]
    other = UniformRandomScores(random_state=second).predict_proba(rows)[:, 1]
    assert np.array_equal(scores, same)
    assert not np.array_equal(scores, other)
    assert scores.min() >= 0.0
    assert scores.max() < 1.0
    # Uniform on [0, 1): the mean of 1000 draws is within 0.05 of 1/2.
    assert abs(scores.mean() - 0.5) < 0.05
    assert model.classes_.tolist() == [0, 1]


def test_the_random_classifier_spec_survives_the_trip_to_a_child_process() -> None:
    spec = estimators.random_classifier_spec(derived_seeds(1)[0])
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.name == "random_classifier"
    assert restored.model().random_state == settings.RANDOM_STATE
    assert not restored.learns_from_labels
    assert "random_classifier" not in MODEL_SPECS


def _fitted(model: object) -> Pipeline:
    from tests.test_evaluation import synthetic_frame

    frame = synthetic_frame()
    pipeline = Pipeline([("preprocess", estimators.full_preprocessor()), ("model", model)])
    return pipeline.fit(frame.drop(columns=["isFraud"]), frame["isFraud"])


def test_splits_on_counts_tree_splits_and_says_none_for_a_linear_model() -> None:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    booster = _fitted(
        LGBMClassifier(
            n_estimators=10, min_child_samples=5, verbose=-1, random_state=settings.RANDOM_STATE
        )
    )
    forest = _fitted(RandomForestClassifier(n_estimators=5, random_state=settings.RANDOM_STATE))
    # The synthetic data has only TRANSFER and PAYMENT, so the DEBIT column is constant and
    # no tree can split on it: the negative control. Every positive is a TRANSFER and every
    # negative a PAYMENT, so the trees separate the classes on the type columns.
    for model in (booster, forest):
        assert splits_on(model, "type_DEBIT") == 0
        assert splits_on(model, "type_TRANSFER") + splits_on(model, "type_PAYMENT") > 0
    assert splits_on(_fitted(LogisticRegression()), "amount") is None
    with pytest.raises(ValueError, match="not among the features"):
        splits_on(booster, "step")
