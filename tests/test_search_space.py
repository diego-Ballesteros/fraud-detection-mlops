"""Tests of fraud_detection.models.search_space: the space, the starting points, the objective."""

from typing import Any

import numpy as np
import optuna
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detection import settings
from fraud_detection.models import search_space
from fraud_detection.models.evaluation import (
    ExpandingStepSplit,
    FoldResult,
    cross_validate,
    features_and_labels,
)
from fraud_detection.models.search_space import (
    CHOICES,
    FLOATS,
    INTS,
    OBJECTIVE_METRIC,
    STARTING_POINTS,
    make_objective,
    suggest_model,
)
from tests.test_evaluation import synthetic_frame


def _inside(name: str, value: Any) -> bool:
    if name in CHOICES:
        return value in CHOICES[name]
    low, high, _ = INTS[name] if name in INTS else FLOATS[name]
    return low <= value <= high


def test_every_parameter_is_declared_once() -> None:
    names = [*CHOICES, *INTS, *FLOATS]
    assert len(names) == len(set(names))


def test_the_starting_points_are_inside_the_space_one_per_algorithm() -> None:
    assert [point["algorithm"] for point in STARTING_POINTS] == list(search_space.ALGORITHMS)
    for point in STARTING_POINTS:
        outside = {name: value for name, value in point.items() if not _inside(name, value)}
        assert outside == {}


def test_a_value_outside_the_space_is_caught() -> None:
    # Control: the check above can fail.
    assert not _inside("rf_max_depth", 17)
    assert not _inside("lr_class_weight", "weird")
    assert not _inside("lgbm_reg_lambda", 0.0)


@pytest.mark.parametrize("point", STARTING_POINTS, ids=lambda p: p["algorithm"])
def test_each_starting_point_builds_its_model_with_the_project_seed(point: dict[str, Any]) -> None:
    algorithm, factory = suggest_model(optuna.trial.FixedTrial(point))
    model = factory()
    assert algorithm == point["algorithm"]
    assert model.random_state == settings.RANDOM_STATE
    params = model.get_params()
    if isinstance(model, LogisticRegression):
        assert (params["C"], params["max_iter"]) == (1.0, search_space.LR_MAX_ITER)
    elif isinstance(model, RandomForestClassifier):
        assert (params["max_depth"], params["min_samples_leaf"], params["n_jobs"]) == (16, 20, -1)
    else:
        assert (params["max_depth"], params["num_leaves"], params["n_jobs"]) == (12, 31, -1)
        assert params["reg_lambda"] == pytest.approx(1e-3)


def test_the_objective_is_the_fold_mean_of_the_harness_metric() -> None:
    frame = synthetic_frame()
    features, labels = features_and_labels(frame)
    splits = list(ExpandingStepSplit().split(features, labels))
    logged: list[tuple[str, int, str]] = []

    def record(trial: optuna.Trial, algorithm: str, folds: list[FoldResult], state: str) -> None:
        logged.append((algorithm, len(folds), state))

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(STARTING_POINTS[0])
    study.optimize(make_objective(features, labels, splits, record), n_trials=1)
    trial = study.trials[0]
    _, factory = suggest_model(optuna.trial.FixedTrial(STARTING_POINTS[0]))
    expected = cross_validate(frame, ExpandingStepSplit(), search_space.full_preprocessor, factory)
    per_fold = [fold.metrics[OBJECTIVE_METRIC] for fold in expected]
    assert trial.value == pytest.approx(float(np.mean(per_fold)))
    assert [trial.intermediate_values[step] for step in range(1, 6)] == pytest.approx(per_fold)
    assert logged == [("logistic_regression", 5, "complete")]


def test_a_pruned_trial_stops_after_its_fold_and_is_logged_as_pruned() -> None:
    features, labels = features_and_labels(synthetic_frame())
    splits = list(ExpandingStepSplit().split(features, labels))
    logged: list[tuple[str, int, str]] = []

    def record(trial: optuna.Trial, algorithm: str, folds: list[FoldResult], state: str) -> None:
        logged.append((algorithm, len(folds), state))

    # A threshold no recall can reach prunes every trial at its first fold.
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=2.0)
    )
    study.enqueue_trial(STARTING_POINTS[0])
    study.optimize(make_objective(features, labels, splits, record), n_trials=1)
    assert study.trials[0].state is optuna.trial.TrialState.PRUNED
    assert logged == [("logistic_regression", 1, "pruned")]
