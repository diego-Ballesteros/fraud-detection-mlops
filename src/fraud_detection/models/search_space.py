"""The search space: the algorithm is one more hyperparameter (Reto ML 1).

A trial first draws the algorithm, then that algorithm's hyperparameters, and is scored by
the harness: recall at precision >= 0.99, fold by fold over the temporal splitter, on the
``full`` configuration. Every fold's value is reported as an intermediate value, so a
median pruner can stop a trial that is already losing; a LightGBM that diverges (ERR-014)
shows it in fold 1.

Every bound is declared once, in ``CHOICES``, ``INTS`` and ``FLOATS``. Parameter names
carry the algorithm's prefix, so the branches never share a name.

**The starting points.** The three default configurations cannot all be expressed in this
space: Random Forest's ``max_depth=None`` and ``min_samples_leaf=1``, and LightGBM's
``max_depth=-1`` and ``reg_lambda=0``, fall outside it, and logistic regression's
``max_iter`` is fixed here at 1000 instead of 100. ``STARTING_POINTS`` takes each default
where it fits and the most permissive bound where it does not. The true defaults were
measured apart (EVAL-006 to EVAL-008), so the search is compared against both.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Final

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detection.config import settings
from fraud_detection.models.estimators import ALL_CORES, full_preprocessor, lightgbm_classifier
from fraud_detection.models.evaluation import (
    DECISION_PRECISION,
    EstimatorFactory,
    FoldResult,
    evaluate_fold,
    floor_key,
)

ALGORITHMS: Final[tuple[str, ...]] = ("logistic_regression", "random_forest", "lightgbm")

OBJECTIVE_METRIC: Final[str] = f"recall_at_{floor_key(DECISION_PRECISION)}"
"""What a trial maximizes, averaged over the folds: the decision metric of ADR 0009."""

LR_MAX_ITER: Final[int] = 1000

CHOICES: Final[dict[str, tuple[Any, ...]]] = {
    "algorithm": ALGORITHMS,
    "lr_class_weight": (None, "balanced"),
    "rf_max_features": ("sqrt", "log2", None),
    "rf_class_weight": (None, "balanced", "balanced_subsample"),
}
INTS: Final[dict[str, tuple[int, int, bool]]] = {
    "rf_n_estimators": (50, 200, False),
    "rf_max_depth": (4, 16, False),
    "rf_min_samples_leaf": (20, 500, True),
    "lgbm_n_estimators": (50, 400, False),
    "lgbm_num_leaves": (15, 127, False),
    "lgbm_max_depth": (3, 12, False),
    "lgbm_min_child_samples": (20, 500, True),
}
"""Integer hyperparameters: (low, high, log scale)."""

FLOATS: Final[dict[str, tuple[float, float, bool]]] = {
    "lr_C": (1e-3, 1e2, True),
    "lgbm_learning_rate": (0.01, 0.3, True),
    "lgbm_reg_lambda": (1e-3, 100.0, True),
    "lgbm_scale_pos_weight": (1.0, 1000.0, True),
}
"""Float hyperparameters: (low, high, log scale)."""

STARTING_POINTS: Final[tuple[dict[str, Any], ...]] = (
    {"algorithm": "logistic_regression", "lr_C": 1.0, "lr_class_weight": None},
    {
        "algorithm": "random_forest",
        "rf_n_estimators": 100,
        "rf_max_depth": 16,
        "rf_min_samples_leaf": 20,
        "rf_max_features": "sqrt",
        "rf_class_weight": None,
    },
    {
        "algorithm": "lightgbm",
        "lgbm_n_estimators": 100,
        "lgbm_learning_rate": 0.1,
        "lgbm_num_leaves": 31,
        "lgbm_max_depth": 12,
        "lgbm_min_child_samples": 20,
        "lgbm_reg_lambda": 1e-3,
        "lgbm_scale_pos_weight": 1.0,
    },
)
"""The defaults where they fit the space, the most permissive bound where they do not."""

TrialReport = Callable[[optuna.Trial, str, list[FoldResult], str], None]
"""Called once per trial with the trial, its algorithm, the folds it ran and its state."""


def _choice(trial: optuna.Trial, name: str) -> Any:
    return trial.suggest_categorical(name, list(CHOICES[name]))


def _int(trial: optuna.Trial, name: str) -> int:
    low, high, log = INTS[name]
    return trial.suggest_int(name, low, high, log=log)


def _float(trial: optuna.Trial, name: str) -> float:
    low, high, log = FLOATS[name]
    return trial.suggest_float(name, low, high, log=log)


def suggest_model(trial: optuna.Trial) -> tuple[str, EstimatorFactory]:
    """Draw the algorithm and its hyperparameters.

    Args:
        trial: The Optuna trial.

    Returns:
        The algorithm's name and a factory of the configured, unfitted classifier.
    """
    algorithm: str = _choice(trial, "algorithm")
    if algorithm == "logistic_regression":
        return algorithm, partial(
            LogisticRegression,
            C=_float(trial, "lr_C"),
            class_weight=_choice(trial, "lr_class_weight"),
            max_iter=LR_MAX_ITER,
            random_state=settings.RANDOM_STATE,
        )
    if algorithm == "random_forest":
        return algorithm, partial(
            RandomForestClassifier,
            n_estimators=_int(trial, "rf_n_estimators"),
            max_depth=_int(trial, "rf_max_depth"),
            min_samples_leaf=_int(trial, "rf_min_samples_leaf"),
            max_features=_choice(trial, "rf_max_features"),
            class_weight=_choice(trial, "rf_class_weight"),
            n_jobs=ALL_CORES,
            random_state=settings.RANDOM_STATE,
        )
    return algorithm, partial(
        lightgbm_classifier,
        n_estimators=_int(trial, "lgbm_n_estimators"),
        learning_rate=_float(trial, "lgbm_learning_rate"),
        num_leaves=_int(trial, "lgbm_num_leaves"),
        max_depth=_int(trial, "lgbm_max_depth"),
        min_child_samples=_int(trial, "lgbm_min_child_samples"),
        reg_lambda=_float(trial, "lgbm_reg_lambda"),
        scale_pos_weight=_float(trial, "lgbm_scale_pos_weight"),
    )


def make_objective(
    features: pd.DataFrame,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    on_trial: TrialReport | None = None,
) -> Callable[[optuna.Trial], float]:
    """Build the objective: the decision metric, averaged over the folds of ``splits``.

    Each fold runs through ``evaluation.evaluate_fold``, the harness's own path, with the
    ``full`` preprocessor. After each fold its value is reported as the intermediate value
    of that step, and the trial is pruned if the pruner says so.

    Args:
        features: Every reference row, without the label.
        labels: Every row's 0/1 label.
        splits: Training and validation positions of each fold, computed once.
        on_trial: Called once per trial, finished or pruned, to log it.

    Returns:
        The objective function for ``study.optimize``.
    """

    def objective(trial: optuna.Trial) -> float:
        algorithm, model = suggest_model(trial)
        trial.set_user_attr("algorithm", algorithm)
        folds: list[FoldResult] = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            result = evaluate_fold(
                features,
                labels,
                train_index,
                validation_index,
                full_preprocessor,
                model,
                fold=fold,
            )
            folds.append(result)
            trial.report(result.metrics[OBJECTIVE_METRIC], step=fold)
            if trial.should_prune():
                if on_trial is not None:
                    on_trial(trial, algorithm, folds, "pruned")
                raise optuna.TrialPruned()
        if on_trial is not None:
            on_trial(trial, algorithm, folds, "complete")
        return float(np.mean([fold.metrics[OBJECTIVE_METRIC] for fold in folds]))

    return objective
