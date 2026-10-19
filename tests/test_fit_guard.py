"""Tests of fraud_detection.models.fit_guard: child processes, the time limit, errors.

The loader and the models live at module level because the child is spawned and finds
them by name. Each test starts real processes, so the suite keeps them few.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from fraud_detection.models.estimators import ModelSpec, raw_rows
from fraud_detection.models.evaluation import ExpandingStepSplit, cross_validate
from fraud_detection.models.fit_guard import run_guarded
from tests.test_evaluation import synthetic_frame

LIMIT_SECONDS = 2.0


def load_small_frame() -> pd.DataFrame:
    return synthetic_frame()


class SleepyClassifier(ClassifierMixin, BaseEstimator):
    """A fit that never finishes in time."""

    def fit(self, X: Any, y: np.ndarray) -> "SleepyClassifier":  # noqa: N803
        time.sleep(600)
        return self


class BrokenClassifier(ClassifierMixin, BaseEstimator):
    """A fit that raises."""

    def fit(self, X: Any, y: np.ndarray) -> "BrokenClassifier":  # noqa: N803
        raise RuntimeError("fit exploded on purpose")


def dummy() -> DummyClassifier:
    return DummyClassifier()


QUICK = ModelSpec("quick", raw_rows, dummy, True)
SLEEPY = ModelSpec("sleepy", raw_rows, SleepyClassifier, True)
BROKEN = ModelSpec("broken", raw_rows, BrokenClassifier, True)


def test_a_guarded_run_equals_the_in_process_harness() -> None:
    guarded = run_guarded(load_small_frame, ExpandingStepSplit(), QUICK)
    direct = cross_validate(load_small_frame(), ExpandingStepSplit(), QUICK.preprocessor, dummy)
    assert guarded.complete
    assert [fold.metrics for fold in guarded.folds] == [fold.metrics for fold in direct]
    assert [fold.train_rows for fold in guarded.folds] == [fold.train_rows for fold in direct]


def test_a_fit_over_the_limit_is_aborted_and_timed() -> None:
    started = time.monotonic()
    run = run_guarded(load_small_frame, ExpandingStepSplit(), SLEEPY, fit_time_limit=LIMIT_SECONDS)
    elapsed = time.monotonic() - started
    assert not run.complete
    assert run.aborted_fold == 1
    assert run.folds == ()
    assert run.aborted_after_seconds is not None
    assert LIMIT_SECONDS <= run.aborted_after_seconds < LIMIT_SECONDS + 5
    # The sleep is 600 s: the child was killed, not waited for.
    assert elapsed < 120


def test_an_error_in_the_child_is_reported_with_its_traceback() -> None:
    run = run_guarded(load_small_frame, ExpandingStepSplit(), BROKEN)
    assert not run.complete
    assert run.aborted_fold is None
    assert run.error is not None
    assert "RuntimeError: fit exploded on purpose" in run.error
