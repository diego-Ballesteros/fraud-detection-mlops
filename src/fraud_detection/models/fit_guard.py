"""Run one cross-validation in a child process, aborting any single fit over a time limit.

A scikit-learn or LightGBM fit cannot be interrupted from inside the process that runs
it, so the cross-validation runs in a child. The child reports when each fit starts and
finishes; the parent holds the clock and terminates the child when a fit exceeds the
limit. The folds finished before the abort are kept and reported.

The child calls ``evaluation.cross_validate``, the same function the in-process harness
uses: a guarded run is not a separate path. It also loads its own data, so the parent
holds no copy of the rows while the child fits.
"""

import multiprocessing
import queue
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from fraud_detection.models.estimators import ModelSpec
from fraud_detection.models.evaluation import (
    FIT_FINISHED,
    FIT_STARTED,
    FOLD_DONE,
    FoldEvent,
    FoldResult,
    Splitter,
    cross_validate,
)

FIT_TIME_LIMIT_SECONDS: Final[float] = 20 * 60
"""A single fit longer than this is aborted, and the model is reported as such."""

_DONE: Final[str] = "done"
_ERROR: Final[str] = "error"


@dataclass(frozen=True)
class GuardedRun:
    """What a guarded cross-validation produced.

    Attributes:
        folds: Results of the folds that finished, in order.
        aborted_fold: The fold whose fit exceeded the limit, or None.
        aborted_after_seconds: How long that fit had run when it was aborted.
        error: Traceback of an exception in the child, or a note on how it died.
        wall_seconds: Wall time of the whole run, child start-up and data loading included.
    """

    folds: tuple[FoldResult, ...]
    aborted_fold: int | None
    aborted_after_seconds: float | None
    error: str | None
    wall_seconds: float

    @property
    def complete(self) -> bool:
        """Whether every fold finished without abort or error."""
        return self.aborted_fold is None and self.error is None


def _child(
    messages: Any,
    load_frame: Callable[[], pd.DataFrame],
    splitter: Splitter,
    spec: ModelSpec,
    shuffle_training_labels: bool,
) -> None:
    """Load the data, cross-validate, and report every event to the parent."""
    try:
        cross_validate(
            load_frame(),
            splitter,
            spec.preprocessor,
            spec.model,
            shuffle_training_labels=shuffle_training_labels,
            on_event=messages.put,
        )
        messages.put((_DONE, None))
    except BaseException:  # the parent must hear about any death, not only Exception
        messages.put((_ERROR, traceback.format_exc()))


def run_guarded(
    load_frame: Callable[[], pd.DataFrame],
    splitter: Splitter,
    spec: ModelSpec,
    *,
    shuffle_training_labels: bool = False,
    fit_time_limit: float = FIT_TIME_LIMIT_SECONDS,
    poll_seconds: float = 1.0,
) -> GuardedRun:
    """Cross-validate ``spec`` in a child process, aborting a fit that runs too long.

    Args:
        load_frame: Module-level function that returns the rows; the child calls it.
        splitter: Picklable splitter.
        spec: The model and its preprocessor, as factories.
        shuffle_training_labels: Passed to ``cross_validate``.
        fit_time_limit: Seconds a single fit may run.
        poll_seconds: How often the parent checks the clock and the child.

    Returns:
        The finished folds and, if it happened, the abort or the error.
    """
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    child = context.Process(
        target=_child,
        args=(messages, load_frame, splitter, spec, shuffle_training_labels),
        daemon=True,
    )
    started = time.monotonic()
    child.start()
    folds: list[FoldResult] = []
    fit_started_at: float | None = None
    current_fold = 0

    def finish(
        aborted_fold: int | None = None,
        aborted_after: float | None = None,
        error: str | None = None,
    ) -> GuardedRun:
        child.join(timeout=30)
        if child.is_alive():
            child.kill()
            child.join()
        return GuardedRun(
            folds=tuple(folds),
            aborted_fold=aborted_fold,
            aborted_after_seconds=aborted_after,
            error=error,
            wall_seconds=time.monotonic() - started,
        )

    while True:
        try:
            message = messages.get(timeout=poll_seconds)
        except queue.Empty:
            now = time.monotonic()
            if fit_started_at is not None and now - fit_started_at > fit_time_limit:
                child.terminate()
                return finish(aborted_fold=current_fold, aborted_after=now - fit_started_at)
            if not child.is_alive():
                # A message put just before dying may still be in flight.
                try:
                    message = messages.get(timeout=poll_seconds)
                except queue.Empty:
                    return finish(error=f"Child exited with code {child.exitcode}, no report")
            else:
                continue
        if isinstance(message, FoldEvent):
            if message.kind == FIT_STARTED:
                current_fold, fit_started_at = message.fold, time.monotonic()
            elif message.kind == FIT_FINISHED:
                fit_started_at = None
            elif message.kind == FOLD_DONE and message.result is not None:
                folds.append(message.result)
            continue
        kind, payload = message
        if kind == _DONE:
            return finish()
        return finish(error=str(payload))
