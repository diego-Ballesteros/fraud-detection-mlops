"""The hyperparameter search with the algorithm as a hyperparameter (Reto ML 1).

Run from the repository root::

    uv run python scripts/run_training.py

The budget is wall time, not trials, because the families differ up to 20 times in cost:
``TIMEOUT_SECONDS`` or ``MAX_TRIALS``, whichever comes first. The study lives in a SQLite
file at the project root (``optuna.db``, gitignored), so a cut leaves it resumable: running
the script again continues the same study, with the trials that remain. Optuna checks the
clock between trials, so the last trial can run past the timeout.

The first three trials are ``search_space.STARTING_POINTS``, the default configurations as
close as the space allows, so the result says whether the search beat its starting point.
Every trial is one nested MLflow run under the search's run, with its hyperparameters and
its per-fold metrics. Only the reference period is read; the splitter is the temporal one.
"""

import sys
import time
from collections import Counter
from typing import Final

import mlflow
import numpy as np
import optuna

from fraud_detection.config import settings
from fraud_detection.models.estimators import EVALUATED_CONFIGURATION
from fraud_detection.models.evaluation import (
    TEMPORAL,
    ExpandingStepSplit,
    FoldResult,
    features_and_labels,
    load_evaluation_frame,
)
from fraud_detection.models.fit_guard import GuardedRun
from fraud_detection.models.search_space import (
    OBJECTIVE_METRIC,
    STARTING_POINTS,
    make_objective,
)
from fraud_detection.models.tracking import base_tags, log_guarded_run, reference_identity
from fraud_detection.tracking.mlflow_setup import configure_with_local_fallback

STUDY_NAME: Final[str] = "search_full_temporal"
STORAGE_URI: Final[str] = f"sqlite:///{(settings.ROOT_DIR / 'optuna.db').as_posix()}"
TIMEOUT_SECONDS: Final[float] = 4 * 3600
MAX_TRIALS: Final[int] = 150

RULE_RECALL: Final[float] = 0.9788
"""The signature rule on the same temporal folds (EVAL-005)."""

DEFAULT_RECALL: Final[dict[str, float]] = {
    "logistic_regression": 0.9911,
    "random_forest": 0.9983,
    "lightgbm": 0.0,
}
"""The true default configurations on the same folds (EVAL-006 to EVAL-008)."""


class TrialLogger:
    """Log each trial as a nested MLflow run and print one line for it."""

    def __init__(self, shared: dict[str, str]) -> None:
        """Keep the tags every trial run shares.

        Args:
            shared: Tags from ``base_tags``.
        """
        self.shared = shared

    def __call__(
        self, trial: optuna.Trial, algorithm: str, folds: list[FoldResult], state: str
    ) -> None:
        """Log one trial.

        Args:
            trial: The trial.
            algorithm: Its algorithm.
            folds: The folds it ran.
            state: ``complete`` or ``pruned``.
        """
        run = GuardedRun(
            folds=tuple(folds),
            aborted_fold=None,
            aborted_after_seconds=None,
            error=None,
            wall_seconds=sum(fold.fit_seconds + fold.predict_seconds for fold in folds),
        )
        tags = {
            **self.shared,
            "study": STUDY_NAME,
            "trial": str(trial.number),
            "trial_state": state,
            "model": algorithm,
            "configuration": EVALUATED_CONFIGURATION,
            "splitter": TEMPORAL,
            "label_shuffle": "none",
            "objective": OBJECTIVE_METRIC,
        }
        run_id = log_guarded_run(
            run,
            run_name=f"trial_{trial.number:03d}__{algorithm}",
            tags=tags,
            params=trial.params,
            nested=True,
        )
        values = [round(fold.metrics[OBJECTIVE_METRIC], 4) for fold in folds]
        fit = sum(fold.fit_seconds for fold in folds)
        print(
            f"trial {trial.number:3d} {algorithm:19} {state:8} folds {values} "
            f"fit {fit:7.1f}s run {run_id[:8]}",
            flush=True,
        )


def _summary(study: optuna.Study) -> None:
    trials = study.trials
    by_state = Counter((t.user_attrs.get("algorithm", "?"), t.state.name) for t in trials)
    print("\n=== TRIALS BY FAMILY AND STATE")
    for algorithm in ("logistic_regression", "random_forest", "lightgbm"):
        counts = {state: by_state[algorithm, state] for state in ("COMPLETE", "PRUNED", "FAIL")}
        done = [
            t for t in trials if t.user_attrs.get("algorithm") == algorithm and t.value is not None
        ]
        best = max(done, key=lambda t: t.value, default=None)
        best_text = f"best {best.value:.4f} (trial {best.number})" if best else "no complete trial"
        print(f"{algorithm:19} {counts} | {best_text}")
    failed = [t for t in trials if t.state.name == "FAIL"]
    for trial in failed:
        print(f"  failed trial {trial.number}: {trial.system_attrs.get('fail_reason', '')[:300]}")

    print("\n=== STARTING POINTS (trials 0-2) AGAINST THE TRUE DEFAULTS")
    for trial in trials[:3]:
        algorithm = trial.user_attrs.get("algorithm", "?")
        value = trial.state.name if trial.value is None else f"{trial.value:.4f}"
        default = DEFAULT_RECALL.get(algorithm)
        print(f"trial {trial.number} {algorithm:19} {value} | true default {default}")

    complete = [t for t in trials if t.value is not None]
    if not complete:
        print("\nNo complete trial.")
        return
    best = study.best_trial
    print("\n=== BEST TRIAL")
    algorithm = best.user_attrs.get("algorithm")
    print(f"trial {best.number} {algorithm} {OBJECTIVE_METRIC} mean {best.value:.4f}")
    print(f"per fold: {[round(v, 4) for _, v in sorted(best.intermediate_values.items())]}")
    print(f"params: {best.params}")
    print(f"against the rule ({RULE_RECALL}): {best.value - RULE_RECALL:+.4f}")
    for algorithm, value in DEFAULT_RECALL.items():
        print(f"against the default {algorithm} ({value}): {best.value - value:+.4f}")

    lightgbm = [t for t in trials if t.user_attrs.get("algorithm") == "lightgbm"]
    first_fold = [t.intermediate_values.get(1) for t in lightgbm if 1 in t.intermediate_values]
    zero = sum(1 for value in first_fold if value == 0.0)
    print("\n=== LIGHTGBM: DOES IT STILL DIVERGE?")
    print(
        f"trials {len(lightgbm)}; fold-1 recall 0 in {zero} of {len(first_fold)}; "
        f"fold-1 recall > 0.9 in {sum(1 for v in first_fold if v > 0.9)}"
    )
    if first_fold:
        print(
            f"fold-1 recall: min {min(first_fold):.4f} median {np.median(first_fold):.4f} "
            f"max {max(first_fold):.4f}"
        )


def main() -> int:
    """Create or resume the study, run it within its budget, and report."""
    # The project's standard defense (ERR-012): redirected, stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    target, failure = configure_with_local_fallback()
    if failure is not None:
        print("Remote tracking failed; logging to the local store. Full error:\n" + failure)
    print(f"tracking URI : {target.tracking_uri}")
    print(f"experiment   : {target.experiment_name} (id {target.experiment_id})")
    print(f"storage      : {STORAGE_URI} | study {STUDY_NAME}")
    print(f"budget       : {TIMEOUT_SECONDS:.0f} s or {MAX_TRIALS} trials, whichever first")
    identity = reference_identity()
    shared = base_tags(identity)

    features, labels = features_and_labels(load_evaluation_frame())
    splits = list(ExpandingStepSplit().split(features, labels))
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URI,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    if not study.trials:
        for params in STARTING_POINTS:
            study.enqueue_trial(params)
    # Enqueued trials wait in the study as WAITING; only finished ones use up the budget,
    # and n_trials below counts the enqueued ones it runs.
    finished = [t for t in study.trials if t.state.is_finished()]
    remaining = MAX_TRIALS - len(finished)
    waiting = len(study.trials) - len(finished)
    print(
        f"trials finished {len(finished)}, waiting or running {waiting}; "
        f"this run may add {max(remaining, 0)}",
        flush=True,
    )
    started = time.monotonic()
    with mlflow.start_run(run_name=f"search__{STUDY_NAME}") as parent:
        mlflow.set_tags({**shared, "study": STUDY_NAME, "storage": STORAGE_URI})
        mlflow.log_params({"timeout_seconds": TIMEOUT_SECONDS, "max_trials": MAX_TRIALS})
        if remaining > 0:
            study.optimize(
                make_objective(features, labels, splits, TrialLogger(shared)),
                n_trials=remaining,
                timeout=TIMEOUT_SECONDS,
                catch=(Exception,),
                gc_after_trial=True,
            )
        complete = [t for t in study.trials if t.value is not None]
        if complete:
            mlflow.log_metrics({"best_value": study.best_value, "trials": len(study.trials)})
            mlflow.set_tag("best_trial", str(study.best_trial.number))
        print(f"\nsearch run {parent.info.run_id}; elapsed {time.monotonic() - started:.0f} s")
    _summary(study)
    return 0


if __name__ == "__main__":
    sys.exit(main())
