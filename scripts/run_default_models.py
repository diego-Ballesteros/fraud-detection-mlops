"""Phase 2A: the baselines and the three families with default hyperparameters.

Run from the repository root::

    uv run python scripts/run_default_models.py
    uv run python scripts/run_default_models.py --models majority_class signature_rule

``--splitters`` narrows the splitters the same way.

Each model runs through both splitters with the ``full`` configuration, every fit guarded
by the 20-minute limit of ``fit_guard``, and every run is logged to MLflow. Only the
reference period is read. The temporal splitter decides; the stratified one measures the
optimism of ignoring time.

At the end: the rows and positives of every fold of each splitter, the mean and sample
standard deviation of each metric per model and splitter, and the fit time of each family
on temporal fold 5, the largest training set, which sets the search budget of turn 2B.
"""

import argparse
import os
import sys

from fraud_detection.models.estimators import MODEL_SPECS
from fraud_detection.models.evaluation import SPLITTERS, TEMPORAL, aggregate, fold_table
from fraud_detection.models.fit_guard import GuardedRun
from fraud_detection.models.tracking import base_tags, evaluate_and_log, reference_identity
from fraud_detection.tracking.mlflow_setup import configure_with_local_fallback

REPORTED = (
    "recall_at_p0.99",
    "recall_at_p0.95",
    "pr_auc",
    "roc_auc",
    "brier",
    "precision_at_baseline_recall",
    "recall_at_baseline_precision",
)
FOLD_COLUMNS = ("prevalence", "recall_at_p0.99", "precision_at_p0.99", "pr_auc", "baseline_recall")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--splitters", nargs="+", choices=list(SPLITTERS), default=list(SPLITTERS))
    return parser.parse_args()


def _report_run(name: str, splitter: str, run: GuardedRun, run_id: str) -> None:
    print(f"\n=== {name} · {splitter} · run {run_id}")
    print(f"complete={run.complete} wall={run.wall_seconds:.0f}s")
    if run.aborted_fold is not None:
        print(f"ABORTED: fold {run.aborted_fold} fit ran {run.aborted_after_seconds:.0f} s")
    if run.error is not None:
        print(f"ERROR:\n{run.error}")
    if run.folds:
        print(fold_table(run.folds, FOLD_COLUMNS).to_string())
    for fold in run.folds:
        if fold.fit_warnings:
            print(f"fold {fold.fold} fit warnings: {list(fold.fit_warnings)}")


def main() -> int:
    """Evaluate and log every requested model through every requested splitter."""
    # Redirected, stdout on Windows is cp1252, and MLflow prints an emoji when a remote run
    # ends: without this, end_run raises and the run stays RUNNING (ERR-012).
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = _arguments()
    target, failure = configure_with_local_fallback()
    if failure is not None:
        print("Remote tracking failed; logging to the local store. Full error:\n" + failure)
    print(f"tracking URI : {target.tracking_uri}")
    print(f"experiment   : {target.experiment_name} (id {target.experiment_id})")
    identity = reference_identity()
    shared = base_tags(identity)
    print(f"reference    : {identity.rows} rows, steps {identity.first_step}-{identity.last_step}")
    print(f"sha256       : {identity.sha256}")
    print(f"logical cores: {os.cpu_count()}")

    results: dict[tuple[str, str], GuardedRun] = {}
    for name in arguments.models:
        for splitter in arguments.splitters:
            run, run_id = evaluate_and_log(MODEL_SPECS[name], splitter, shared_tags=shared)
            results[name, splitter] = run
            _report_run(name, splitter, run, run_id)

    print("\n=== SUMMARY: mean ± sample std across folds")
    for (name, splitter), run in results.items():
        summary = aggregate(run.folds)
        cells = [
            f"{metric}={summary[metric].mean:.4f}±{summary[metric].std:.4f}"
            for metric in REPORTED
            if metric in summary
        ]
        state = "complete" if run.complete else f"INCOMPLETE ({len(run.folds)} folds)"
        print(f"{name:20} {splitter:19} {state} | " + " ".join(cells))

    print("\n=== FIT TIME ON TEMPORAL FOLD 5 (train steps 1-360), seconds")
    for (name, splitter), run in results.items():
        if splitter != TEMPORAL:
            continue
        fifth = [fold for fold in run.folds if fold.fold == 5]
        if fifth:
            print(f"{name:20} fit {fifth[0].fit_seconds:8.1f} s on {fifth[0].train_rows} rows")
        elif run.aborted_fold == 5:
            print(f"{name:20} ABORTED after {run.aborted_after_seconds:.0f} s")
        else:
            print(f"{name:20} not reached (aborted at fold {run.aborted_fold})")
    return 0 if all(run.complete for run in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
