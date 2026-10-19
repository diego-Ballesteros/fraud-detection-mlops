"""The shuffled-target test: the positive control of the whole evaluation harness.

Run from the repository root::

    uv run python scripts/run_leakage_check.py

For each family that learns from labels (logistic regression, random forest, LightGBM),
the labels of the training part of every temporal fold are permuted with
``settings.RANDOM_STATE``, which keeps the class counts exactly, and the validation labels
stay intact. The full Pipeline then runs through the same harness as every other model,
not through a separate path. Shuffling the validation labels too would prove nothing: the
PR-AUC would equal the base rate trivially.

Collapse criterion, declared before the first run of this script:

    For every family and every fold, PR-AUC <= COLLAPSE_LIFT x the fold's prevalence.

Why 10. On synthetic data where a leak is impossible by construction (plain scikit-learn
and LightGBM, no project harness), with positives clustered and a rare binary feature that
marks 98% of them, as the signature does here, the shuffled-label PR-AUC reached 4.6 times
the prevalence for logistic regression and 2.0 times for LightGBM, over 12 permutations
each. A margin of 2 would flag that leak-free setup in 3 of 12 permutations. A leak gives
the model the true labels, and through the signature even a small leak produces a nearly
perfect ranking: 10 times the prevalence (about 0.0077) is under 1% of the baseline's
0.979. The criterion holds per fold, not on average, because a leak through the fold
structure can be specific to one fold. (The synthetic measurement was ad hoc and is not
versioned.)

If a fold fails, the line also shows how many shuffled positives landed on training rows
with the signature against how many were expected: chance alignment and leakage are
told apart there. The verdict does not change: a fold that does not collapse is a blocker.

Exit code 0 if every fold of every family collapses; 1 otherwise, including when a family
could not be measured (aborted or failed).
"""

import sys
from typing import Final

from fraud_detection.models.estimators import MODEL_SPECS
from fraud_detection.models.evaluation import TEMPORAL, fold_table
from fraud_detection.models.tracking import base_tags, evaluate_and_log, reference_identity
from fraud_detection.tracking.mlflow_setup import configure_with_local_fallback

COLLAPSE_LIFT: Final[float] = 10.0
"""Largest PR-AUC / prevalence ratio accepted as collapse, in every fold."""

FAMILIES: Final[tuple[str, ...]] = ("logistic_regression", "random_forest", "lightgbm")


def main() -> int:
    """Run the test for every family and report the verdict per fold."""
    target, failure = configure_with_local_fallback()
    if failure is not None:
        print("Remote tracking failed; logging to the local store. Full error:\n" + failure)
    print(f"tracking URI : {target.tracking_uri}")
    print(f"experiment   : {target.experiment_name} (id {target.experiment_id})")
    identity = reference_identity()
    shared = base_tags(identity)
    print(f"reference    : {identity.rows} rows, steps {identity.first_step}-{identity.last_step}")
    print(f"sha256       : {identity.sha256}")
    print(f"criterion    : PR-AUC <= {COLLAPSE_LIFT:g} x prevalence, every fold of every family")

    collapsed = True
    for name in FAMILIES:
        run, run_id = evaluate_and_log(
            MODEL_SPECS[name], TEMPORAL, shared_tags=shared, shuffle_training_labels=True
        )
        print(f"\n=== {name} · shuffled training labels · run {run_id}")
        print(f"complete={run.complete} wall={run.wall_seconds:.0f}s")
        if not run.complete:
            collapsed = False
            print(
                f"NOT MEASURED: aborted fold {run.aborted_fold} after "
                f"{run.aborted_after_seconds} s; error: {run.error}"
            )
        table = fold_table(run.folds, ("prevalence", "pr_auc", "roc_auc", "recall_at_p0.99"))
        table["lift"] = table["pr_auc"] / table["prevalence"]
        table["signature_positives"] = [fold.train_signature_positives for fold in run.folds]
        table["expected"] = [
            fold.train_signature_rows * fold.train_positives / fold.train_rows for fold in run.folds
        ]
        table["collapsed"] = table["lift"] <= COLLAPSE_LIFT
        print(table.to_string())
        for fold in run.folds:
            if fold.fit_warnings:
                print(f"fold {fold.fold} fit warnings: {list(fold.fit_warnings)}")
        collapsed &= bool(table["collapsed"].all())
    print(f"\nVERDICT: {'COLLAPSED' if collapsed else 'DID NOT COLLAPSE (blocker)'}")
    return 0 if collapsed else 1


if __name__ == "__main__":
    sys.exit(main())
