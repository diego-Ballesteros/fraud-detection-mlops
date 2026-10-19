"""The shuffled-target test as a permutation test, with the random classifier as its null.

Run from the repository root::

    uv run python scripts/run_leakage_check.py

Two measurements, in order, both on the temporal splitter and both through the same
harness as every model:

1. **The random classifier:** uniform scores, no model, one run per seed of
   ``derived_seeds(PERMUTATIONS)``. Its PR-AUC distribution is what a ranking that knows
   nothing reaches on these labels, the upward bias of few positives included.
2. **The permutation test:** the training labels of every fold permuted with each seed,
   the validation labels intact, the full Pipeline fitted. LightGBM, the cheapest and most
   sensitive detector, and logistic regression get 20 permutations; Random Forest gets 1. A
   leak is a property of the pipeline and the fold structure, not of the model family.

Seed 0 of ``derived_seeds`` is ``settings.RANDOM_STATE``, the single permutation of the
first test, so draw 0 must reproduce it exactly.

Criterion, set by the Architect after the single-permutation test failed (ADR 0009, status
note):

- **Blocking:** for every family, the largest PR-AUC over every permutation and fold is
  below ``MAX_PERMUTED_PR_AUC``, in absolute value.
- **Reported, not blocking**, as the Executor operationalized it before measuring:
  R1, 0.5 lies within the [min, max] of the family's ROC-AUC values; R2, no systematic
  elevation: the family's median ROC-AUC is not above the largest ROC-AUC the random
  classifier reaches.

Exit code 0 if the blocking criterion holds and every run completed; 1 otherwise.
"""

import sys
from collections.abc import Sequence
from typing import Final

import mlflow
import numpy as np
import pandas as pd

from fraud_detection.models.estimators import MODEL_SPECS, ModelSpec, random_classifier_spec
from fraud_detection.models.evaluation import PERMUTATIONS, TEMPORAL, derived_seeds
from fraud_detection.models.fit_guard import GuardedRun
from fraud_detection.models.tracking import base_tags, evaluate_and_log, reference_identity
from fraud_detection.tracking.mlflow_setup import configure_with_local_fallback

MAX_PERMUTED_PR_AUC: Final[float] = 0.05
"""Blocking: every PR-AUC under permutation, in absolute value, stays below this."""

DRAWS: Final[dict[str, int]] = {
    "lightgbm": PERMUTATIONS,
    "logistic_regression": PERMUTATIONS,
    "random_forest": 1,
}
"""Permutations per family. Random Forest gets one: 20 x 5 fits of ~320 s buy nothing."""

DIAGNOSED: Final[tuple[str, int]] = ("lightgbm", 2)
"""The family and fold that failed the single-permutation test; all its values are printed."""

QUANTILE: Final[float] = 95.0


def _describe(values: Sequence[float]) -> dict[str, float]:
    """Min, median, 95th percentile (linear interpolation) and max."""
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, QUANTILE)),
        "max": float(array.max()),
    }


def _values(runs: Sequence[GuardedRun], metric: str) -> pd.DataFrame:
    """One row per draw, one column per fold."""
    return pd.DataFrame(
        [{fold.fold: fold.metrics[metric] for fold in run.folds} for run in runs]
    ).rename_axis(index="draw", columns="fold")


def _lift(runs: Sequence[GuardedRun]) -> pd.DataFrame:
    return _values(runs, "pr_auc") / _values(runs, "prevalence")


def _distribution_table(runs: Sequence[GuardedRun]) -> pd.DataFrame:
    """Per fold and pooled: the distribution of PR-AUC, of PR-AUC / prevalence, of ROC-AUC."""
    columns = {"pr_auc": _values(runs, "pr_auc"), "lift": _lift(runs)}
    columns["roc_auc"] = _values(runs, "roc_auc")
    rows = {}
    for fold in [*columns["pr_auc"].columns, "pooled"]:
        row = {}
        for name, table in columns.items():
            series = table.to_numpy().ravel() if fold == "pooled" else table[fold].to_numpy()
            row.update({f"{name}_{key}": value for key, value in _describe(series).items()})
        rows[fold] = row
    return pd.DataFrame(rows).T.rename_axis("fold")


def _run_study(
    name: str,
    draws: Sequence[tuple[ModelSpec, int]],
    *,
    shuffle: bool,
    shared: dict[str, str],
) -> list[GuardedRun]:
    """Run every draw as a child of one parent MLflow run; log the distribution on the parent."""
    runs = []
    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({**shared, "study": name, "draws": str(len(draws))})
        for draw, (spec, seed) in enumerate(draws):
            seed_tag = {"draw": str(draw), "study": name}
            if not shuffle:
                seed_tag["score_seed"] = str(seed)
            run, run_id = evaluate_and_log(
                spec,
                TEMPORAL,
                shared_tags=shared,
                shuffle_training_labels=shuffle,
                shuffle_seed=seed,
                extra_tags=seed_tag,
                run_suffix=f"__draw{draw:02d}",
                nested=True,
            )
            runs.append(run)
            pr_auc = [round(fold.metrics["pr_auc"], 5) for fold in run.folds]
            print(
                f"  draw {draw:02d} seed {seed} run {run_id[:8]} complete={run.complete} "
                f"wall={run.wall_seconds:.0f}s pr_auc={pr_auc}",
                flush=True,
            )
        table = _distribution_table(runs)
        pooled = table.loc["pooled"]
        mlflow.log_metrics({f"pooled_{key}": float(value) for key, value in pooled.items()})
        for fold in table.index.drop("pooled"):
            mlflow.log_metrics(
                {f"{key}": float(value) for key, value in table.loc[fold].items()}, step=int(fold)
            )
        mlflow.set_tag("complete", str(all(run.complete for run in runs)).lower())
        print(f"  parent run {parent.info.run_id}")
    return runs


def main() -> int:
    """Measure the random classifier's null, then the permutation test, then decide."""
    # Redirected, stdout on Windows is cp1252, and MLflow prints an emoji when a remote run
    # ends: without this, end_run raises and the run stays RUNNING (ERR-012).
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    target, failure = configure_with_local_fallback()
    if failure is not None:
        print("Remote tracking failed; logging to the local store. Full error:\n" + failure)
    print(f"tracking URI : {target.tracking_uri}")
    print(f"experiment   : {target.experiment_name} (id {target.experiment_id})")
    identity = reference_identity()
    shared = base_tags(identity)
    print(f"reference    : {identity.rows} rows, steps {identity.first_step}-{identity.last_step}")
    print(f"sha256       : {identity.sha256}")
    print(f"blocking     : max PR-AUC under permutation < {MAX_PERMUTED_PR_AUC}, every family")

    seeds = derived_seeds(PERMUTATIONS)
    print(f"\n=== RANDOM CLASSIFIER · {len(seeds)} seeds {seeds[0]}..{seeds[-1]}", flush=True)
    null_runs = _run_study(
        "random_classifier__null",
        [(random_classifier_spec(seed), seed) for seed in seeds],
        shuffle=False,
        shared=shared,
    )
    null_table = _distribution_table(null_runs)
    print(null_table.to_string(float_format=lambda value: f"{value:.6g}"))
    null_roc_max = float(null_table.loc["pooled", "roc_auc_max"])

    studies: dict[str, list[GuardedRun]] = {}
    for family, count in DRAWS.items():
        print(f"\n=== PERMUTATION TEST · {family} · {count} permutation(s)", flush=True)
        draws = [(MODEL_SPECS[family], seed) for seed in seeds[:count]]
        studies[family] = _run_study(
            f"permutation_test__{family}", draws, shuffle=True, shared=shared
        )
        print(_distribution_table(studies[family]).to_string(float_format=lambda v: f"{v:.6g}"))

    family, fold = DIAGNOSED
    print(f"\n=== {family} fold {fold}: the value of every permutation")
    diagnosed = pd.DataFrame(
        {
            "seed": seeds[: len(studies[family])],
            "pr_auc": _values(studies[family], "pr_auc")[fold],
            "lift": _lift(studies[family])[fold],
            "roc_auc": _values(studies[family], "roc_auc")[fold],
        }
    )
    print(diagnosed.to_string(float_format=lambda value: f"{value:.6g}"))

    print("\n=== VERDICT")
    complete = all(run.complete for run in null_runs) and all(
        run.complete for runs in studies.values() for run in runs
    )
    passed = True
    for family, runs in studies.items():
        pr_auc = _values(runs, "pr_auc").to_numpy().ravel()
        roc_auc = _values(runs, "roc_auc").to_numpy().ravel()
        blocking = float(pr_auc.max()) < MAX_PERMUTED_PR_AUC
        r1 = float(roc_auc.min()) <= 0.5 <= float(roc_auc.max())
        r2 = float(np.median(roc_auc)) <= null_roc_max
        passed &= blocking
        print(
            f"{family:20} values={pr_auc.size:3d} max PR-AUC={pr_auc.max():.6f} "
            f"(< {MAX_PERMUTED_PR_AUC}: {blocking}) | ROC-AUC min={roc_auc.min():.4f} "
            f"median={np.median(roc_auc):.4f} max={roc_auc.max():.4f} | R1 contains 0.5: {r1} "
            f"| R2 median <= random max {null_roc_max:.4f}: {r2}"
        )
    print(f"all runs complete: {complete}")
    verdict = passed and complete
    print(f"VERDICT: {'PASSES' if verdict else 'BLOCKER'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
