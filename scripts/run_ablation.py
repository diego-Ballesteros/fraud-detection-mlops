"""The ablation of ADR 0007: how much of the performance is the simulator's signature.

Run from the repository root::

    uv run python scripts/run_ablation.py

**Declared before measuring, in this docstring, before the first run (2026-10-22).**

- **What is ablated.** The ``no_signature`` configuration drops ``oldbalanceOrg`` and the
  flag ``amount_equals_origin_balance``: without the origin balance the model cannot rebuild
  the signature. Everything else is trial 5 as registered (ADR 0010): the same
  hyperparameters (``registry.CHOSEN_PARAMS``) and the same seed, not re-tuned for this
  configuration. The ablation measures the chosen model without the signature, not the best
  model one could build without it: a search on this configuration might do better.
- **Validation.** The same guarded harness and the same five temporal folds as EVAL-009,
  with ``full`` (trial 5) and the rule (``SignatureRule``) beside it, the three logged as
  children of one MLflow run. Control: ``full`` must reproduce trial 5's recall fold by
  fold, to the four decimals EVAL-010 records, or the script stops before reading
  production: a comparison against a ``full`` that is not trial 5 compares against another
  model.
- **Metrics.** Recall at precision >= 0.99, the decision metric, with PR-AUC and Brier score
  beside it; recall at precision >= 0.95 and ROC-AUC as context. If no threshold reaches
  precision 0.99, the recall is 0, and that is the result, not a failure.
- **Where it does reach.** For ``no_signature``, on the scores of each fold, refitted by the
  same code path (``evaluation.score_one_fold``): the recall at each precision floor of
  ``FLOORS``, and the best precision any threshold reaches while flagging at least one
  fraud, with its recall. A configuration that cannot operate at the required floor still
  says how much signal is left without the signature. The refit must give the harness's
  fold metrics again, or the script stops.
- **Production.** ``no_signature`` is fitted on every reference row, and its threshold is
  the decision metric's point on its own reference scores, in-sample: the policy of
  ADR 0011, applied to this model. Not the ``full`` artifact's threshold: they are two
  models, and the same score does not mean the same thing in both. If no reference threshold
  reaches precision 0.99, there is no operating point, and the production recall at the
  decision floor is 0 by definition. Production is scored once at that threshold, beside the
  ``full`` artifact at its own fixed threshold and the rule. The ``FLOORS`` points on
  production are read, never applied. Nothing fitted here is serialized or registered.

Steps: 1. the rule, ``full`` and ``no_signature`` through the harness; 2. where
``no_signature`` reaches, fold by fold; 3. its fit on the whole reference and its in-sample
threshold; 4. production; 5. MLflow. Tables go to ``reports/ablation/``, not versioned.
"""

import sys
import time
from fractions import Fraction
from functools import partial
from typing import Any, Final

import mlflow
import numpy as np
import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data.loader import read_production
from fraud_detection.models.baseline import SignatureRule
from fraud_detection.models.estimators import (
    ABLATION_CONFIGURATION,
    EVALUATED_CONFIGURATION,
    MODEL_SPECS,
    ModelSpec,
    full_preprocessor,
    no_signature_preprocessor,
)
from fraud_detection.models.evaluation import (
    DECISION_PRECISION,
    EVALUATION_COLUMNS,
    SPLITTERS,
    TEMPORAL,
    aggregate,
    compose,
    features_and_labels,
    load_evaluation_frame,
    precision_at_recall,
    recall_at_precision,
    score_fold,
    score_one_fold,
)
from fraud_detection.models.registry import ARTIFACT_PATH, CHOSEN_TRIAL, chosen_model, load_pipeline
from fraud_detection.models.search_space import OBJECTIVE_METRIC
from fraud_detection.models.tracking import (
    base_tags,
    evaluate_and_log,
    file_sha256,
    reference_identity,
)
from fraud_detection.monitoring.performance import OPERATING_THRESHOLD, evaluate_rows
from fraud_detection.tracking.mlflow_setup import configure_with_local_fallback

ARTIFACT_SHA256: Final[str] = "41869ee0e5239ca66e0024a10bff4b727cf7ba0a6b6a8087a45da010de5d4ebd"
"""The artifact ADR 0010 registered: the ``full`` model on production is this one."""

TRIAL_5_FOLDS: Final[tuple[float, ...]] = (0.9914, 0.9956, 1.0, 1.0, 0.9978)
"""Trial 5's decision metric per temporal fold, as EVAL-010 records it (four decimals)."""

FLOORS: Final[tuple[float, ...]] = (0.99, 0.95, 0.90, 0.80, 0.50)
"""Precision floors at which ``no_signature``'s recall is read, the decision floor first."""

FOLD_METRICS: Final[tuple[str, ...]] = (
    OBJECTIVE_METRIC,
    "attained_p0.99",
    "recall_at_p0.95",
    "pr_auc",
    "brier",
    "roc_auc",
)
REFIT_TOLERANCE: Final[float] = 1e-12
OUTPUT_DIR: Final = settings.REPORTS_DIR / "ablation"
MODEL: Final = partial(chosen_model, settings.RANDOM_STATE)


def _where_it_reaches(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Recall at each floor of ``FLOORS``, and the best precision flagging one fraud or more."""
    row = {}
    for floor in FLOORS:
        point = recall_at_precision(labels, scores, floor)
        row[f"recall_at_p{floor:.2f}"] = point.recall if point else 0.0
    best = precision_at_recall(labels, scores, Fraction(1, int(np.sum(labels))))
    row["best_precision"] = best.precision
    row["recall_at_best_precision"] = best.recall
    row["flagged_at_best_precision"] = float(best.flagged)
    return row


def _fold_table(folds: Any) -> pd.DataFrame:
    return pd.DataFrame(
        {
            fold.fold: {name: fold.metrics.get(name, np.nan) for name in FOLD_METRICS}
            for fold in folds
        }
    ).T.rename_axis("fold")


def main() -> int:
    """Run the five steps; exit with 1 if a run is incomplete or a control fails."""
    # The project's standard defense (ERR-012): redirected, stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    target, failure = configure_with_local_fallback()
    if failure is not None:
        print("Remote tracking failed; using the local store. Full error:\n" + failure)
    print(f"tracking URI : {target.tracking_uri} | experiment {target.experiment_name}")
    shared = {**base_tags(reference_identity()), "turn": "5", "purpose": "ablation"}
    specs = {
        "rule": MODEL_SPECS["signature_rule"],
        EVALUATED_CONFIGURATION: ModelSpec("lightgbm", full_preprocessor, MODEL, True),
        ABLATION_CONFIGURATION: ModelSpec("lightgbm", no_signature_preprocessor, MODEL, True),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=f"ablation__{ABLATION_CONFIGURATION}") as parent:
        mlflow.set_tags({**shared, "decision": "ADR 0007", "phase": "5"})

        print("\n=== 1. THE HARNESS: FIVE TEMPORAL FOLDS")
        folds: dict[str, Any] = {}
        for name, spec in specs.items():
            started = time.perf_counter()
            run, run_id = evaluate_and_log(
                spec,
                TEMPORAL,
                shared_tags=shared,
                extra_tags={"configuration": name, "decision": "ADR 0007"},
                run_suffix=f"__ablation_{name}",
                nested=True,
            )
            print(f"{name}: run {run_id[:8]}, {time.perf_counter() - started:.0f} s")
            if not run.complete:
                print(f"incomplete ({run.error or run.aborted_fold}); stopping")
                return 1
            folds[name] = run.folds
            print(_fold_table(run.folds).to_string(float_format=lambda v: f"{v:.6f}"))
        tables = {name: _fold_table(fold_results) for name, fold_results in folds.items()}
        aggregates = {name: aggregate(fold_results) for name, fold_results in folds.items()}
        summary = pd.DataFrame(
            {
                name: {
                    f"{metric}_{stat}": getattr(values[metric], stat)
                    for metric in FOLD_METRICS
                    if metric in values
                    for stat in ("mean", "std")
                }
                for name, values in aggregates.items()
            }
        )
        print("\nmean and sample std (ddof = 1) over the five folds:")
        print(summary.to_string(float_format=lambda v: f"{v:.6f}"))
        reproduced = tuple(round(v, 4) for v in tables[EVALUATED_CONFIGURATION][OBJECTIVE_METRIC])
        print(f"control: full reproduces trial {CHOSEN_TRIAL} fold by fold: {reproduced}")
        if reproduced != TRIAL_5_FOLDS:
            print(f"expected {TRIAL_5_FOLDS} (EVAL-010); stopping before production")
            return 1

        print(f"\n=== 2. WHERE {ABLATION_CONFIGURATION} REACHES, FOLD BY FOLD (refitted)")
        frame = load_evaluation_frame()
        splitter = SPLITTERS[TEMPORAL]()
        reach = {}
        for fold in folds[ABLATION_CONFIGURATION]:
            scored = score_one_fold(frame, splitter, fold.fold, no_signature_preprocessor, MODEL)
            again = score_fold(
                scored.labels, scored.scores, SignatureRule().predict(scored.validation)
            )
            gap = max(
                abs(again[m] - fold.metrics[m]) for m in (OBJECTIVE_METRIC, "pr_auc", "brier")
            )
            if gap > REFIT_TOLERANCE:
                print(
                    f"fold {fold.fold}: the refit differs from the harness by {gap:.2e}; stopping"
                )
                return 1
            reach[fold.fold] = {
                **_where_it_reaches(scored.labels, scored.scores),
                "positives": float(scored.labels.sum()),
            }
        reach_table = pd.DataFrame(reach).T.rename_axis("fold")
        print("(the refit equals the harness in every fold, to 1e-12)")
        print(reach_table.to_string(float_format=lambda v: f"{v:.6f}"))
        del frame

        print(f"\n=== 3. {ABLATION_CONFIGURATION} ON THE WHOLE REFERENCE, IN-SAMPLE THRESHOLD")
        reference = load_evaluation_frame()
        features, labels = features_and_labels(reference)
        ablated = compose(no_signature_preprocessor, MODEL)
        started = time.perf_counter()
        ablated.fit(features, labels)
        print(
            f"fitted on {len(labels)} rows, {int(labels.sum())} positives, in "
            f"{time.perf_counter() - started:.1f} s"
        )
        reference_scores = ablated.predict_proba(features)[:, 1]
        point = recall_at_precision(labels, reference_scores, DECISION_PRECISION)
        threshold = point.threshold if point else float("inf")
        if point:
            print(
                f"in-sample point at precision >= {DECISION_PRECISION}: threshold "
                f"{point.threshold!r}, flags {point.flagged} ({point.true_positives} fraud), "
                f"precision {point.precision:.6f}, recall {point.recall:.6f}"
            )
        else:
            print(
                f"no reference threshold reaches precision {DECISION_PRECISION}, in-sample: "
                "no operating point; production recall at the decision floor is 0"
            )
        in_sample = pd.Series(_where_it_reaches(labels, reference_scores), name="reference")
        del features, reference

        print("\n=== 4. PRODUCTION (read once; nothing is fitted on it)")
        digest = file_sha256(ARTIFACT_PATH)
        if digest != ARTIFACT_SHA256:
            print(f"artifact sha256 {digest} is not ADR 0010's; stopping")
            return 1
        production = read_production()[list(EVALUATION_COLUMNS)]
        rows, production_labels = features_and_labels(production)
        rule_flags = SignatureRule().predict(rows)
        ablated_scores = ablated.predict_proba(rows)[:, 1]
        side = {
            ABLATION_CONFIGURATION: evaluate_rows(
                production_labels, ablated_scores, rule_flags, threshold
            ),
            EVALUATED_CONFIGURATION: evaluate_rows(
                production_labels,
                load_pipeline(ARTIFACT_PATH).predict_proba(rows)[:, 1],
                rule_flags,
                OPERATING_THRESHOLD,
            ),
        }
        production_table = pd.DataFrame(
            {
                **{
                    name: {
                        "threshold": t,
                        "flagged": m["model_flagged"],
                        "false_positives": m["model_false_positives"],
                        "false_negatives": m["model_false_negatives"],
                        "precision": m["model_precision"],
                        "recall": m["model_recall"],
                        "pr_auc": m["pr_auc"],
                        "roc_auc": m["roc_auc"],
                        "brier": m["brier"],
                    }
                    for (name, m), t in zip(
                        side.items(), (threshold, OPERATING_THRESHOLD), strict=True
                    )
                },
                "rule": {
                    "threshold": np.nan,
                    "flagged": side[EVALUATED_CONFIGURATION]["rule_flagged"],
                    "false_positives": side[EVALUATED_CONFIGURATION]["rule_false_positives"],
                    "false_negatives": side[EVALUATED_CONFIGURATION]["rule_false_negatives"],
                    "precision": side[EVALUATED_CONFIGURATION]["rule_precision"],
                    "recall": side[EVALUATED_CONFIGURATION]["rule_recall"],
                    "pr_auc": side[EVALUATED_CONFIGURATION]["rule_pr_auc"],
                    "roc_auc": side[EVALUATED_CONFIGURATION]["rule_roc_auc"],
                    "brier": side[EVALUATED_CONFIGURATION]["rule_brier"],
                },
            }
        )
        print(f"rows {len(production_labels)}, positives {int(production_labels.sum())}")
        print(production_table.to_string(float_format=lambda v: f"{v:.6f}"))
        read = pd.Series(_where_it_reaches(production_labels, ablated_scores), name="production")
        print(f"\nwhere {ABLATION_CONFIGURATION} reaches, read and never applied:")
        print(pd.concat([in_sample, read], axis=1).to_string(float_format=lambda v: f"{v:.6f}"))

        print("\n=== 5. MLFLOW")
        summary.to_csv(OUTPUT_DIR / "folds_summary.csv")
        pd.concat(tables, names=["configuration"]).to_csv(OUTPUT_DIR / "folds.csv")
        reach_table.to_csv(OUTPUT_DIR / "where_it_reaches_folds.csv")
        pd.concat([in_sample, read], axis=1).to_csv(OUTPUT_DIR / "where_it_reaches_periods.csv")
        production_table.to_csv(OUTPUT_DIR / "production.csv")
        mlflow.log_params({"threshold_no_signature": repr(threshold), "floors": str(FLOORS)})
        mlflow.log_metrics(
            {
                **{
                    f"{name}_{key}": float(value)
                    for name, column in summary.items()
                    for key, value in column.items()
                    if pd.notna(value)
                },
                **{
                    f"production_{name}_{key}": float(value)
                    for name, column in production_table.items()
                    for key, value in column.items()
                    if pd.notna(value) and np.isfinite(value)
                },
            }
        )
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="ablation")
        print(f"run {parent.info.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
