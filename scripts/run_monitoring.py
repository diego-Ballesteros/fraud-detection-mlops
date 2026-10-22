"""Monitor the artifact on the production period: performance by window, drift, the HTML report.

Run from the repository root::

    uv run python scripts/run_monitoring.py

The production period (``step >= 401``) is read here, and nothing is fitted. The artifact is
loaded and its sha256 checked against ADR 0010; the threshold is the fixed one
(``performance.OPERATING_THRESHOLD``, ADR 0011). The reference period is used only to
recompute that threshold, as a check that the constant still belongs to this artifact, and
to profile the features for drift. Each calculation prints its wall time.

Steps, in order (ADR 0012):

0. **Identity:** the artifact's sha256 and the rows and steps of both periods.
1. **Threshold check:** the decision metric's point on the reference scores must be the
   constant; otherwise the script stops with exit code 1.
2. **The rule on production,** before any number of the model.
3. **Performance by window and aggregate:** the model at the fixed threshold beside the rule.
4. **The degradation criterion,** with its false-alarm probability under the null at the
   window sizes production actually had.
5. **Drift:** of the target (prevalence) and of the 11 features, PSI and KS, over the whole
   period and window by window. Then what the numbers need to be read: the legitimate rows
   alone (a change in the features, not in the mix of classes), the share of zero balances,
   and PSI's null, its sampling noise at the size of the windows.
6. **Evidently report** to ``reports/deliverables/``, and the cross-check of the two
   instruments; a disagreement where they must agree stops the script with exit code 1.
7. **MLflow:** one run with the aggregate, the per-window series and the tables.

Tables go to ``reports/monitoring/``, which is not versioned.
"""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import mlflow
import numpy as np
import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data.loader import PRODUCTION_PATH, STEP_COLUMN, read_production
from fraud_detection.explain.global_explain import model_features
from fraud_detection.explain.local_explain import operating_point
from fraud_detection.features.builder import SIGNATURE_FEATURE, amount_equals_origin_balance
from fraud_detection.features.pipeline import CONFIGURATIONS
from fraud_detection.models.baseline import SignatureRule
from fraud_detection.models.evaluation import (
    DECISION_PRECISION,
    EVALUATION_COLUMNS,
    derived_seeds,
    features_and_labels,
    load_evaluation_frame,
)
from fraud_detection.models.registry import ARTIFACT_PATH, REGISTERED_MODEL_NAME, load_pipeline
from fraud_detection.models.tracking import file_sha256, git_tags, reference_identity
from fraud_detection.monitoring.drift import (
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    FeatureProfile,
    drift_table,
    hour_profile,
    null_psi,
    psi_level,
    reference_profiles,
    volume_by_window,
    window_drift,
    zero_shares,
)
from fraud_detection.monitoring.performance import (
    ALERT_RUN,
    DEGRADATION_LINE,
    DEGRADATION_RUN,
    OPERATING_THRESHOLD,
    REFERENCE_DISPERSION,
    REFERENCE_RECALL,
    WINDOW_STEPS,
    below_line,
    degradation_status,
    evaluate_rows,
    probability_below,
    probability_of_run,
    runs,
    spread,
    step_windows,
    window_numbers,
    window_table,
)
from fraud_detection.monitoring.report import (
    build_report,
    classification_values,
    drift_values,
    report_frame,
    save_report,
)
from fraud_detection.tracking.mlflow_setup import configure_with_local_fallback

ARTIFACT_SHA256: Final[str] = "41869ee0e5239ca66e0024a10bff4b727cf7ba0a6b6a8087a45da010de5d4ebd"
"""The artifact ADR 0010 registered; the monitoring describes that one and no other."""

OUTPUT_DIR: Final = settings.REPORTS_DIR / "monitoring"
REPORT_PATH: Final = settings.REPORTS_DIR / "deliverables" / "monitoring_report.html"

REFERENCE_RULE: Final[str] = (
    "reference: precision 1.000, recall 0.979 (4,385 of 4,477); temporal folds 0.9788 ± 0.0101"
)
MLFLOW_HTML_LIMIT_BYTES: Final[int] = 50_000_000
"""The HTML is versioned in git already; above this size it is not uploaded to MLflow too."""
NULL_DRAWS: Final[int] = 20
"""Samples drawn from the reference per window size to read PSI's sampling noise."""
NULL_QUANTILE: Final[float] = 0.95
CROSS_CHECK_TOLERANCE: Final[float] = 1e-9
"""Largest gap accepted where Evidently and the project's instruments measure the same thing."""

WINDOW_COLUMNS: Final[tuple[str, ...]] = (
    "first_step",
    "last_step",
    "complete",
    "rows",
    "positives",
    "prevalence",
    "rule_precision",
    "rule_recall",
    "rule_false_negatives",
    "model_precision",
    "model_recall",
    "model_false_negatives",
    "model_false_positives",
    "model_flagged",
    "rule_pr_auc",
    "pr_auc",
    "rule_brier",
    "brier",
)
SPREAD_COLUMNS: Final[tuple[str, ...]] = (
    "prevalence",
    "rule_precision",
    "rule_recall",
    "model_precision",
    "model_recall",
    "model_false_positives",
    "rule_pr_auc",
    "pr_auc",
    "rule_roc_auc",
    "roc_auc",
    "rule_brier",
    "brier",
)


@contextmanager
def timed(name: str) -> Iterator[None]:
    """Print the wall time of the calculation inside the block."""
    started = time.perf_counter()
    yield
    print(f"[time] {name}: {time.perf_counter() - started:.1f} s")


def _fmt(frame: pd.DataFrame) -> str:
    return str(frame.to_string(float_format=lambda value: f"{value:.4f}"))


def main() -> int:
    """Run the eight steps; exit with 1 if a check fails."""
    # The project's standard defense (ERR-012): redirected, stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    print("=== 0. IDENTITY")
    digest = file_sha256(ARTIFACT_PATH)
    print(f"artifact {ARTIFACT_PATH.name} sha256 {digest}")
    if digest != ARTIFACT_SHA256:
        print(f"expected {ARTIFACT_SHA256} (ADR 0010): this is another artifact; stopping.")
        return 1
    pipeline = load_pipeline(ARTIFACT_PATH)
    model = pipeline[-1]
    with timed("load both periods"):
        reference = load_evaluation_frame()
        production = read_production()[list(EVALUATION_COLUMNS)]
    reference_id, production_id = reference_identity(), reference_identity(PRODUCTION_PATH)
    for name, frame, identity in (
        ("reference", reference, reference_id),
        ("production", production, production_id),
    ):
        steps = frame[STEP_COLUMN]
        print(
            f"{name}: {len(frame)} rows, steps {steps.min()}-{steps.max()} "
            f"({steps.nunique()} distinct) | parquet sha256 {identity.sha256[:16]}…"
        )

    print("\n=== 1. THRESHOLD CHECK (reference only)")
    reference_rows, reference_labels = features_and_labels(reference)
    with timed("features and scores of the reference period"):
        reference_features = model_features(pipeline, reference_rows)
        reference_scores = model.predict_proba(reference_features)[:, 1]
    point = operating_point(reference_labels, reference_scores)
    print(
        f"recall at precision >= {DECISION_PRECISION} on reference, in-sample: threshold "
        f"{point.threshold!r}, flags {point.flagged} ({point.true_positives} fraud), "
        f"precision {point.precision:.6f}, recall {point.recall:.6f}"
    )
    if point.threshold != OPERATING_THRESHOLD:
        print(f"the fixed threshold is {OPERATING_THRESHOLD!r}: they differ; stopping.")
        return 1
    print(f"equals the fixed threshold {OPERATING_THRESHOLD!r}: the constant is this artifact's")

    production_rows, labels = features_and_labels(production)
    with timed("features and scores of the production period"):
        production_features = model_features(pipeline, production_rows)
        scores = model.predict_proba(production_features)[:, 1]
    rule_flags = SignatureRule().predict(production_rows)
    steps = production_rows[STEP_COLUMN].to_numpy()
    overall = evaluate_rows(labels, scores, rule_flags)

    print("\n=== 2. THE RULE ON PRODUCTION")
    print(
        f"rule: flags {overall['rule_flagged']:.0f}, of which {overall['rule_true_positives']:.0f} "
        f"fraud and {overall['rule_false_positives']:.0f} legitimate; frauds "
        f"{overall['positives']:.0f}, missed {overall['rule_false_negatives']:.0f} | precision "
        f"{overall['rule_precision']:.6f} recall {overall['rule_recall']:.6f}"
    )
    print(f"({REFERENCE_RULE})")
    signature = amount_equals_origin_balance(production_rows).to_numpy()
    by_type = pd.crosstab(
        production_rows["type"].astype(str), labels, margins=True, margins_name="total"
    )
    with_signature = pd.crosstab(
        production_rows["type"].astype(str)[signature], labels[signature]
    ).reindex(by_type.index[:-1], fill_value=0)
    print("rows by type and label (columns: isFraud), all rows and rows with the signature:")
    print(pd.concat({"all rows": by_type, SIGNATURE_FEATURE: with_signature}, axis=1).to_string())
    fraud = labels == 1
    rule = rule_flags.astype(bool)
    model_flags = scores >= OPERATING_THRESHOLD
    print(
        f"frauds the rule misses: {int((fraud & ~rule).sum())}, of which the model flags "
        f"{int((fraud & ~rule & model_flags).sum())} | frauds the rule catches and the model "
        f"misses: {int((fraud & rule & ~model_flags).sum())}"
    )

    print(f"\n=== 3. PERFORMANCE: {WINDOW_STEPS}-STEP WINDOWS AND AGGREGATE")
    windows = step_windows(settings.PRODUCTION_FIRST_STEP, int(steps.max()))
    table = window_table(steps, labels, scores, rule_flags, windows)
    print(_fmt(table[list(WINDOW_COLUMNS)]))
    print("\naggregate over production:")
    print(
        f"rows {overall['rows']:.0f}, positives {overall['positives']:.0f}, prevalence "
        f"{overall['prevalence']:.6f} (reference {reference_labels.mean():.6f})"
    )
    print(
        f"model at {OPERATING_THRESHOLD:.6f}: flags {overall['model_flagged']:.0f}, "
        f"{overall['model_true_positives']:.0f} fraud, {overall['model_false_positives']:.0f} "
        f"legitimate, misses {overall['model_false_negatives']:.0f} | precision "
        f"{overall['model_precision']:.6f} recall {overall['model_recall']:.6f}"
    )
    print(f"rule: precision {overall['rule_precision']:.6f} recall {overall['rule_recall']:.6f}")
    for name in ("pr_auc", "roc_auc", "brier"):
        print(f"{name}: model {overall[name]:.6f} | rule {overall['rule_' + name]:.6f}")
    summary = spread(table, SPREAD_COLUMNS)
    print("\nspread over the complete windows (sample std, ddof = 1):")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))

    print("\n=== 4. DEGRADATION CRITERION (declared before measuring, ADR 0012)")
    complete = table.loc[table["complete"].astype(bool)]
    below = [
        below_line(int(tp), int(n))
        for tp, n in zip(complete["model_true_positives"], complete["positives"], strict=True)
    ]
    status = degradation_status(below)
    print(
        f"line {float(DEGRADATION_LINE):.4f} = {float(REFERENCE_RECALL):.4f} - "
        f"{float(REFERENCE_DISPERSION):.4f}; alert at {ALERT_RUN} consecutive complete windows "
        f"below, degradation at {DEGRADATION_RUN}; the partial window is out"
    )
    below_windows = [f"w{n}" for n, flag in zip(complete.index, below, strict=True) if flag]
    print(f"below the line: {', '.join(below_windows) or 'none'}")
    found = runs(below)
    print(
        "runs of windows below: "
        + (
            ", ".join(
                f"w{complete.index[start]}-w{complete.index[start + length - 1]} ({length})"
                for start, length in found
            )
            or "none"
        )
    )
    print(f"STATUS: {status}")
    chances = [probability_below(int(n), float(REFERENCE_RECALL)) for n in complete["positives"]]
    null_alert = probability_of_run(chances, ALERT_RUN)
    null_degradation = probability_of_run(chances, DEGRADATION_RUN)
    print(
        f"under the null (true recall {float(REFERENCE_RECALL):.4f} in every window, this "
        f"production's positives per window): P(a window below) "
        f"{min(chances):.3f}-{max(chances):.3f}, P(alert) {null_alert:.3f}, "
        f"P(degradation) {null_degradation:.3f}"
    )
    under_floor = complete.index[complete["model_precision"] < DECISION_PRECISION].tolist()
    print(f"complete windows with model precision under {DECISION_PRECISION}: {under_floor}")

    print("\n=== 5. DRIFT")
    print(
        f"target: prevalence {reference_labels.mean():.6f} in reference, "
        f"{overall['prevalence']:.6f} in production "
        f"({overall['prevalence'] / reference_labels.mean():.2f} times)"
    )
    volume = pd.concat(
        [
            volume_by_window(
                reference_rows[STEP_COLUMN].to_numpy(),
                reference_labels,
                np.full(len(reference_labels), "reference"),
            ),
            volume_by_window(steps, labels, window_numbers(steps, windows)),
        ]
    )
    print("rows per step, by class (the fraud agent against the legitimate volume):")
    print(_fmt(volume))
    hours = pd.concat(
        {
            "reference": hour_profile(reference_rows[STEP_COLUMN].to_numpy(), reference_labels),
            "production": hour_profile(steps, labels),
        },
        axis=1,
    )
    print("\nshare of each class's rows by hour of the day (step % 24):")
    print(_fmt(hours.T))
    money = CONFIGURATIONS["full"].money
    discrete = [name for name in production_features.columns if name not in money]
    with timed("reference profiles of the 11 features"):
        profiles = reference_profiles(reference_features, discrete)
    with timed("drift of the whole production period"):
        whole = drift_table(profiles, production_features)
    print(_fmt(whole))
    with timed("drift window by window"):
        window_psi, window_ks = window_drift(
            profiles, production_features, window_numbers(steps, windows)
        )
    print(
        f"\nPSI by window (cuts by convention: moderate >= {PSI_MODERATE}, "
        f"significant >= {PSI_SIGNIFICANT}):"
    )
    print(_fmt(window_psi))
    print("\nKS distance by window:")
    print(_fmt(window_ks))
    print("\nwindow by window: the criterion beside the features that drift")
    for number in table.index:
        moved = window_psi.loc[number]
        moved = moved[moved >= PSI_MODERATE].sort_values(ascending=False)
        row = table.loc[number]
        flag = (
            below_line(int(row["model_true_positives"]), int(row["positives"]))
            if row["complete"]
            else "partial"
        )
        drifting = ", ".join(
            f"{name} {value:.3f} ({psi_level(value)})" for name, value in moved.items()
        )
        print(
            f"w{number:<2} recall {row['model_recall']:.4f} below {flag!s:<7} | "
            f"{drifting or 'no feature >= 0.1'}"
        )

    print("\nthe legitimate rows alone, against the legitimate reference rows:")
    print("(separates a change in the features from a change in the mix of the two classes)")
    legitimate_reference = reference_labels == 0
    with timed("profiles of the legitimate reference rows"):
        legitimate_profiles = {
            name: FeatureProfile.from_reference(
                name, reference_features[name].to_numpy()[legitimate_reference], name in discrete
            )
            for name in reference_features.columns
        }
    legitimate = labels == 0
    legitimate_psi, _ = window_drift(
        legitimate_profiles,
        production_features.loc[legitimate],
        window_numbers(steps, windows)[legitimate],
    )
    print(_fmt(legitimate_psi))
    zeros = zero_shares(
        production_rows.loc[legitimate], money, window_numbers(steps, windows)[legitimate]
    )
    zeros.loc["reference"] = reference_rows.loc[legitimate_reference, list(money)].eq(0).mean()
    print("\nshare of zeros in the money columns, legitimate rows:")
    print(_fmt(zeros))

    sizes = {
        "smallest complete window": int(complete["rows"].min()),
        "partial window": int(table.loc[~table["complete"].astype(bool), "rows"].min()),
    }
    seeds = derived_seeds(NULL_DRAWS)
    with timed(f"null PSI, {NULL_DRAWS} draws from reference per size"):
        null = pd.DataFrame(
            {
                f"{label} ({size} rows)": {
                    name: float(
                        np.quantile(
                            null_psi(profile, reference_features[name].to_numpy(), size, seeds),
                            NULL_QUANTILE,
                        )
                    )
                    for name, profile in profiles.items()
                }
                for label, size in sizes.items()
            }
        )
    print(
        f"\nnull PSI: percentile {NULL_QUANTILE:.0%} of {NULL_DRAWS} samples drawn from the "
        f"reference itself, seeds {seeds[0]}..{seeds[-1]} (pure sampling noise at that size):"
    )
    print(_fmt(null))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "windows.csv")
    summary.to_csv(OUTPUT_DIR / "windows_spread.csv")
    whole.to_csv(OUTPUT_DIR / "drift_features.csv")
    window_psi.to_csv(OUTPUT_DIR / "drift_windows_psi.csv")
    window_ks.to_csv(OUTPUT_DIR / "drift_windows_ks.csv")
    legitimate_psi.to_csv(OUTPUT_DIR / "drift_windows_psi_legitimate.csv")
    zeros.to_csv(OUTPUT_DIR / "zero_shares_legitimate.csv")
    null.to_csv(OUTPUT_DIR / "null_psi.csv")
    volume.to_csv(OUTPUT_DIR / "volume_by_window.csv")
    hours.to_csv(OUTPUT_DIR / "hour_profile.csv")

    print("\n=== 6. EVIDENTLY REPORT AND CROSS-CHECK")
    features = list(production_features.columns)
    with timed("report frames"):
        reference_frame = report_frame(reference_features, reference_labels, reference_scores)
        del reference_features
        production_frame = report_frame(production_features, labels, scores)
    with timed("Evidently report"):
        snapshot = build_report(reference_frame, production_frame, features, discrete)
    with timed("save the HTML"):
        size = save_report(snapshot, REPORT_PATH)
    print(f"{REPORT_PATH.relative_to(settings.ROOT_DIR)}: {size / 1e6:.1f} MB")
    theirs = drift_values(snapshot).reindex(whole.index)
    check = pd.DataFrame({"kind": whole["kind"], "ours": whole["psi"], "evidently": theirs})
    check["gap"] = (check["ours"] - check["evidently"]).abs()
    print(_fmt(check))
    discrete_gap = float(check.loc[discrete, "gap"].max())
    measured = classification_values(snapshot)
    performance_gap = max(
        abs(measured["precision"] - overall["model_precision"]),
        abs(measured["recall"] - overall["model_recall"]),
    )
    print(
        f"largest PSI gap on the discrete features {discrete_gap:.2e}; Evidently precision "
        f"{measured['precision']:.6f} recall {measured['recall']:.6f}, gap {performance_gap:.2e} "
        f"(tolerance {CROSS_CHECK_TOLERANCE})"
    )
    check.to_csv(OUTPUT_DIR / "evidently_cross_check.csv")
    if max(discrete_gap, performance_gap) > CROSS_CHECK_TOLERANCE:
        print("the two instruments disagree where they must agree; stopping before MLflow.")
        return 1

    print("\n=== 7. MLFLOW")
    target, failure = configure_with_local_fallback()
    if failure is not None:
        print("Remote tracking failed; using the local store. Full error:\n" + failure)
    print(f"tracking URI : {target.tracking_uri} | experiment {target.experiment_name}")
    tags = {
        **reference_id.tags(),
        "data.production_rows": str(production_id.rows),
        "data.production_steps": f"{production_id.first_step}-{production_id.last_step}",
        "data.production_sha256": production_id.sha256,
        **git_tags(),
        "phase": "4",
        "purpose": "monitoring",
        "decision": "ADR 0012",
        "model": REGISTERED_MODEL_NAME,
        "artifact_sha256": digest,
        "degradation_status": status,
    }
    params = {
        "threshold": repr(OPERATING_THRESHOLD),
        "window_steps": WINDOW_STEPS,
        "windows": len(windows),
        "degradation_line": float(DEGRADATION_LINE),
        "alert_run": ALERT_RUN,
        "degradation_run": DEGRADATION_RUN,
        "psi_moderate": PSI_MODERATE,
        "psi_significant": PSI_SIGNIFICANT,
    }
    with mlflow.start_run(run_name="monitoring__production") as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics({f"production_{name}": value for name, value in overall.items()})
        mlflow.log_metrics(
            {
                "null_p_alert": null_alert,
                "null_p_degradation": null_degradation,
                **{f"psi_{name}": value for name, value in whole["psi"].items()},
                **{f"ks_{name}": value for name, value in whole["ks"].items()},
                **{f"spread_{name}_mean": value for name, value in summary["mean"].items()},
                **{f"spread_{name}_std": value for name, value in summary["std"].items()},
            }
        )
        for number in table.index:
            row = table.loc[number]
            values = {
                f"window_{name}": float(row[name])
                for name in WINDOW_COLUMNS
                if name not in ("first_step", "last_step") and pd.notna(row[name])
            }
            values |= {
                f"window_psi_{name}": value for name, value in window_psi.loc[number].items()
            }
            values |= {f"window_ks_{name}": value for name, value in window_ks.loc[number].items()}
            mlflow.log_metrics(values, step=int(number))
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="monitoring")
        if size <= MLFLOW_HTML_LIMIT_BYTES:
            mlflow.log_artifact(str(REPORT_PATH), artifact_path="deliverables")
        else:
            print(
                f"HTML of {size / 1e6:.1f} MB not uploaded "
                f"(limit {MLFLOW_HTML_LIMIT_BYTES / 1e6:.0f} MB)"
            )
        print(f"run {run.info.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
