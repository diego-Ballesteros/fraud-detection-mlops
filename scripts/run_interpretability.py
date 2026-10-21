"""Explain the one serialized artifact, globally and locally, starting with the gate.

Run from the repository root::

    uv run python scripts/run_interpretability.py

Everything is measured on the loaded artifact (``models.registry.load_pipeline``); nothing is
refitted. Only the reference period is read. Each calculation prints its wall time. Figures
and tables go to ``reports/interpretability/``, which is not versioned.

Steps, in order (ADR 0011):

0. **Gate.** SHAP must find the simulator's signature on the declared sample. If the three
   signature features do not hold the majority of the mean |SHAP|, or the flag is not among
   the first three, the script stops before producing any figure, with exit code 1: the
   problem would be the explainer, not the model.
1. **SHAP summary** of the sample: the beeswarm and the mean |SHAP| table.
2. **Scores** of the artifact on every reference row, the input of steps 3 and 5.
3. **Permutation importance** on the decision metric, over every reference row, at the level
   of the model features, and the **three orderings** side by side: SHAP, permutation and the
   model's own importance (split and gain).
4. **ICE and PDP** of three raw columns, through the whole artifact, by rule group.
5. **Local cases:** the operating threshold, the four cases of ``local_explain`` and their
   waterfalls.
"""

import sys
import time
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Files only: no window opens, whatever the platform's default.

import matplotlib.pyplot as plt

from fraud_detection.config import settings
from fraud_detection.data.schema import TARGET_COLUMN
from fraud_detection.explain.global_explain import (
    EXPLANATION_NEGATIVE_ROWS,
    GATE_FLAG_MAX_RANK,
    GATE_MAJORITY,
    ICE_GRID_POINTS,
    ICE_GRID_QUANTILES,
    PERMUTATION_REPEATS,
    SIGNATURE_FEATURES,
    compare_rankings,
    explanation_sample,
    ice_curves,
    mean_abs_shap,
    model_features,
    model_importance,
    outside_grid,
    permutation_importance,
    plot_beeswarm,
    plot_ice,
    raw_grid,
    rule_groups,
    shap_values,
    signature_gate,
    trees_splitting_on,
)
from fraud_detection.explain.local_explain import (
    CASE_TITLES,
    display_values,
    operating_point,
    plot_waterfall,
    select_cases,
    top_contributions,
)
from fraud_detection.models.baseline import SignatureRule
from fraud_detection.models.evaluation import (
    DECISION_PRECISION,
    derived_seeds,
    features_and_labels,
    load_evaluation_frame,
)
from fraud_detection.models.registry import (
    ARTIFACT_PATH,
    MODEL_INPUT_COLUMNS,
    load_pipeline,
)

OUTPUT_DIR: Final = settings.REPORTS_DIR / "interpretability"
ICE_COLUMNS: Final[tuple[str, ...]] = ("oldbalanceOrg", "oldbalanceDest", "amount")
"""The two raw columns among the first three model features by mean |SHAP|, and the flag's
other input; the flag itself is not an input and has no curve (ADR 0011)."""
ICE_CURVES_PER_GROUP: Final[int] = 100


class Clock:
    """Print the wall time of each named calculation."""

    def __init__(self) -> None:
        """Start with no calculation timed."""
        self.started = time.perf_counter()
        self.name = ""

    def start(self, name: str) -> None:
        """Start timing a calculation.

        Args:
            name: What is being calculated.
        """
        self.name, self.started = name, time.perf_counter()

    def stop(self) -> float:
        """Print and return the seconds since ``start``."""
        seconds = time.perf_counter() - self.started
        print(f"[time] {self.name}: {seconds:.1f} s")
        return seconds


def _save(figure: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"figure: {path.relative_to(settings.ROOT_DIR)}")


def main() -> int:
    """Run the six steps; stop with exit code 1 if the gate fails."""
    # The project's standard defense (ERR-012): redirected, stdout on Windows is cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    clock = Clock()
    pipeline = load_pipeline(ARTIFACT_PATH)
    model = pipeline[-1]
    clock.start("load the reference period")
    reference = load_evaluation_frame()
    clock.stop()
    print(f"artifact: {ARTIFACT_PATH.name} | reference rows {len(reference)}")

    print("\n=== 0. GATE: SHAP MUST FIND THE SIGNATURE")
    sample = explanation_sample(reference)
    positives = int(sample[TARGET_COLUMN].sum())
    print(
        f"sample: {len(sample)} rows = {positives} positives (all of reference) + "
        f"{len(sample) - positives} negatives (declared {EXPLANATION_NEGATIVE_ROWS}), "
        f"seed {settings.RANDOM_STATE}"
    )
    sample_rows = sample.drop(columns=[TARGET_COLUMN])
    clock.start("SHAP values of the sample (TreeExplainer, tree_path_dependent)")
    result = shap_values(pipeline, sample_rows)
    clock.stop()
    print(
        f"base value {result.base_value:.4f} log-odds | "
        f"max additivity gap {result.max_additivity_gap:.2e}"
    )
    table = mean_abs_shap(result.values)
    total = float(table.sum())
    print(f"\n{'rank':>4}  {'feature':<30} {'mean |SHAP|':>12} {'share':>7}")
    for rank, (name, value) in enumerate(table.items(), start=1):
        mark = "  <- signature" if name in SIGNATURE_FEATURES else ""
        print(f"{rank:>4}  {name:<30} {value:>12.4f} {value / total:>7.1%}{mark}")
    gate = signature_gate(table)
    print(
        f"\nsignature share {gate.signature_share:.1%} (must be > {GATE_MAJORITY:.0%}) | "
        f"flag rank {gate.flag_rank} (must be <= {GATE_FLAG_MAX_RANK}) | "
        f"GATE {'PASSED' if gate.passed else 'FAILED'}"
    )
    if not gate.passed:
        print("The explainer did not find the known answer: stopping before any figure.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== 1. SHAP SUMMARY")
    table.rename("mean_abs_shap").to_csv(OUTPUT_DIR / "mean_abs_shap.csv")
    _save(plot_beeswarm(result), "shap_beeswarm.png")

    print("\n=== 2. SCORES ON EVERY REFERENCE ROW")
    raw_rows, labels = features_and_labels(reference)
    clock.start("model features of the reference period")
    features = model_features(pipeline, raw_rows)
    clock.stop()
    clock.start("scores of the reference period")
    scores = pd.Series(model.predict_proba(features)[:, 1], index=raw_rows.index)
    clock.stop()
    from_shap = 1 / (1 + np.exp(-result.raw_output))
    print(
        "sample scores equal the logistic of the SHAP raw output: "
        f"{np.allclose(scores.loc[sample.index], from_shap, rtol=0, atol=1e-12)}"
    )

    print("\n=== 3. PERMUTATION IMPORTANCE AND THE THREE ORDERINGS")
    seeds = derived_seeds(PERMUTATION_REPEATS)
    clock.start(f"permutation importance, {len(features.columns)} features x {len(seeds)} seeds")
    permutation = permutation_importance(model, features, labels, seeds)
    clock.stop()
    print(
        f"decision metric, unpermuted, in-sample on {len(labels)} rows: "
        f"{permutation.baseline:.6f} | seeds {seeds[0]}..{seeds[-1]}"
    )
    summary = permutation.summary()
    print(summary.to_string(float_format=lambda v: f"{v:.6f}"))
    permutation.drops.to_csv(OUTPUT_DIR / "permutation_drops.csv")
    importance = model_importance(pipeline)
    orderings = compare_rankings(
        {
            "shap": table,
            "permutation": summary["mean"],
            "split": importance["split"],
            "gain": importance["gain"],
        }
    )
    print()
    print(orderings.to_string(float_format=lambda v: f"{v:.4g}"))
    orderings.to_csv(OUTPUT_DIR / "orderings.csv")
    del features
    flag = SIGNATURE_FEATURES[0]
    print(f"\nsplit nodes on {flag} (left: value <= threshold, i.e. False):")
    print(trees_splitting_on(pipeline, flag).to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== 4. ICE AND PDP ON RAW COLUMNS")
    rule_flags = pd.Series(SignatureRule().predict(raw_rows).astype(bool), index=raw_rows.index)
    groups = rule_groups(sample[TARGET_COLUMN], rule_flags.loc[sample.index])
    print(groups.value_counts().to_string())
    curves = {}
    for column in ICE_COLUMNS:
        grid = raw_grid(raw_rows[column])
        clock.start(f"ICE of {column}: {len(sample_rows)} rows x {len(grid)} grid points")
        curves[column] = ice_curves(pipeline, sample_rows, column, grid)
        clock.stop()
        pdp = curves[column].groupby(groups).mean()
        pdp.loc["whole sample"] = curves[column].mean()
        print(
            f"{column}: grid {len(grid)} points ({ICE_GRID_POINTS} quantiles "
            f"{ICE_GRID_QUANTILES[0]}-{ICE_GRID_QUANTILES[1]} of reference), "
            f"{grid[0]:.2f} to {grid[-1]:.2f}; PDP min / max / range by group, log-odds:"
        )
        for name, row in pdp.iterrows():
            print(
                f"    {name:<32} {row.min():>8.3f} {row.max():>8.3f} {row.max() - row.min():>8.3f}"
            )
        spread = curves[column].max(axis=1) - curves[column].min(axis=1)
        by_group = spread.groupby(groups).describe(percentiles=[0.5])
        print(f"    per-row ICE range (max - min) by group:\n{by_group.to_string()}")
        outside = outside_grid(sample_rows[column], grid).groupby(groups).agg(["sum", "size"])
        print(f"    rows whose own {column} lies outside the grid, by group:")
        for name, row in outside.iterrows():
            print(f"    {name:<32} {int(row['sum']):>6} of {int(row['size'])}")
    _save(plot_ice(curves, groups, ICE_CURVES_PER_GROUP), "ice_pdp.png")

    print("\n=== 5. LOCAL CASES")
    point = operating_point(labels, scores.to_numpy())
    flagged = scores >= point.threshold
    fraud = pd.Series(labels == 1, index=raw_rows.index)
    print(
        f"threshold {point.threshold!r} (precision >= {DECISION_PRECISION}, in-sample): "
        f"flags {point.flagged}, of which {point.true_positives} fraud and "
        f"{point.flagged - point.true_positives} legitimate | precision {point.precision:.6f} "
        f"recall {point.recall:.6f}"
    )
    print(
        f"frauds the rule covers: {int((fraud & rule_flags).sum())}, the model flags "
        f"{int((fraud & rule_flags & flagged).sum())} | frauds the rule misses: "
        f"{int((fraud & ~rule_flags).sum())}, the model flags "
        f"{int((fraud & ~rule_flags & flagged).sum())} | legitimate flagged: reference "
        f"{int((~fraud & flagged).sum())}, sample {int((~fraud & flagged).loc[sample.index].sum())}"
    )
    cases = select_cases(
        sample[TARGET_COLUMN],
        scores.loc[sample.index],
        rule_flags.loc[sample.index],
        point.threshold,
    )
    rows = []
    for case, row_id in cases.items():
        print(f"\n--- {CASE_TITLES[case]}")
        if row_id is None:
            print("no row of the sample falls in this case")
            continue
        raw = sample_rows.loc[row_id]
        values = result.values.loc[row_id]
        top = top_contributions(values)
        inputs = ", ".join(f"{name}={raw[name]}" for name in MODEL_INPUT_COLUMNS)
        print(f"row {row_id} (position in reference.parquet) | {inputs}")
        print(
            f"label {int(sample.loc[row_id, TARGET_COLUMN])} | rule {bool(rule_flags[row_id])} | "
            f"probability {scores[row_id]:.6f} | log-odds {result.raw_output[row_id]:.4f} | "
            f"threshold {point.threshold:.6f}"
        )
        print("top-3 SHAP: " + ", ".join(f"{name} {value:+.4f}" for name, value in top.items()))
        rows.append(
            {
                "case": case,
                "row": row_id,
                **{name: raw[name] for name in MODEL_INPUT_COLUMNS},
                "label": int(sample.loc[row_id, TARGET_COLUMN]),
                "rule": bool(rule_flags[row_id]),
                "probability": scores[row_id],
                "log_odds": result.raw_output[row_id],
                **{f"top{i}": f"{n} {v:+.4f}" for i, (n, v) in enumerate(top.items(), start=1)},
            }
        )
        features_row = result.features.loc[row_id]
        figure = plot_waterfall(
            values,
            result.base_value,
            display_values(raw, features_row),
            features_row,
            f"{CASE_TITLES[case]} (fila {row_id})",
        )
        _save(figure, f"waterfall_{case}.png")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "cases.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
