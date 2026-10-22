"""Tests of fraud_detection.monitoring.performance, with expected values computed by hand."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from fraud_detection.monitoring.performance import (
    ALERT,
    DEGRADATION,
    DEGRADATION_LINE,
    NO_ALERT,
    OPERATING_THRESHOLD,
    Window,
    below_line,
    confusion,
    degradation_status,
    evaluate_rows,
    lowering_cost,
    probability_below,
    probability_of_run,
    runs,
    segment_table,
    spread,
    step_windows,
    window_numbers,
    window_table,
)

# Six rows. Threshold 0.5: the model flags 0.9, 0.5, 0.6 and 0.5 (a score equal to the
# threshold is flagged) -> flagged 4, of which 2 fraud. The rule flags only the first fraud.
LABELS = np.array([1, 1, 1, 0, 0, 0])
SCORES = np.array([0.9, 0.5, 0.1, 0.6, 0.2, 0.5])
RULE = np.array([1, 0, 0, 0, 0, 0])


def test_the_fixed_threshold_is_the_one_documented_in_phase_3() -> None:
    # docs/analysis/interpretability.md, section 4: chosen in-sample on reference.
    assert OPERATING_THRESHOLD == 0.42775088540908546


def test_the_degradation_line_is_exact() -> None:
    # 0.9970 - 0.0036 in floats is 0.9934000000000001; the line is the exact fraction.
    assert Fraction(4967, 5000) == DEGRADATION_LINE


def test_production_cuts_into_fourteen_days_and_a_partial_window() -> None:
    windows = step_windows(401, 743)
    assert len(windows) == 15
    assert windows[0] == Window(1, 401, 424, complete=True)
    assert windows[13] == Window(14, 713, 736, complete=True)
    assert windows[14] == Window(15, 737, 743, complete=False)
    assert windows[14].steps == 7
    assert sum(window.steps for window in windows) == 343


def test_an_exact_multiple_leaves_every_window_complete() -> None:
    windows = step_windows(1, 48)
    assert [(w.first_step, w.last_step, w.complete) for w in windows] == [
        (1, 24, True),
        (25, 48, True),
    ]
    with pytest.raises(ValueError, match="Cannot cut"):
        step_windows(10, 9)


def test_window_numbers_by_hand_and_outside_the_windows() -> None:
    windows = step_windows(401, 743)
    assert window_numbers(np.array([401, 424, 425, 736, 737, 743]), windows).tolist() == [
        1,
        1,
        2,
        14,
        15,
        15,
    ]
    with pytest.raises(ValueError, match="outside"):
        window_numbers(np.array([400, 401]), windows)


def test_confusion_by_hand() -> None:
    counts = confusion(LABELS, SCORES >= 0.5)
    assert (counts.rows, counts.positives, counts.flagged, counts.true_positives) == (6, 3, 4, 2)
    assert (counts.false_positives, counts.false_negatives, counts.true_negatives) == (2, 1, 1)
    assert counts.precision == 0.5
    assert counts.recall == pytest.approx(2 / 3)
    empty = confusion(np.array([0, 0]), np.array([0, 0]))
    assert math.isnan(empty.precision)
    assert math.isnan(empty.recall)


def test_evaluate_rows_by_hand() -> None:
    metrics = evaluate_rows(LABELS, SCORES, RULE, threshold=0.5)
    assert metrics["rows"] == 6
    assert metrics["positives"] == 3
    assert metrics["prevalence"] == 0.5
    assert (metrics["model_flagged"], metrics["model_false_positives"]) == (4, 2)
    assert metrics["model_precision"] == 0.5
    assert metrics["model_recall"] == pytest.approx(2 / 3)
    assert (metrics["rule_precision"], metrics["rule_recall"]) == (1.0, pytest.approx(1 / 3))
    # Average precision: recall steps of 1/3 at precision 1, 1/2 and 1/2.
    assert metrics["pr_auc"] == pytest.approx(2 / 3)
    # Pairs (positive, negative): 0.9 beats 3, 0.5 beats 1 and ties 1, 0.1 beats none.
    assert metrics["roc_auc"] == pytest.approx(4.5 / 9)
    assert metrics["brier"] == pytest.approx((0.01 + 0.25 + 0.81 + 0.36 + 0.04 + 0.25) / 6)


def test_the_rule_gets_the_same_context_metrics_from_its_flags() -> None:
    metrics = evaluate_rows(LABELS, SCORES, RULE, threshold=0.5)
    # Flags as scores: precision 1 at recall 1/3, then 3/6 at recall 1.
    assert metrics["rule_pr_auc"] == pytest.approx(1 / 3 * 1 + 2 / 3 * 0.5)
    # The flagged fraud beats the 3 negatives; the 2 unflagged frauds tie with them.
    assert metrics["rule_roc_auc"] == pytest.approx((3 + 2 * 3 * 0.5) / 9)
    # Two frauds missed, each costing (0 - 1)^2.
    assert metrics["rule_brier"] == pytest.approx(2 / 6)


def test_a_single_class_leaves_the_rankings_undefined() -> None:
    metrics = evaluate_rows(np.zeros(3), np.array([0.1, 0.2, 0.3]), np.zeros(3), threshold=0.5)
    assert math.isnan(metrics["pr_auc"])
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["model_precision"])
    assert math.isnan(metrics["model_recall"])
    assert metrics["brier"] == pytest.approx((0.01 + 0.04 + 0.09) / 3)


def test_window_table_evaluates_each_window_on_its_own_rows() -> None:
    windows = step_windows(1, 5, size=2)  # 1-2, 3-4, and 5 alone (partial)
    steps = np.array([1, 2, 3, 4, 5, 5])
    table = window_table(steps, LABELS, SCORES, RULE, windows, threshold=0.5)
    assert table.index.tolist() == [1, 2, 3]
    assert table["complete"].tolist() == [True, True, False]
    assert table["rows"].tolist() == [2, 2, 2]
    # Window 1: two frauds, both flagged. Window 2: a missed fraud and a false positive.
    assert table.loc[1, "model_recall"] == 1.0
    assert (table.loc[2, "model_recall"], table.loc[2, "model_precision"]) == (0.0, 0.0)
    assert table.loc[3, "model_false_positives"] == 1


def test_segment_table_by_hand_keeps_the_undefined_as_nan() -> None:
    segments = np.array(["B", "B", "A", "A", "C", "C"])
    table = segment_table(segments, LABELS, SCORES, RULE, threshold=0.5)
    assert table.index.tolist() == ["A", "B", "C"]
    # B: frauds 0.9 and 0.5, both flagged; the rule flags only the first.
    assert (table.loc["B", "model_precision"], table.loc["B", "model_recall"]) == (1.0, 1.0)
    assert (table.loc["B", "rule_precision"], table.loc["B", "rule_recall"]) == (1.0, 0.5)
    # A: the fraud at 0.1 is missed and the legitimate 0.6 is flagged; the rule flags nothing.
    assert (table.loc["A", "model_precision"], table.loc["A", "model_recall"]) == (0.0, 0.0)
    assert math.isnan(table.loc["A", "rule_precision"])
    assert table.loc["A", "rule_recall"] == 0.0
    # C: no fraud. The model's false positive makes its precision 0, and its recall is not
    # perfect but undefined; the rule flags nothing there, so both of its rates are undefined.
    assert table.loc["C", "model_false_positives"] == 1
    assert table.loc["C", "model_precision"] == 0.0
    assert math.isnan(table.loc["C", "model_recall"])
    assert math.isnan(table.loc["C", "rule_precision"])
    assert math.isnan(table.loc["C", "rule_recall"])
    assert table["rows"].sum() == len(LABELS)
    with pytest.raises(ValueError, match="differ in length"):
        segment_table(segments[:5], LABELS, SCORES, RULE)


def test_lowering_cost_by_hand() -> None:
    # At 0.55 the model flags 0.9 (fraud) and 0.6 (legitimate); the frauds at 0.5 and 0.1 are
    # missed. Down to 0.5: 0.5 (fraud) and 0.5 (legitimate) join. Down to 0.1: all six.
    cost = lowering_cost(LABELS, SCORES, threshold=0.55)
    assert cost["score"].tolist() == [0.5, 0.1]
    assert cost["positives_gained"].tolist() == [1, 2]
    assert cost["negatives_added"].tolist() == [1, 2]
    assert cost["recall"].tolist() == pytest.approx([2 / 3, 1.0])
    assert cost["precision"].tolist() == [0.5, 0.5]


def test_lowering_cost_gains_tied_positives_together_and_can_be_empty() -> None:
    # Nothing flagged at 0.5; the two frauds share 0.3, so one row gains both, with the
    # legitimate 0.4 above them.
    cost = lowering_cost(np.array([1, 1, 0]), np.array([0.3, 0.3, 0.4]), threshold=0.5)
    assert cost[["positives_gained", "negatives_added"]].to_numpy().tolist() == [[2, 1]]
    assert cost.loc[0, "precision"] == pytest.approx(2 / 3)
    # Every fraud already flagged: no price to pay, and the columns are still there.
    empty = lowering_cost(LABELS, SCORES, threshold=0.05)
    assert empty.empty
    assert empty.columns.tolist() == [
        "score",
        "positives_gained",
        "negatives_added",
        "recall",
        "precision",
    ]
    with pytest.raises(ValueError, match="differ in shape"):
        lowering_cost(LABELS[:5], SCORES)


def test_spread_reads_only_the_complete_windows() -> None:
    table = pd.DataFrame({"complete": [True, True, False], "x": [1.0, 3.0, 100.0]})
    summary = spread(table, ["x"])
    assert summary.loc["x", "mean"] == 2.0
    assert summary.loc["x", "std"] == pytest.approx(math.sqrt(2))
    assert summary.loc["x", "windows"] == 2


def test_below_line_is_strict_and_exact() -> None:
    # 4967 / 5000 is exactly the line: not below. One fraud fewer is.
    assert below_line(4967, 5000) is False
    assert below_line(4966, 5000) is True
    # 299 / 301 = 0.99336..., below 0.9934 by less than the float error of 0.9970 - 0.0036.
    assert below_line(299, 301) is True
    assert below_line(300, 301) is False
    assert below_line(0, 0) is None


def test_runs_and_the_two_levels() -> None:
    assert runs([True, True, False, True, True, True, None, True]) == [(0, 2), (3, 3), (7, 1)]
    assert degradation_status([True, False, True, False]) == NO_ALERT
    assert degradation_status([False, True, True, False]) == ALERT
    assert degradation_status([False, True, True, True]) == DEGRADATION
    # A window without positives says nothing, and it breaks a run.
    assert degradation_status([True, None, True]) == NO_ALERT
    assert degradation_status([]) == NO_ALERT


def test_probability_of_run_by_hand() -> None:
    assert probability_of_run([0.5] * 3, 2) == pytest.approx(3 / 8)  # BBx, xBB without BBB twice
    assert probability_of_run([0.5] * 3, 3) == pytest.approx(1 / 8)
    assert probability_of_run([0.5] * 3, 1) == pytest.approx(7 / 8)
    # B1 B2, or not B1 then B2 B3: 0.2 * 0.5 + 0.8 * 0.5 * 0.4.
    assert probability_of_run([0.2, 0.5, 0.4], 2) == pytest.approx(0.26)


def test_probability_of_run_matches_enumeration() -> None:
    probabilities = np.random.default_rng(0).uniform(0, 1, 8)
    for length in (1, 2, 3, 4):
        expected = 0.0
        for outcome in itertools.product([False, True], repeat=len(probabilities)):
            chance = np.prod(
                [p if hit else 1 - p for p, hit in zip(probabilities, outcome, strict=True)]
            )
            if any(length <= n for _, n in runs(list(outcome))):
                expected += chance
        assert probability_of_run(probabilities, length) == pytest.approx(expected), length


def test_probability_below_by_hand_and_against_the_binomial() -> None:
    # One positive: below only if it is missed. Two: below if at most one is caught
    # (2 * 4967 - 1) // 5000 = 1.
    assert probability_below(1, 0.9) == pytest.approx(0.1)
    assert probability_below(2, 0.9) == pytest.approx(1 - 0.81)
    assert probability_below(0, 0.9) == 0.0
    n, recall = 300, 0.997
    expected = sum(
        math.comb(n, k) * recall**k * (1 - recall) ** (n - k)
        for k in range(n + 1)
        if k * DEGRADATION_LINE.denominator < n * DEGRADATION_LINE.numerator
    )
    assert probability_below(n, recall) == pytest.approx(expected, rel=1e-9)
