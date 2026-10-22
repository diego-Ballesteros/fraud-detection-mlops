"""Performance on the production period, window by window, at a fixed threshold, beside the rule.

**Production is measured here for the first time.** The model, the preprocessor and the
threshold were fitted without seeing one production row (CLAUDE.md), so what this module
measures is an honest estimate on a period the artifact never saw, not a self-evaluation.
Nothing here fits anything: every function receives the scores and flags it evaluates.

**The threshold is fixed** at ``OPERATING_THRESHOLD``, the point the decision metric picked
in-sample on the reference period (ADR 0011). It is never re-optimized on production.
Choosing it again with production labels would make the threshold a statistic fitted on the
very rows it is then judged on: the precision and recall measured at it would be a
self-evaluation, and the temporal cut would protect nothing. A monitoring that moves the
threshold to where production looks best cannot see degradation, by construction.

**Windows** have ``WINDOW_STEPS`` steps, anchored at the first production step. Any 24
consecutive steps hold each hour of the day exactly once, so every complete window has the
same mix of hours whatever its first step. A last window shorter than that is reported,
marked partial, and stays out of the degradation criterion; its rows count in the aggregate.

**The baseline goes beside the model, always:** ``SignatureRule`` on the same rows.

**Segments** split the same rows by a column, such as the transaction type, because an
aggregate hides a segment with no positives: there the model cannot miss a fraud, so its
recall is undefined, not perfect, and the table keeps it as NaN.

**The price of the next frauds** is read off the scores below the threshold, never applied:
``lowering_cost`` says what lowering the threshold to each missed positive would add to the
queue. It is a price list for a decision that belongs to whoever pays the reviews, not a way
to pick a threshold on production.

**Degradation criterion,** declared before measuring production (ADR 0012). The model's
recall at the fixed threshold falls below ``DEGRADATION_LINE`` in a complete window when it
drops more than ``REFERENCE_DISPERSION`` from ``REFERENCE_RECALL``, the mean and spread of the
decision metric over the five temporal folds of the reference period (EVAL-009, EVAL-010).
``ALERT_RUN`` consecutive complete windows below the line are an alert; ``DEGRADATION_RUN``
are degradation. One window alone is noise. Precision is not part of the criterion: at a
fixed threshold it moves with the prevalence, and it is read against the 0.99 floor instead.
The comparison is exact, in integers, like ``evaluation.recall_at_precision``.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

OPERATING_THRESHOLD: Final[float] = 0.42775088540908546
"""The operating threshold: recall at precision >= 0.99 on the 5,787,030 reference rows,
in-sample (ADR 0011, docs/analysis/interpretability.md). Fixed: never re-optimized on
production, for the reason in the module docstring."""

WINDOW_STEPS: Final[int] = 24
"""Steps per window: one simulated day."""

REFERENCE_RECALL: Final[Fraction] = Fraction("0.9970")
"""Mean decision metric of the chosen configuration over the five temporal folds (EVAL-009)."""

REFERENCE_DISPERSION: Final[Fraction] = Fraction("0.0036")
"""Sample standard deviation of that metric across the same folds (EVAL-009, EVAL-010)."""

DEGRADATION_LINE: Final[Fraction] = REFERENCE_RECALL - REFERENCE_DISPERSION
"""0.9934: a complete window whose recall is strictly below this has dropped more than the
reference dispersion."""

ALERT_RUN: Final[int] = 2
"""Consecutive complete windows below the line that make an alert."""

DEGRADATION_RUN: Final[int] = 3
"""Consecutive complete windows below the line that make degradation."""

NO_ALERT: Final[str] = "none"
ALERT: Final[str] = "alert"
DEGRADATION: Final[str] = "degradation"


# --- Windows ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A closed range of steps of the production period.

    Attributes:
        number: 1 for the first window.
        first_step: First step it holds.
        last_step: Last step it holds.
        complete: Whether it holds the full number of steps of a window.
    """

    number: int
    first_step: int
    last_step: int
    complete: bool

    @property
    def steps(self) -> int:
        """Number of steps the window holds."""
        return self.last_step - self.first_step + 1


def step_windows(first_step: int, last_step: int, size: int = WINDOW_STEPS) -> tuple[Window, ...]:
    """Cut a range of steps into consecutive windows, anchored at its first step.

    Args:
        first_step: First step of the range.
        last_step: Last step of the range.
        size: Steps per window.

    Returns:
        The windows in order. Only the last one can be shorter, and then it is not complete.

    Raises:
        ValueError: If the range is empty or the size is not positive.
    """
    if size < 1 or last_step < first_step:
        raise ValueError(f"Cannot cut steps {first_step}..{last_step} into windows of {size}")
    windows = []
    for number, start in enumerate(range(first_step, last_step + 1, size), start=1):
        end = min(start + size - 1, last_step)
        windows.append(Window(number, start, end, complete=end - start + 1 == size))
    return tuple(windows)


def window_numbers(steps: np.ndarray, windows: Sequence[Window]) -> np.ndarray:
    """The window each row falls in.

    Args:
        steps: The step of each row.
        windows: Consecutive windows from ``step_windows``.

    Returns:
        The window number of each row.

    Raises:
        ValueError: If a step falls outside every window.
    """
    steps = np.asarray(steps, dtype=np.int64)
    first, last = windows[0].first_step, windows[-1].last_step
    outside = (steps < first) | (steps > last)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} rows fall outside steps {first}..{last}")
    starts = np.asarray([window.first_step for window in windows], dtype=np.int64)
    return np.searchsorted(starts, steps, side="right").astype(np.int64)


# --- Metrics ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Confusion:
    """What a set of 0/1 flags catches on some rows.

    Attributes:
        rows: Rows evaluated.
        positives: Positive rows.
        flagged: Rows flagged.
        true_positives: Flagged rows that are positive.
    """

    rows: int
    positives: int
    flagged: int
    true_positives: int

    @property
    def false_positives(self) -> int:
        """Flagged rows that are legitimate: each one is a review that finds nothing."""
        return self.flagged - self.true_positives

    @property
    def false_negatives(self) -> int:
        """Positive rows not flagged."""
        return self.positives - self.true_positives

    @property
    def true_negatives(self) -> int:
        """Legitimate rows not flagged."""
        return self.rows - self.positives - self.false_positives

    @property
    def precision(self) -> float:
        """Share of flagged rows that are positive; NaN when nothing is flagged."""
        return self.true_positives / self.flagged if self.flagged else float("nan")

    @property
    def recall(self) -> float:
        """Share of positive rows that are flagged; NaN when there is no positive."""
        return self.true_positives / self.positives if self.positives else float("nan")


def confusion(labels: np.ndarray, flags: np.ndarray) -> Confusion:
    """Count what the flags catch.

    Args:
        labels: 0/1 labels.
        flags: 0/1 or boolean flags on the same rows.

    Returns:
        The counts.

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    y = np.asarray(labels).astype(bool)
    f = np.asarray(flags).astype(bool)
    if y.shape != f.shape:
        raise ValueError(f"labels {y.shape} and flags {f.shape} differ in shape")
    return Confusion(
        rows=len(y),
        positives=int(y.sum()),
        flagged=int(f.sum()),
        true_positives=int((y & f).sum()),
    )


def evaluate_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    rule_flags: np.ndarray,
    threshold: float = OPERATING_THRESHOLD,
) -> dict[str, float]:
    """Every number of one set of rows: size, the model at the threshold, the rule, context.

    The model flags a row when its score is at least ``threshold``, the convention of the
    whole project. PR-AUC (average precision), ROC-AUC and Brier score are context; the first
    two are NaN when the rows hold a single class, because a ranking needs both. The rule gets
    the same context metrics, with its 0/1 flags as its scores (``SignatureRule.predict_proba``
    gives exactly that), so every number has the baseline beside it.

    Args:
        labels: 0/1 labels.
        scores: The model's score for the positive class.
        rule_flags: ``SignatureRule``'s 0/1 flags on the same rows.
        threshold: The model's threshold; the fixed one unless a test passes another.

    Returns:
        Metric name to value. ``model_*`` and ``rule_*`` hold the same counts and rates; the
        model's context metrics are ``pr_auc``, ``roc_auc`` and ``brier``, the rule's
        ``rule_pr_auc``, ``rule_roc_auc`` and ``rule_brier``.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    r = np.asarray(rule_flags, dtype=float)
    metrics: dict[str, float] = {"rows": float(len(y)), "positives": float(y.sum())}
    metrics["prevalence"] = float(y.mean()) if len(y) else float("nan")
    for name, flags in (("model", s >= threshold), ("rule", r)):
        counts = confusion(y, flags)
        metrics |= {
            f"{name}_flagged": float(counts.flagged),
            f"{name}_true_positives": float(counts.true_positives),
            f"{name}_false_positives": float(counts.false_positives),
            f"{name}_false_negatives": float(counts.false_negatives),
            f"{name}_precision": counts.precision,
            f"{name}_recall": counts.recall,
        }
    both_classes = 0 < y.sum() < len(y)
    for prefix, values in (("", s), ("rule_", r)):
        metrics[f"{prefix}pr_auc"] = (
            float(average_precision_score(y, values)) if both_classes else float("nan")
        )
        metrics[f"{prefix}roc_auc"] = (
            float(roc_auc_score(y, values)) if both_classes else float("nan")
        )
        metrics[f"{prefix}brier"] = float(brier_score_loss(y, values)) if len(y) else float("nan")
    return metrics


def window_table(
    steps: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    rule_flags: np.ndarray,
    windows: Sequence[Window],
    threshold: float = OPERATING_THRESHOLD,
) -> pd.DataFrame:
    """``evaluate_rows`` for each window.

    Args:
        steps: The step of each row.
        labels: 0/1 labels.
        scores: The model's score for the positive class.
        rule_flags: ``SignatureRule``'s 0/1 flags.
        windows: Consecutive windows covering every step.
        threshold: The model's threshold.

    Returns:
        One row per window, indexed by window number, with its steps, whether it is complete,
        and its metrics.
    """
    number = window_numbers(steps, windows)
    y, s, r = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(rule_flags)
    rows = []
    for window in windows:
        rows_in = number == window.number
        rows.append(
            {
                "window": window.number,
                "first_step": window.first_step,
                "last_step": window.last_step,
                "complete": window.complete,
                **evaluate_rows(y[rows_in], s[rows_in], r[rows_in], threshold),
            }
        )
    return pd.DataFrame(rows).set_index("window")


def segment_table(
    segments: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    rule_flags: np.ndarray,
    threshold: float = OPERATING_THRESHOLD,
) -> pd.DataFrame:
    """``evaluate_rows`` for each value of a segmenting column, such as the transaction type.

    Args:
        segments: The segment of each row.
        labels: 0/1 labels.
        scores: The model's score for the positive class.
        rule_flags: ``SignatureRule``'s 0/1 flags.
        threshold: The model's threshold.

    Returns:
        One row per distinct segment, sorted, indexed by segment, with its metrics. A segment
        without positives keeps its recall as NaN, and one where nothing is flagged keeps its
        precision as NaN: neither is a perfect score.

    Raises:
        ValueError: If the arrays differ in length.
    """
    g = np.asarray(segments)
    y, s, r = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(rule_flags)
    if not len(g) == len(y) == len(s) == len(r):
        raise ValueError(
            f"segments {len(g)}, labels {len(y)}, scores {len(s)} and rule flags {len(r)} "
            "differ in length"
        )
    rows = []
    for segment in np.unique(g):
        rows_in = g == segment
        rows.append(
            {"segment": segment, **evaluate_rows(y[rows_in], s[rows_in], r[rows_in], threshold)}
        )
    return pd.DataFrame(rows).set_index("segment")


def lowering_cost(
    labels: np.ndarray, scores: np.ndarray, threshold: float = OPERATING_THRESHOLD
) -> pd.DataFrame:
    """What catching the positives the threshold misses would cost, one score at a time.

    For each distinct score of a missed positive, from the highest down, a threshold lowered
    to that score flags every row whose score is at least that high, so positives with the
    same score are gained together. Counts are cumulative from ``threshold``: the last row is
    the price of catching every positive. Nothing is applied; the module docstring says why.

    Args:
        labels: 0/1 labels.
        scores: The model's score for the positive class.
        threshold: The threshold whose misses are priced.

    Returns:
        One row per distinct score of a missed positive, highest first: ``score``,
        ``positives_gained`` and ``negatives_added`` (cumulative, against ``threshold``), and
        ``recall`` and ``precision`` with the threshold at that score. Empty when no positive
        is missed.

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"labels {y.shape} and scores {s.shape} differ in shape")
    now = confusion(y, s >= threshold)
    missed = np.unique(s[y & (s < threshold)])[::-1]
    rows = []
    for score in missed:
        lowered = confusion(y, s >= score)
        rows.append(
            {
                "score": float(score),
                "positives_gained": lowered.true_positives - now.true_positives,
                "negatives_added": lowered.false_positives - now.false_positives,
                "recall": lowered.recall,
                "precision": lowered.precision,
            }
        )
    columns = ["score", "positives_gained", "negatives_added", "recall", "precision"]
    return pd.DataFrame(rows, columns=columns)


def spread(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof = 1) of some columns over the complete windows.

    Args:
        table: From ``window_table``.
        columns: Metric columns to summarize.

    Returns:
        One row per column: ``mean``, ``std``, ``min``, ``max`` and ``windows``.
    """
    complete = table.loc[table["complete"].astype(bool), list(columns)]
    return pd.DataFrame(
        {
            "mean": complete.mean(),
            "std": complete.std(ddof=1),
            "min": complete.min(),
            "max": complete.max(),
            "windows": complete.count(),
        }
    )


# --- Degradation criterion --------------------------------------------------------------


def below_line(
    true_positives: int, positives: int, line: Fraction = DEGRADATION_LINE
) -> bool | None:
    """Whether a recall of ``true_positives / positives`` is strictly below the line.

    Args:
        true_positives: Positives the model flags.
        positives: Positives in the window.
        line: The degradation line.

    Returns:
        The comparison, exact in integers; None when there is no positive, because recall is
        then undefined and the window says nothing.
    """
    if positives == 0:
        return None
    return true_positives * line.denominator < positives * line.numerator


def runs(flags: Sequence[bool | None]) -> list[tuple[int, int]]:
    """The runs of consecutive True values.

    Args:
        flags: One value per window, in order; None counts as not True and breaks a run.

    Returns:
        (position of the first window of the run, length) for each run, in order.
    """
    found, start = [], None
    for position, flag in enumerate([*flags, False]):
        if flag is True and start is None:
            start = position
        elif flag is not True and start is not None:
            found.append((start, position - start))
            start = None
    return found


def degradation_status(flags: Sequence[bool | None]) -> str:
    """Apply the criterion to the complete windows, in order.

    Args:
        flags: ``below_line`` of each complete window, in window order.

    Returns:
        ``DEGRADATION`` with a run of at least ``DEGRADATION_RUN``, ``ALERT`` with one of at
        least ``ALERT_RUN``, ``NO_ALERT`` otherwise.
    """
    longest = max((length for _, length in runs(flags)), default=0)
    if longest >= DEGRADATION_RUN:
        return DEGRADATION
    if longest >= ALERT_RUN:
        return ALERT
    return NO_ALERT


def probability_below(positives: int, recall: float, line: Fraction = DEGRADATION_LINE) -> float:
    """Chance that a window falls below the line when the model's true recall is ``recall``.

    Each positive of the window is caught independently with probability ``recall``, so the
    positives caught are binomial. The window is below the line when it catches at most
    ``(positives * line.numerator - 1) // line.denominator`` of them.

    Args:
        positives: Positives in the window.
        recall: The model's true recall, the same in every window under the null.
        line: The degradation line.

    Returns:
        The probability, 0 when there is no positive.
    """
    if positives == 0:
        return 0.0
    most_caught_below = (positives * line.numerator - 1) // line.denominator
    fewest_missed_below = positives - most_caught_below
    # Summing the few terms with fewer misses is exact enough; the tail has hundreds of terms.
    not_below = sum(
        math.comb(positives, missed) * (1 - recall) ** missed * recall ** (positives - missed)
        for missed in range(fewest_missed_below)
    )
    return max(0.0, 1.0 - not_below)


def probability_of_run(probabilities: Sequence[float], length: int) -> float:
    """Chance of at least one run of ``length`` consecutive successes in independent trials.

    Args:
        probabilities: Success probability of each trial, in order.
        length: Run length that counts.

    Returns:
        The probability.
    """
    # state[j]: probability that no run of `length` happened yet and the current run is j.
    state = np.zeros(length)
    state[0] = 1.0
    reached = 0.0
    for p in probabilities:
        reached += state[length - 1] * p
        moved = np.zeros(length)
        moved[0] = state.sum() * (1 - p)
        moved[1:] = state[:-1] * p
        state = moved
    return float(reached)
