"""Local interpretation of the artifact: the operating threshold, four cases, their SHAP values.

**The threshold** is the one the decision metric picks: ``evaluation.recall_at_precision`` at
precision >= 0.99, over the artifact's scores on every row of the reference period, with the
tie policy declared in phase 2 (ADR 0011). The artifact was fitted on those same rows, so
the threshold and every count at it are in-sample, and optimistic. The project declared no
operating threshold before this one; production does not take part in choosing it.

**The four cases** are chosen by criterion, declared before measuring, from the explanation
sample of ``global_explain``. A row is flagged by the model when its score is at least the
threshold, and by the rule when ``SignatureRule`` flags it.

- **a.** A fraud both flag: among them, the median-score row.
- **b.** A fraud the model flags and the rule misses, the frauds the model exists for:
  among them, the median-score row.
- **c.** A false positive of the model: the highest-score negative it flags; none if no
  negative of the sample is flagged.
- **d.** A legitimate row near the threshold: the highest-score negative the model does not
  flag.

"Median" is the row at position ``(n - 1) // 2`` once the group is sorted by score and then
by row identifier, ascending. "Highest" is the largest score, the smallest identifier among
ties. The median stands for the typical member of a group; the highest, for the one closest
to the threshold from its side.
"""

from collections.abc import Mapping
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure

from fraud_detection.features.pipeline import CONFIGURATIONS
from fraud_detection.models.evaluation import (
    DECISION_PRECISION,
    OperatingPoint,
    recall_at_precision,
)

CASE_RULE_AND_MODEL: Final[str] = "a"
CASE_MODEL_ONLY: Final[str] = "b"
CASE_FALSE_POSITIVE: Final[str] = "c"
CASE_NEAR_THRESHOLD: Final[str] = "d"

CASE_TITLES: Final[Mapping[str, str]] = {
    CASE_RULE_AND_MODEL: "a · Fraude que la regla y el modelo capturan",
    CASE_MODEL_ONLY: "b · Fraude que la regla no captura y el modelo sí",
    CASE_FALSE_POSITIVE: "c · Falso positivo del modelo",
    CASE_NEAR_THRESHOLD: "d · Transacción legítima cerca del umbral",
}

MONEY_COLUMNS: Final[tuple[str, ...]] = CONFIGURATIONS["full"].money
"""Shown in the waterfall as the raw amount, not as the scaled feature the model receives."""

TOP_CONTRIBUTIONS: Final[int] = 3


def operating_point(labels: np.ndarray, scores: np.ndarray) -> OperatingPoint:
    """The point the decision metric picks: largest recall with precision >= 0.99.

    Args:
        labels: 0/1 labels.
        scores: Scores, higher meaning more likely positive.

    Returns:
        The operating point, with its threshold.

    Raises:
        ValueError: If no threshold reaches the floor, so there is no operating point.
    """
    point = recall_at_precision(labels, scores, DECISION_PRECISION)
    if point is None:
        raise ValueError(f"No threshold reaches precision {DECISION_PRECISION}")
    return point


def _median_row(scores: pd.Series) -> int | None:
    if scores.empty:
        return None
    ordered = scores.rename("score").rename_axis("row").reset_index()
    ordered = ordered.sort_values(["score", "row"], kind="stable")
    return int(ordered["row"].iloc[(len(ordered) - 1) // 2])


def _highest_row(scores: pd.Series) -> int | None:
    if scores.empty:
        return None
    return int(scores.index[scores == scores.max()].min())


def select_cases(
    labels: pd.Series, scores: pd.Series, rule_flags: pd.Series, threshold: float
) -> dict[str, int | None]:
    """Pick the four cases of the module docstring.

    Args:
        labels: 0/1 labels, indexed by row identifier.
        scores: The model's score of each row, on the same index.
        rule_flags: Whether ``SignatureRule`` flags each row, on the same index.
        threshold: A row is flagged by the model when its score is at least this.

    Returns:
        Case letter to row identifier, or None when the case's group is empty.
    """
    fraud = labels.to_numpy() == 1
    rule = rule_flags.to_numpy(dtype=bool)
    model = scores.to_numpy() >= threshold
    return {
        CASE_RULE_AND_MODEL: _median_row(scores[fraud & rule & model]),
        CASE_MODEL_ONLY: _median_row(scores[fraud & ~rule & model]),
        CASE_FALSE_POSITIVE: _highest_row(scores[~fraud & model]),
        CASE_NEAR_THRESHOLD: _highest_row(scores[~fraud & ~model]),
    }


def top_contributions(values: pd.Series, k: int = TOP_CONTRIBUTIONS) -> pd.Series:
    """The ``k`` largest SHAP values of one row by absolute value, with their sign.

    Args:
        values: One row's SHAP value per feature.
        k: How many to keep.

    Returns:
        The ``k`` values, largest |value| first; ties keep the feature order.
    """
    order = values.abs().sort_values(ascending=False, kind="stable").index
    return values[order[:k]]


def _spanish_number(value: float) -> str:
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def display_values(raw_row: pd.Series, features_row: pd.Series) -> list[str]:
    """How each feature of one row reads in its waterfall.

    Money features show the raw amount, which the scaled feature is a monotone function of;
    every other feature shows the value the model received.

    Args:
        raw_row: The row's raw columns.
        features_row: The row's model features.

    Returns:
        One text per model feature, in the features' order.
    """
    texts = []
    for name, value in features_row.items():
        if name in MONEY_COLUMNS:
            texts.append(_spanish_number(float(raw_row[name])))
        elif isinstance(value, bool | np.bool_):
            texts.append(str(bool(value)))
        else:
            texts.append(f"{float(value):.3g}")
    return texts


def plot_waterfall(
    values: pd.Series, base_value: float, display: list[str], features_row: pd.Series, title: str
) -> Figure:
    """SHAP waterfall of one row, from the base value to its raw output.

    Args:
        values: The row's SHAP value per feature.
        base_value: The explainer's expected value.
        display: How each feature reads, as ``display_values`` returns it.
        features_row: The row's model features, as the model received them.
        title: Figure title.

    Returns:
        The figure.
    """
    explanation = shap.Explanation(
        values=values.to_numpy(dtype=float),
        base_values=base_value,
        data=features_row.to_numpy(dtype=float),
        display_data=np.asarray(display, dtype=object),
        feature_names=list(values.index),
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=len(values), show=False)
    figure = plt.gcf()
    figure.suptitle(title)
    return figure
