"""Tests of fraud_detection.explain.local_explain, with expected values computed by hand."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fraud_detection.explain.local_explain import (
    CASE_FALSE_POSITIVE,
    CASE_MODEL_ONLY,
    CASE_NEAR_THRESHOLD,
    CASE_RULE_AND_MODEL,
    display_values,
    operating_point,
    plot_waterfall,
    select_cases,
    top_contributions,
)

# Row identifiers 10 to 19. Threshold 0.5.
#   10, 11, 12: fraud the rule flags, scores 0.9, 0.7, 0.8 -> sorted 11, 12, 10 -> median 12
#   13, 14:     fraud the rule misses, both 0.6           -> tie, smaller id -> 13
#   15:         fraud the rule misses, 0.2                 -> the model misses it too
#   16, 17:     legitimate, 0.55 and 0.5 (0.5 is flagged)  -> highest flagged -> 16
#   18, 19:     legitimate, 0.45 and 0.3                   -> highest unflagged -> 18
IDS = list(range(10, 20))
LABELS = pd.Series([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=IDS)
SCORES = pd.Series([0.9, 0.7, 0.8, 0.6, 0.6, 0.2, 0.55, 0.5, 0.45, 0.3], index=IDS)
RULE = pd.Series([True, True, True, False, False, False, False, False, False, False], index=IDS)


def test_the_four_cases_by_hand() -> None:
    assert select_cases(LABELS, SCORES, RULE, threshold=0.5) == {
        CASE_RULE_AND_MODEL: 12,
        CASE_MODEL_ONLY: 13,
        CASE_FALSE_POSITIVE: 16,
        CASE_NEAR_THRESHOLD: 18,
    }


def test_a_score_equal_to_the_threshold_is_flagged() -> None:
    scores = SCORES.copy()
    scores[16] = 0.4
    # Row 17 (0.5) is now the only flagged legitimate row; 16 (0.4) is below the threshold.
    cases = select_cases(LABELS, scores, RULE, threshold=0.5)
    assert cases[CASE_FALSE_POSITIVE] == 17
    assert cases[CASE_NEAR_THRESHOLD] == 18


def test_an_empty_group_gives_no_case() -> None:
    scores = SCORES.copy()
    scores[[16, 17]] = [0.1, 0.1]
    cases = select_cases(LABELS, scores, RULE, threshold=0.5)
    assert cases[CASE_FALSE_POSITIVE] is None
    assert cases[CASE_NEAR_THRESHOLD] == 18


def test_operating_point_by_hand_and_when_none_exists() -> None:
    point = operating_point(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.1, 0.2]))
    assert (point.threshold, point.flagged, point.true_positives) == (0.8, 2, 2)
    with pytest.raises(ValueError, match="No threshold"):
        operating_point(np.array([0, 1]), np.array([1.0, 0.0]))


def test_top_contributions_by_hand() -> None:
    values = pd.Series({"a": 0.5, "b": -2.0, "c": 1.0, "d": -0.1})
    top = top_contributions(values)
    assert list(top.index) == ["b", "c", "a"]
    assert top.tolist() == [-2.0, 1.0, 0.5]


def test_display_values_show_raw_money_and_model_values_otherwise() -> None:
    raw = pd.Series({"amount": 181_000.5, "oldbalanceOrg": 0.0, "oldbalanceDest": 12.0})
    features = pd.Series(
        {
            "amount": 1.7,
            "oldbalanceOrg": -0.4,
            "oldbalanceDest": 0.2,
            "type_TRANSFER": 1.0,
            "amount_equals_origin_balance": True,
            "hour_sin": 0.258819,
        },
        dtype=object,
    )
    assert display_values(raw, features) == ["181.000,50", "0,00", "12,00", "1", "True", "0.259"]


def test_the_waterfall_draws() -> None:
    values = pd.Series({"x": 1.0, "y": -0.5})
    figure = plot_waterfall(values, -2.0, ["1", "2"], pd.Series({"x": 1.0, "y": 2.0}), "t")
    assert figure.axes
    plt.close("all")
