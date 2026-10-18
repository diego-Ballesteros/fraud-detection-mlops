"""Tests of fraud_detection.models.baseline: the signature rule as a classifier."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from fraud_detection.data import profiling
from fraud_detection.data.schema import COLUMN_SPECS, EXPECTED_COLUMNS
from fraud_detection.features import builder
from fraud_detection.models import baseline
from fraud_detection.models.baseline import SignatureRule

ROWS: list[tuple[object, ...]] = [
    # b1: TRANSFER taking the whole balance: flagged
    (1, "TRANSFER", 100.0, "C1", 100.0, 0.0, "C9", 0.0, 0.0, 1, 0),
    # b2: CASH_OUT moving 0 over 0: not flagged
    (2, "CASH_OUT", 0.0, "C2", 0.0, 0.0, "C9", 0.0, 0.0, 1, 0),
    # b3: CASH_OUT taking the whole balance: flagged
    (3, "CASH_OUT", 55.5, "C3", 55.5, 0.0, "C8", 0.0, 55.5, 0, 0),
    # b4: PAYMENT taking the whole balance: not flagged, the type is outside the rule
    (4, "PAYMENT", 15.0, "C4", 15.0, 0.0, "M1", 0.0, 0.0, 0, 0),
    # b5: TRANSFER moving part of the balance: not flagged
    (5, "TRANSFER", 7.0, "C5", 10.0, 3.0, "C7", 0.0, 0.0, 1, 0),
]


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = pd.DataFrame(ROWS, columns=list(EXPECTED_COLUMNS))
    return rows.astype({spec.name: spec.dtype for spec in COLUMN_SPECS})


def test_predictions_follow_the_rule(frame: pd.DataFrame) -> None:
    rule = SignatureRule().fit(frame)
    assert rule.predict(frame).tolist() == [1, 0, 1, 0, 0]
    assert rule.predict_proba(frame).tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    assert rule.classes_.tolist() == [0, 1]


def test_fit_records_nothing(frame: pd.DataFrame) -> None:
    rule = SignatureRule()
    state = dict(vars(rule))
    rule.fit(frame, frame["isFraud"])
    assert vars(rule) == state
    # Fitted on two different datasets, the rule predicts the same on a third.
    other = frame.assign(
        isFraud=1 - frame["isFraud"],
        amount=frame["amount"] * 3,
        oldbalanceOrg=frame["oldbalanceOrg"] + 1,
    )
    first = SignatureRule().fit(frame, frame["isFraud"])
    second = SignatureRule().fit(other, other["isFraud"])
    third = frame.iloc[::-1].reset_index(drop=True)
    assert np.array_equal(first.predict(third), second.predict(third))


def test_scikit_learn_sees_the_stateless_objects_as_fitted() -> None:
    # The phase 2 harness may check fitting before predicting; a rule has nothing to fit.
    check_is_fitted(SignatureRule())
    check_is_fitted(builder.DerivedFeatures())


def test_one_definition_of_the_signature() -> None:
    # The feature, the baseline and the measurements call the same function object.
    assert baseline.amount_equals_origin_balance is builder.amount_equals_origin_balance
    assert profiling.amount_equals_origin_balance is builder.amount_equals_origin_balance


def test_the_estimator_and_the_measured_rule_agree(frame: pd.DataFrame) -> None:
    measured = profiling.signature_rule_outcome(frame, types=baseline.BASELINE_TYPES)
    predicted = profiling.rule_outcome(SignatureRule().predict(frame), frame["isFraud"])
    # Flagged b1 (positive) and b3 (negative); b2 and b5 are positives left out.
    assert (
        predicted
        == measured
        == profiling.RuleOutcome(
            flagged_rows=2,
            true_positives=1,
            false_positives=1,
            false_negatives=2,
            true_negatives=1,
        )
    )
