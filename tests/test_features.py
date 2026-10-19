"""Tests of fraud_detection.features: the derived features and the preprocessing Pipeline.

Expected values are literals computed by hand from the rows written here.
"""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer

from fraud_detection.data.schema import (
    COLUMN_SPECS,
    COLUMNS,
    EXPECTED_COLUMNS,
    MODEL_ELIGIBLE_COLUMNS,
    Availability,
)
from fraud_detection.features import builder
from fraud_detection.features.pipeline import CONFIGURATIONS, build_preprocessor

ROWS: list[tuple[object, ...]] = [
    # f1: hour 1; takes the whole balance: signature
    (1, "TRANSFER", 100.0, "C1", 100.0, 0.0, "C9", 0.0, 0.0, 1, 0),
    # f2: hour 0; moves 0 over a balance of 0: no signature
    (24, "CASH_OUT", 0.0, "C2", 0.0, 0.0, "C9", 0.0, 0.0, 1, 0),
    # f3: hour 6
    (30, "PAYMENT", 20.0, "C3", 50.0, 30.0, "M1", 0.0, 0.0, 0, 0),
    # f4: hour 23
    (47, "CASH_IN", 30.0, "C4", 10.0, 40.0, "C8", 50.0, 20.0, 0, 0),
    # f5: hour 0
    (48, "DEBIT", 7.0, "C5", 10.0, 3.0, "C7", 0.0, 7.0, 0, 0),
]
ONE_HOT = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
EXPECTED_OUTPUT = {
    "full": [
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        *ONE_HOT,
        "amount_equals_origin_balance",
        "hour_sin",
        "hour_cos",
    ],
    "no_signature": ["amount", "oldbalanceDest", *ONE_HOT, "hour_sin", "hour_cos"],
}
EXPECTED_BRANCHES = {
    "full": {
        "money": ["amount", "oldbalanceOrg", "oldbalanceDest"],
        "categorical": ["type"],
        "derived": ["amount_equals_origin_balance", "hour_sin", "hour_cos"],
    },
    "no_signature": {
        "money": ["amount", "oldbalanceDest"],
        "categorical": ["type"],
        "derived": ["hour_sin", "hour_cos"],
    },
}
DERIVED = {builder.SIGNATURE_FEATURE, builder.HOUR_SIN, builder.HOUR_COS}


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = pd.DataFrame(ROWS, columns=list(EXPECTED_COLUMNS))
    return rows.astype({spec.name: spec.dtype for spec in COLUMN_SPECS})


def _branches(configuration: str) -> dict[str, list[str]]:
    columns = build_preprocessor(configuration).named_steps["columns"]
    return {name: list(selected) for name, _, selected in columns.transformers}


def _read_type(values: pd.DataFrame) -> pd.DataFrame:
    return values[["type"]]


def _read_amount(values: pd.DataFrame) -> pd.DataFrame:
    return values[["amount"]]


# --- The partition of the selected columns ---------------------------------------------------


@pytest.mark.parametrize("configuration", ["full", "no_signature"])
def test_branches_partition_the_selected_columns_against_the_schema(configuration: str) -> None:
    branches = _branches(configuration)
    assert branches == EXPECTED_BRANCHES[configuration]
    routed = [column for selected in branches.values() for column in selected]
    # Each selected column goes to exactly one branch, and nothing else is routed.
    assert len(routed) == len(set(routed))
    assert set(routed) == set(CONFIGURATIONS[configuration].selected)
    # Every routed column is a derived feature or a column of the schema that exists
    # before the transaction.
    from_schema = [column for column in routed if column in COLUMNS]
    assert set(routed) - set(from_schema) <= DERIVED
    assert all(COLUMNS[c].availability is Availability.PRE_TRANSACTION for c in from_schema)
    remainder = build_preprocessor(configuration).named_steps["columns"].remainder
    assert remainder == "drop"


@pytest.mark.parametrize("configuration", ["full", "no_signature"])
def test_output_names_are_the_expected_ones_and_never_step(
    frame: pd.DataFrame, configuration: str
) -> None:
    pipeline = build_preprocessor(configuration).fit(frame)
    names = list(pipeline.get_feature_names_out())
    assert names == EXPECTED_OUTPUT[configuration]
    assert "step" not in names
    assert list(pipeline.transform(frame).columns) == EXPECTED_OUTPUT[configuration]


def test_raw_columns_each_configuration_reads_are_eligible() -> None:
    assert CONFIGURATIONS["full"].raw_inputs == (
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "type",
        "step",
    )
    assert CONFIGURATIONS["no_signature"].raw_inputs == ("amount", "oldbalanceDest", "type", "step")
    for configuration in CONFIGURATIONS.values():
        assert set(configuration.raw_inputs) <= MODEL_ELIGIBLE_COLUMNS


# --- Level 1: a violation raises instead of working ------------------------------------------


def test_a_branch_cannot_read_another_branch_column(frame: pd.DataFrame) -> None:
    # The money branch receives only its own columns: reading the categorical one fails.
    violating = build_preprocessor("full").set_params(
        columns__money=FunctionTransformer(_read_type)
    )
    with pytest.raises(KeyError):
        violating.fit(frame)
    # Control: the same replacement reading one of its own columns fits.
    build_preprocessor("full").set_params(columns__money=FunctionTransformer(_read_amount)).fit(
        frame
    )


def test_a_derivation_cannot_read_beyond_its_inputs(frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        builder.cyclic_hour(frame[["amount"]])
    with pytest.raises(KeyError):
        builder.amount_equals_origin_balance(frame[["step", "amount"]])
    # The builder keeps the eligible columns only, so nothing after it can reach the rest.
    derived = builder.DerivedFeatures().transform(frame)
    assert not set(derived.columns) & (set(EXPECTED_COLUMNS) - MODEL_ELIGIBLE_COLUMNS)
    with pytest.raises(KeyError):
        builder.DerivedFeatures().transform(frame.drop(columns="oldbalanceDest"))


# --- The derived features --------------------------------------------------------------------


def test_signature_is_false_for_zero_over_zero(frame: pd.DataFrame) -> None:
    # f1 takes its whole balance; f2 moves 0 over a balance of 0.
    assert builder.amount_equals_origin_balance(frame).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_cyclic_hour(frame: pd.DataFrame) -> None:
    hours = builder.cyclic_hour(frame)
    # f2 and f5 are hour 0: (sin, cos) = (0, 1). f3 is hour 6: (1, 0).
    assert hours.loc[[1, 4]].to_numpy().ravel().tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert hours.loc[2].tolist() == pytest.approx([1.0, 0.0])
    # Hour 23 (f4) and hour 0 (f5) are as close as hour 0 (f2) and hour 1 (f1): one hour
    # apart on the circle, a chord of 2 sin(pi / 24).
    chord = 2 * math.sin(math.pi / 24)
    assert np.linalg.norm(hours.loc[3] - hours.loc[4]) == pytest.approx(chord)
    assert np.linalg.norm(hours.loc[1] - hours.loc[0]) == pytest.approx(chord)


# --- The factory and the fitted object -------------------------------------------------------


def test_the_factory_returns_independent_pipelines(frame: pd.DataFrame) -> None:
    first, second = build_preprocessor("full"), build_preprocessor("full")
    assert first is not second
    assert first.named_steps["columns"] is not second.named_steps["columns"]
    first.fit(frame)
    # Fitting the first leaves the second unfitted.
    with pytest.raises(NotFittedError):
        second.transform(frame)
    before = first.transform(frame)
    shifted = frame.assign(
        amount=frame["amount"] * 1000, oldbalanceDest=frame["oldbalanceDest"] + 5
    )
    second.fit(shifted)
    # Fitting the second leaves the first as it was.
    pd.testing.assert_frame_equal(first.transform(frame), before)
    centers = [
        pipeline.named_steps["columns"].named_transformers_["money"].named_steps["scale"].center_
        for pipeline in (first, second)
    ]
    assert not np.allclose(centers[0], centers[1])


def test_a_pickled_pipeline_transforms_identically(frame: pd.DataFrame) -> None:
    fitted = build_preprocessor("full").fit(frame)
    restored = pickle.loads(pickle.dumps(fitted))
    pd.testing.assert_frame_equal(restored.transform(frame), fitted.transform(frame))


def test_negative_money_is_refused(frame: pd.DataFrame) -> None:
    negative = frame.assign(oldbalanceDest=frame["oldbalanceDest"] - 100)
    with pytest.raises(ValueError, match="Negative values"):
        build_preprocessor("full").fit(negative)


def test_an_unknown_type_raises_instead_of_encoding_zeros(frame: pd.DataFrame) -> None:
    fitted = build_preprocessor("full").fit(frame)
    unknown = frame.assign(type=frame["type"].astype(str).replace("DEBIT", "REFUND"))
    with pytest.raises(ValueError, match="unknown categor"):
        fitted.transform(unknown)
