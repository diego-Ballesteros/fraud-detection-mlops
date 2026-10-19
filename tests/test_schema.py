"""Tests of the data contract declared in fraud_detection.data.schema.

The header test reads the real CSV and is skipped where the file is absent, as in CI,
because data/ is gitignored. Every other test runs on the declarations alone.
"""

import dataclasses
from collections import Counter

import pytest

from fraud_detection.data import loader, schema
from fraud_detection.data.schema import Availability

requires_real_csv = pytest.mark.skipif(
    not loader.RAW_CSV_PATH.is_file(),
    reason=f"real CSV absent (data/ is gitignored): {loader.RAW_CSV_PATH}",
)


def test_every_declared_column_is_classified_exactly_once() -> None:
    names = [spec.name for spec in schema.COLUMN_SPECS]
    assert [name for name, count in Counter(names).items() if count > 1] == []
    assert all(isinstance(spec.availability, Availability) for spec in schema.COLUMN_SPECS)
    assert all(spec.justification.strip() for spec in schema.COLUMN_SPECS)

    # The classes partition the columns: every column in one class, none in two.
    classes = {
        availability: {s.name for s in schema.COLUMN_SPECS if s.availability is availability}
        for availability in Availability
    }
    assert sum(len(members) for members in classes.values()) == len(names)
    assert set().union(*classes.values()) == set(schema.EXPECTED_COLUMNS) == set(schema.COLUMNS)


def test_model_eligible_columns_are_exactly_pre_transaction() -> None:
    pre = {s.name for s in schema.COLUMN_SPECS if s.availability is Availability.PRE_TRANSACTION}
    others = {s.name for s in schema.COLUMN_SPECS} - pre
    assert pre == schema.MODEL_ELIGIBLE_COLUMNS
    # Control: there are non-eligible columns, and none of them leaked into the set.
    assert others
    assert not others & schema.MODEL_ELIGIBLE_COLUMNS
    assert schema.TARGET_COLUMN not in schema.MODEL_ELIGIBLE_COLUMNS


@requires_real_csv
def test_expected_columns_match_the_real_csv_header() -> None:
    # Real file, header only. If the CSV gains, loses or renames a column and the schema
    # does not follow, this fails before any pipeline processes an unclassified column.
    assert loader.read_csv_header(loader.RAW_CSV_PATH) == schema.EXPECTED_COLUMNS


def test_exposed_objects_are_immutable() -> None:
    spec = schema.COLUMNS["isFlaggedFraud"]
    with pytest.raises(TypeError):
        schema.COLUMNS["extra"] = spec
    with pytest.raises(TypeError):
        del schema.COLUMNS["isFlaggedFraud"]
    with pytest.raises(AttributeError):
        schema.MODEL_ELIGIBLE_COLUMNS.add("newbalanceOrig")
    with pytest.raises(TypeError):
        schema.EXPECTED_COLUMNS[0] = "extra"
    with pytest.raises(TypeError):
        schema.COLUMN_SPECS[0] = spec
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.availability = Availability.PRE_TRANSACTION
    with pytest.raises(TypeError):
        schema.TRANSACTION_TYPES[0] = "REFUND"


def test_the_non_negative_columns_are_the_three_model_money_columns() -> None:
    non_negative = {spec.name for spec in schema.COLUMN_SPECS if spec.non_negative}
    assert non_negative == {"amount", "oldbalanceOrg", "oldbalanceDest"}
