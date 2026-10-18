"""Tests of fraud_detection.data.profiling: one test per measurement function.

Every expected value was computed by hand from the rows written here and is written as a
literal, or as the fraction it was computed as. A test that computes its expectation with
the function under test measures nothing.

The base rows carry the edge cases the real measurements touched: an amount of zero (r8),
an origin balance of zero (r3, r8), a balance error of exactly one cent (r6), repeated
identifiers (C1 in nameOrig, C90 in nameDest), types with no positive row (CASH_IN,
PAYMENT) and a declared type with no row at all (DEBIT).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import loader, profiling
from fraud_detection.data.profiling import DuplicateRows, RuleOutcome
from fraud_detection.data.schema import COLUMN_SPECS, EXPECTED_COLUMNS

# Columns in schema order: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
# nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud.
# Per row, by hand: o = oldbalanceOrg - amount - newbalanceOrig,
# d = oldbalanceDest + amount - newbalanceDest, "both zero" = both destination balances
# zero with amount > 0, "equal" = amount == oldbalanceOrg.
ROWS: list[tuple[object, ...]] = [
    # r1: o=0, d=100, both zero, equal
    (1, "TRANSFER", 100.0, "C1", 100.0, 0.0, "C90", 0.0, 0.0, 1, 0),
    # r2: o=0, d=0
    (1, "CASH_OUT", 50.0, "C1", 80.0, 30.0, "C90", 10.0, 60.0, 0, 0),
    # r3: o=-20, d=20, both zero; origin balance zero
    (2, "PAYMENT", 20.0, "C2", 0.0, 0.0, "M1", 0.0, 0.0, 0, 0),
    # r4: o=-60, d=60; money enters the origin
    (2, "CASH_IN", 30.0, "C3", 10.0, 40.0, "C90", 50.0, 20.0, 0, 0),
    # r5: o=-500000, d=500000, both zero, equal, flagged by the system
    (3, "TRANSFER", 500000.0, "C4", 500000.0, 500000.0, "C91", 0.0, 0.0, 1, 1),
    # r6: o=0.01 (exactly one cent), d=0
    (3, "CASH_OUT", 0.01, "C5", 0.02, 0.0, "C92", 0.0, 0.01, 0, 0),
    # r7: o=0, d=7, both zero; moves 7 of a balance of 10, so not equal
    (4, "TRANSFER", 7.0, "C6", 10.0, 3.0, "C93", 0.0, 0.0, 1, 0),
    # r8: amount zero: o=0, d=0, equal (0 == 0); not both zero, because amount is 0
    (4, "CASH_OUT", 0.0, "C7", 0.0, 0.0, "C94", 0.0, 0.0, 1, 0),
    # r9: o=0, d=15, both zero, equal; a legitimate payment
    (5, "PAYMENT", 15.0, "C8", 15.0, 0.0, "M2", 0.0, 0.0, 0, 0),
]
# Label groups: isFraud=0 is r2, r3, r4, r6, r9 (5 rows); isFraud=1 is r1, r5, r7, r8 (4).


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = pd.DataFrame(ROWS, columns=list(EXPECTED_COLUMNS))
    return rows.astype({spec.name: spec.dtype for spec in COLUMN_SPECS})


# --- Structure of the files ---------------------------------------------------------------


@pytest.mark.parametrize("chunk_bytes", [3, 1 << 24])
def test_measure_file_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_bytes: int
) -> None:
    # Bytes: a b \r \n c d \r x \r \n e f. With 3-byte chunks both CRLF pairs are split
    # across two chunks; the counts must not change with the chunk size.
    monkeypatch.setattr(loader, "_SCAN_CHUNK_BYTES", chunk_bytes)
    path = tmp_path / "lines.csv"
    path.write_bytes(b"ab\r\ncd\rx\r\nef")
    structure = profiling.measure_file_structure(path, head_size=5)
    assert (structure.name, structure.size_bytes, structure.head) == ("lines.csv", 12, b"ab\r\nc")
    lines = structure.lines
    assert (lines.lf, lines.crlf, lines.lone_cr, lines.ends_with_newline) == (2, 2, 1, False)
    assert lines.line_count == 3


def test_count_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_bytes(b"step\r\n1\r\n2\r\n401\r\n")
    reference_path = tmp_path / "reference.parquet"
    production_path = tmp_path / "production.parquet"
    pd.DataFrame({"step": [1, 2]}).to_parquet(reference_path)
    pd.DataFrame({"step": [401]}).to_parquet(production_path)
    counts = profiling.count_rows(csv_path, reference_path, production_path)
    assert (counts.csv_data_lines, counts.reference_rows, counts.production_rows) == (3, 2, 1)
    assert counts.period_sum == 3


# --- Quality --------------------------------------------------------------------------------


def test_memory_footprint() -> None:
    frame = pd.DataFrame(
        {
            "a": np.array([1, 2, 3], dtype="int16"),  # 3 x 2 bytes
            "b": np.array([1.0, 2.0, 3.0]),  # 3 x 8 bytes
            "c": np.array([0, 1, 0], dtype="int8"),  # 3 x 1 byte
        }
    )
    table = profiling.memory_footprint(frame)
    assert table["dtype"].to_dict() == {"a": "int16", "b": "float64", "c": "int8", "TOTAL": ""}
    assert table["bytes"].to_dict() == {"a": 6, "b": 24, "c": 3, "TOTAL": 33}


def test_inferred_memory_footprint(tmp_path: Path) -> None:
    path = tmp_path / "inferred.csv"
    path.write_bytes(b"a,b\r\n1,2.5\r\n3,4.5\r\n")
    table = profiling.inferred_memory_footprint(path)
    # pandas infers int64 and float64: 2 rows x 8 bytes each.
    assert table["dtype"].to_dict() == {"a": "int64", "b": "float64", "TOTAL": ""}
    assert table["bytes"].to_dict() == {"a": 16, "b": 16, "TOTAL": 32}


def test_memory_comparison() -> None:
    inferred = pd.DataFrame({"dtype": ["int64", ""], "bytes": [16, 16]}, index=["a", "TOTAL"])
    declared = pd.DataFrame({"dtype": ["int8", ""], "bytes": [2, 2]}, index=["a", "TOTAL"])
    table = profiling.memory_comparison(inferred, declared)
    assert list(table.columns) == [
        "inferred_dtype",
        "inferred_bytes",
        "declared_dtype",
        "declared_bytes",
        "declared/inferred",
    ]
    assert table.loc["a", "declared_dtype"] == "int8"
    assert table["declared/inferred"].to_dict() == {"a": 0.125, "TOTAL": 0.125}


def test_null_counts(frame: pd.DataFrame) -> None:
    assert profiling.null_counts(frame)["nulls"].tolist() == [0] * 11
    with_gap = pd.DataFrame(
        {"amount": [1.0, np.nan, 3.0, 4.0], "step": np.array([1, 2, 3, 4], dtype="int16")}
    )
    table = profiling.null_counts(with_gap)
    assert table.loc["amount"].tolist() == [1, 25.0]
    assert table.loc["step"].tolist() == [0, 0.0]


def test_duplicate_rows(frame: pd.DataFrame) -> None:
    assert profiling.duplicate_rows(frame) == DuplicateRows(9, 0, 0)
    # r1, r1, r2: one redundant copy, two rows in the duplicate group.
    repeated = frame.iloc[[0, 0, 1]].reset_index(drop=True)
    assert profiling.duplicate_rows(repeated) == DuplicateRows(3, 1, 2)


# --- The phenomenon ------------------------------------------------------------------------


def test_class_balance(frame: pd.DataFrame) -> None:
    table = profiling.class_balance(frame)
    assert table["rows"].to_dict() == {0: 5, 1: 4, "total": 9}
    assert table["proportion"].to_dict() == pytest.approx({0: 5 / 9, 1: 4 / 9, "total": 1.0})


def test_label_by_type(frame: pd.DataFrame) -> None:
    result = profiling.label_by_type(frame)
    assert list(result.counts.index) == [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
        "total",
    ]
    assert result.counts.to_dict(orient="index") == {
        "CASH_IN": {0: 1, 1: 0, "total": 1},
        "CASH_OUT": {0: 2, 1: 1, "total": 3},
        "DEBIT": {0: 0, 1: 0, "total": 0},
        "PAYMENT": {0: 2, 1: 0, "total": 2},
        "TRANSFER": {0: 0, 1: 3, "total": 3},
        "total": {0: 5, 1: 4, "total": 9},
    }
    assert result.types_without_positives == ("CASH_IN", "PAYMENT")
    assert result.unobserved_types == ("DEBIT",)


def test_step_distribution(frame: pd.DataFrame) -> None:
    steps = profiling.step_distribution(frame, period_last_step=6)
    assert (steps.first, steps.last, steps.distinct) == (1, 5, 5)
    assert steps.missing == (6,)
    assert steps.rows_per_step.to_dict() == {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 0}
    # Rows per step sorted: 0, 1, 2, 2, 2, 2.
    assert steps.summary[["count", "mean", "min", "50%", "max"]].tolist() == [6, 1.5, 0, 2, 2]


def test_identifier_repetition(frame: pd.DataFrame) -> None:
    table = profiling.identifier_repetition(frame)
    assert table.to_dict(orient="index") == {
        # C1 twice, seven other values once.
        "nameOrig": {
            "rows": 9,
            "distinct": 8,
            "values_repeated": 1,
            "rows_with_repeated_value": 2,
            "max_occurrences": 2,
        },
        # C90 three times, six other values once.
        "nameDest": {
            "rows": 9,
            "distinct": 7,
            "values_repeated": 1,
            "rows_with_repeated_value": 3,
            "max_occurrences": 3,
        },
    }


# --- Leakage hunt ----------------------------------------------------------------------------


def test_flag_profile(frame: pd.DataFrame) -> None:
    flag = profiling.flag_profile(frame)
    assert flag.flagged_rows == 1
    assert flag.by_label.to_dict(orient="index") == {0: {0: 5, 1: 3}, 1: {0: 0, 1: 1}}
    assert flag.types_when_flagged.to_dict() == {
        "CASH_IN": 0,
        "CASH_OUT": 0,
        "DEBIT": 0,
        "PAYMENT": 0,
        "TRANSFER": 1,
    }
    # Not flagged, sorted: 0, 0.01, 7, 15, 20, 30, 50, 100.
    assert flag.amount_stats.to_dict(orient="index") == {
        "flagged": {"rows": 1, "min": 500000.0, "median": 500000.0, "max": 500000.0},
        "not flagged": {"rows": 8, "min": 0.0, "median": 17.5, "max": 100.0},
    }


def test_balance_errors(frame: pd.DataFrame) -> None:
    errors = profiling.balance_errors(frame)
    assert errors["origin_error"].tolist() == pytest.approx(
        [0, 0, -20, -60, -500000, 0.01, 0, 0, 0]
    )
    assert errors["destination_error"].tolist() == pytest.approx(
        [100, 0, 20, 60, 500000, 0, 7, 0, 15]
    )


def test_balance_consistency(frame: pd.DataFrame) -> None:
    table = profiling.balance_consistency(frame)
    # Zero origin error: r1, r2, r7, r8, r9. Zero destination error: r2, r6, r8.
    assert table[["rows", "origin_zero", "destination_zero"]].to_dict(orient="index") == {
        "all": {"rows": 9, "origin_zero": 5, "destination_zero": 3},
        "isFraud=0": {"rows": 5, "origin_zero": 2, "destination_zero": 2},
        "isFraud=1": {"rows": 4, "origin_zero": 3, "destination_zero": 1},
    }
    assert table["origin_share"].to_dict() == pytest.approx(
        {"all": 5 / 9, "isFraud=0": 0.4, "isFraud=1": 0.75}
    )
    assert table["destination_share"].to_dict() == pytest.approx(
        {"all": 1 / 3, "isFraud=0": 0.4, "isFraud=1": 0.25}
    )


def test_balance_tolerance_sensitivity(frame: pd.DataFrame) -> None:
    table = profiling.balance_tolerance_sensitivity(frame)
    assert list(table.index) == [1e-6, 0.005, 0.015, 1.0]
    strict = {
        "origin all": 5 / 9,
        "origin isFraud=0": 0.4,
        "origin isFraud=1": 0.75,
        "destination all": 1 / 3,
        "destination isFraud=0": 0.4,
        "destination isFraud=1": 0.25,
    }
    # r6's one-cent origin error counts as zero only from 0.015 up; no other error lies
    # between one cent and one unit.
    loose = {**strict, "origin all": 6 / 9, "origin isFraud=0": 0.6}
    for tolerance, expected in ((1e-6, strict), (0.005, strict), (0.015, loose), (1.0, loose)):
        assert table.loc[tolerance].to_dict() == pytest.approx(expected)


def test_balance_error_magnitude(frame: pd.DataFrame) -> None:
    table = profiling.balance_error_magnitude(frame)
    stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    # |origin error|, isFraud=1: 0, 0, 0, 500000 -> std = sqrt(1.875e11 / 3) = 250000.
    assert table.loc[("origin_error", 1), stats].tolist() == pytest.approx(
        [4, 125000, 250000, 0, 0, 0, 125000, 500000]
    )
    # |origin error|, isFraud=0, sorted: 0, 0, 0.01, 20, 60.
    assert table.loc[
        ("origin_error", 0), ["count", "mean", "25%", "50%", "75%", "max"]
    ].tolist() == (pytest.approx([5, 16.002, 0, 0.01, 20, 60]))
    # |destination error|, isFraud=0, sorted: 0, 0, 15, 20, 60.
    assert table.loc[("destination_error", 0), ["count", "mean", "50%", "75%", "max"]].tolist() == (
        pytest.approx([5, 19, 15, 20, 60])
    )
    # |destination error|, isFraud=1, sorted: 0, 7, 100, 500000.
    assert table.loc[("destination_error", 1), ["mean", "25%", "50%", "75%"]].tolist() == (
        pytest.approx([125026.75, 5.25, 53.5, 125075])
    )


def test_cent_resolution(frame: pd.DataFrame) -> None:
    assert profiling.cent_resolution(frame)["not_whole_cents"].tolist() == [0, 0, 0, 0, 0]
    # 1.005 is not a whole number of cents; the largest real amount still is.
    odd = pd.DataFrame({"amount": [1.005, 2.0, 0.01, 92445516.64]})
    assert profiling.cent_resolution(odd, columns=("amount",)).loc["amount"].tolist() == [1, 0.25]


def test_zero_destination_balances(frame: pd.DataFrame) -> None:
    table = profiling.zero_destination_balances(frame)
    # Both zero with amount > 0: r1, r3, r5, r7, r9. r8 has both zero but amount 0.
    assert table[["rows", "both_zero_with_amount"]].to_dict(orient="index") == {
        "all": {"rows": 9, "both_zero_with_amount": 5},
        "isFraud=0": {"rows": 5, "both_zero_with_amount": 2},
        "isFraud=1": {"rows": 4, "both_zero_with_amount": 3},
    }
    assert table["share"].to_dict() == pytest.approx(
        {"all": 5 / 9, "isFraud=0": 0.4, "isFraud=1": 0.75}
    )


def test_label_correlations() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 2.0, 10.0],
            "copy": np.array([0, 0, 1, 1], dtype="int8"),
            "nameOrig": pd.array(["C1", "C2", "C3", "C4"], dtype=pd.StringDtype("pyarrow")),
            "isFraud": np.array([0, 0, 1, 1], dtype="int8"),
        }
    )
    table = profiling.label_correlations(frame)
    # Neither the label itself nor the text column gets a row.
    assert list(table.index) == ["x", "copy"]
    # Pearson of x: deviations (-2.75, -1.75, -1.75, 6.25) against (-0.5, -0.5, 0.5, 0.5)
    # give 4.5 / sqrt(52.75 * 1). Spearman: average ranks (1, 2.5, 2.5, 4) against
    # (1.5, 1.5, 3.5, 3.5) give 3 / sqrt(4.5 * 4) = 3 / sqrt(18).
    assert table.loc["x"].tolist() == pytest.approx([4.5 / math.sqrt(52.75), 3 / math.sqrt(18)])
    assert table.loc["copy"].tolist() == pytest.approx([1.0, 1.0])


# --- Beyond the requested list ---------------------------------------------------------------


def test_balance_consistency_by_type(frame: pd.DataFrame) -> None:
    table = profiling.balance_consistency_by_type(frame)
    # rows, zero origin share, zero destination share, both-zero share.
    expected = {
        ("CASH_IN", 0): [1, 0.0, 0.0, 0.0],
        ("CASH_OUT", 0): [2, 0.5, 1.0, 0.0],
        ("CASH_OUT", 1): [1, 1.0, 1.0, 0.0],
        ("PAYMENT", 0): [2, 0.5, 0.0, 1.0],
        ("TRANSFER", 1): [3, 2 / 3, 0.0, 1.0],
    }
    assert list(table.index) == list(expected)
    for key, values in expected.items():
        assert table.loc[key].tolist() == pytest.approx(values)


def test_label_by_step(frame: pd.DataFrame) -> None:
    by_step = profiling.label_by_step(frame, period_last_step=6)
    assert by_step.positives_per_step.to_dict() == {1: 1, 2: 0, 3: 1, 4: 2, 5: 0, 6: 0}
    # Step 4 holds r7 and r8, both positive.
    assert (by_step.steps_all_positive, by_step.positives_in_those_steps) == (1, 2)
    assert by_step.total_positives == 4
    # Rows (2, 2, 2, 2, 1, 0) against positives (1, 0, 1, 2, 0, 0): cross deviations sum
    # to 2, squared deviations to 3.5 and 10/3.
    assert by_step.volume_correlation == pytest.approx(2 / math.sqrt(3.5 * 10 / 3))
    assert by_step.summary[["count", "mean", "max"]].tolist() == pytest.approx([6, 4 / 6, 2])


def test_amount_equals_origin_balance(frame: pd.DataFrame) -> None:
    table = profiling.amount_equals_origin_balance(frame)
    # rows, rows where amount == oldbalanceOrg, share.
    expected = {
        ("CASH_IN", 0): [1, 0, 0.0],
        ("CASH_OUT", 0): [2, 0, 0.0],
        ("CASH_OUT", 1): [1, 1, 1.0],  # r8: 0 == 0
        ("PAYMENT", 0): [2, 1, 0.5],  # r9
        ("TRANSFER", 1): [3, 2, 2 / 3],  # r1 and r5; r7 moves 7 of 10
    }
    assert list(table.index) == list(expected)
    for key, values in expected.items():
        assert table.loc[key].tolist() == pytest.approx(values)


def test_signature_rule_outcome(frame: pd.DataFrame) -> None:
    # Equal rows: r1, r5, r8 (positive) and r9 (negative); r7 is a positive left out.
    anywhere = profiling.signature_rule_outcome(frame)
    assert anywhere == RuleOutcome(
        flagged_rows=4, true_positives=3, false_positives=1, false_negatives=1
    )
    assert (anywhere.precision, anywhere.recall) == (0.75, 0.75)
    # Restricted to TRANSFER and CASH_OUT, the legitimate PAYMENT r9 drops out.
    restricted = profiling.signature_rule_outcome(frame, types=("TRANSFER", "CASH_OUT"))
    assert restricted == RuleOutcome(
        flagged_rows=3, true_positives=3, false_positives=0, false_negatives=1
    )
    assert (restricted.precision, restricted.recall) == (1.0, 0.75)
    nothing = RuleOutcome(flagged_rows=0, true_positives=0, false_positives=0, false_negatives=0)
    assert math.isnan(nothing.precision)
    assert math.isnan(nothing.recall)


def test_destination_prefix_by_type(frame: pd.DataFrame) -> None:
    table = profiling.destination_prefix_by_type(frame)
    assert table.to_dict(orient="index") == {
        "CASH_IN": {"C": 1, "M": 0},
        "CASH_OUT": {"C": 3, "M": 0},
        "PAYMENT": {"C": 0, "M": 2},
        "TRANSFER": {"C": 3, "M": 0},
    }
