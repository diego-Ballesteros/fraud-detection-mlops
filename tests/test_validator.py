"""Tests of fraud_detection.data.validator.

The corrupted CSV below is the validator's contract in one file: six defects in the
reference period that one run must report together, and one out-of-domain value in
production that must be reported as True or False only, never by its value or a count.
"""

import re
from pathlib import Path

import pytest

from fraud_detection.data import validator
from fraud_detection.data.schema import EXPECTED_COLUMNS
from fraud_detection.data.validator import Finding, Severity, Status

HEADER = ",".join(EXPECTED_COLUMNS)
CLEAN_ROW = "1,PAYMENT,9839.64,C1,170136.0,160296.36,M1,0.0,0.0,0,0"

CORRUPTED_LINES = [
    HEADER + ",extraCol",  # defect 1: a column the schema does not declare
    CLEAN_ROW + ",x",
    "2,REFUND,10.0,C2,10.0,0.0,C3,0.0,0.0,0,0,x",  # defect 2: type outside its domain
    "3,TRANSFER,,C4,10.0,0.0,C5,0.0,0.0,0,0,x",  # defect 3: a missing amount
    "4,CASH_OUT,10.0,C6,10.0,0.0,C7,0.0,0.0,300,0,x",  # defect 4: 300 overflows int8
    "5,CASH_OUT,10.0,C8,10.0,0.0,C9,0.0,0.0,0,1.5,x",  # defect 5: 1.5 is not an integer
    "",  # defect 6: a blank line, which the parser drops
    "401,FOO,10.0,C10,10.0,0.0,C11,0.0,0.0,0,0,x",  # production: type outside its domain
]
SIX_DEFECTS = {
    "columns",
    "dtypes:isFraud",
    "dtypes:isFlaggedFraud",
    "reference domain:type",
    "reference nulls:amount",
    "rows:parsed",
}
UNREADABLE = ("isFraud", "isFlaggedFraud")


def _write(path: Path, lines: list[str]) -> Path:
    # CRLF terminators, like the real file.
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path


def _checks(report: validator.ValidationReport, status: Status) -> set[str]:
    return {finding.check for finding in report.findings if finding.status is status}


@pytest.fixture
def corrupted_report(tmp_path: Path) -> validator.ValidationReport:
    # One single run of the validator over the corrupted file.
    return validator.validate_raw_csv(_write(tmp_path / "corrupted.csv", CORRUPTED_LINES))


def test_reports_the_six_defects_in_one_pass(corrupted_report: validator.ValidationReport) -> None:
    blocking = {finding.check for finding in corrupted_report.blocking_failures}
    assert blocking == SIX_DEFECTS | {"production domain:type"}
    with pytest.raises(validator.DataContractError) as raised:
        corrupted_report.raise_if_blocking()
    message = str(raised.value)
    assert [check for check in SIX_DEFECTS if f"] {check}:" not in message] == []


def test_a_check_that_cannot_run_is_skipped_never_passed(
    corrupted_report: validator.ValidationReport, tmp_path: Path
) -> None:
    # Unreadable columns: each content check on them is skipped, in both periods.
    assert _checks(corrupted_report, Status.SKIPPED) == {
        f"{period} {kind}:{name}"
        for period in ("reference", "production")
        for kind in ("domain", "nulls")
        for name in UNREADABLE
    }
    passed = _checks(corrupted_report, Status.PASSED)
    assert not any(check.endswith(UNREADABLE) for check in passed)
    # The summary that passes counts only the columns it checked: 11, minus amount, which
    # failed, minus the two that could not be read.
    summary = next(f for f in corrupted_report.findings if f.check == "reference nulls")
    assert summary.detail == "no forbidden missing values in 8 columns"

    # Without a usable step the periods cannot be split: nothing that needs them passes.
    no_step = _write(tmp_path / "no_step.csv", [HEADER, "abc" + CLEAN_ROW[1:], CLEAN_ROW])
    report = validator.validate_raw_csv(no_step)
    assert _checks(report, Status.PASSED) == {"columns", "dtypes", "rows:parsed"}
    assert _checks(report, Status.SKIPPED) == {
        "reference domains",
        "reference nulls",
        "reference non-negative",
        "production domains",
        "production nulls",
        "production non-negative",
        "rows:periods",
    }

    # An unreadable money column: its non-negativity cannot be checked, so it is skipped.
    bad_amount = "1,PAYMENT,abc,C1,170136.0,160296.36,M1,0.0,0.0,0,0"
    report = validator.validate_raw_csv(_write(tmp_path / "bad_amount.csv", [HEADER, bad_amount]))
    assert "reference non-negative:amount" in _checks(report, Status.SKIPPED)
    assert not any(c.endswith("non-negative:amount") for c in _checks(report, Status.PASSED))

    # Without a single declared column, or with every declared column unreadable, nothing
    # can be read, and nothing passes.
    for name, lines in (("undeclared", ["foo,bar", "1,x"]), ("unreadable", ["step", "abc"])):
        report = validator.validate_raw_csv(_write(tmp_path / f"{name}.csv", lines))
        assert _checks(report, Status.PASSED) == set()
        assert "rows:parsed" in _checks(report, Status.SKIPPED)


def test_production_findings_say_only_true_or_false(
    corrupted_report: validator.ValidationReport,
) -> None:
    production = [
        finding
        for finding in corrupted_report.findings
        if finding.check.startswith("production") or finding.check == "rows:periods"
    ]
    # Control: there are production findings, so the loop below cannot pass empty.
    assert len(production) > 1
    for finding in production:
        assert re.search(r"\d", finding.detail) is None, finding
    domain = next(f for f in production if f.check == "production domain:type")
    assert (domain.status, domain.detail) == (
        Status.FAILED,
        "every value within the declared domain: False",
    )
    # The offending production value is never named, anywhere in the report.
    assert "FOO" not in corrupted_report.render()


def test_a_clean_file_passes_every_check_and_skips_none(tmp_path: Path) -> None:
    lines = [
        HEADER,
        CLEAN_ROW,
        "2,TRANSFER,181.0,C2,181.0,0.0,C3,0.0,0.0,1,1",
        "3,CASH_OUT,181.0,C4,181.0,0.0,C5,21182.0,0.0,1,0",
        "4,DEBIT,50.0,C6,100.0,50.0,C7,0.0,50.0,0,0",
        "5,CASH_IN,30.0,C8,10.0,40.0,C9,50.0,20.0,0,0",
        "401,TRANSFER,7.0,C10,7.0,0.0,C11,0.0,0.0,1,0",
    ]
    report = validator.validate_raw_csv(_write(tmp_path / "clean.csv", lines))
    assert {finding.status for finding in report.findings} == {Status.PASSED}
    assert {f.check for f in report.findings if f.check.startswith("production")} == {
        "production domain:type",
        "production domain:isFraud",
        "production domain:isFlaggedFraud",
        "production nulls",
        "production non-negative",
    }
    report.raise_if_blocking()


def test_missing_values_are_counted_on_reference_and_only_flagged_on_production(
    tmp_path: Path,
) -> None:
    lines = [
        HEADER,
        "1,TRANSFER,181.0,C1,181.0,0.0,C2,0.0,0.0,,0",  # reference: isFraud missing
        "401,TRANSFER,,C3,7.0,0.0,C4,0.0,0.0,1,0",  # production: amount missing
    ]
    report = validator.validate_raw_csv(_write(tmp_path / "gaps.csv", lines))
    failed = {f.check: f.detail for f in report.findings if f.status is Status.FAILED}
    # An integer column with a missing value cannot keep its integer dtype.
    assert failed["dtypes:isFraud"].startswith("parsed, but held as float64 instead of int8")
    assert failed["reference nulls:isFraud"] == "1 missing values; the schema admits none"
    assert failed["production nulls:amount"] == "no forbidden missing values: False"


def test_negative_money_is_counted_on_reference_and_only_flagged_on_production(
    tmp_path: Path,
) -> None:
    lines = [
        HEADER,
        "1,TRANSFER,-5.0,C1,181.0,0.0,C2,0.0,0.0,1,0",  # reference: negative amount
        "2,CASH_OUT,9.0,C3,-1.0,0.0,C4,0.0,0.0,0,0",  # reference: negative origin balance
        "401,TRANSFER,7.0,C5,7.0,0.0,C6,-1.0,0.0,1,0",  # production: negative destination
    ]
    report = validator.validate_raw_csv(_write(tmp_path / "negative.csv", lines))
    failed = {f.check: f.detail for f in report.findings if f.status is Status.FAILED}
    assert failed["reference non-negative:amount"] == "1 negative values; the schema admits none"
    assert failed["reference non-negative:oldbalanceOrg"] == (
        "1 negative values; the schema admits none"
    )
    assert failed["production non-negative:oldbalanceDest"] == "no negative values: False"
    # newbalanceOrig and newbalanceDest are not declared non-negative: never checked.
    assert not [check for check in failed if "newbalance" in check]


def test_check_columns_names_each_kind_of_problem() -> None:
    missing_and_extra = validator.check_columns(("step", "type", "extraCol"))
    assert [finding.detail.split(":")[0] for finding in missing_and_extra] == [
        "declared but absent from the file",
        "in the file but not declared in the schema",
    ]
    repeated = validator.check_columns((*EXPECTED_COLUMNS, "step"))
    assert [finding.detail for finding in repeated] == ["repeated in the header: 'step'"]
    reordered = validator.check_columns(tuple(reversed(EXPECTED_COLUMNS)))
    assert [(f.status, f.severity) for f in reordered] == [(Status.FAILED, Severity.INFORMATIONAL)]


def test_row_checks_fail_when_rows_are_lost() -> None:
    assert validator.check_parsed_rows(csv_data_lines=3, parsed_rows=3).status is Status.PASSED
    assert validator.check_parsed_rows(csv_data_lines=3, parsed_rows=2).status is Status.FAILED
    assert validator.check_periods(parsed_rows=10, reference_rows=6, production_rows=3) == Finding(
        "rows:periods",
        Status.FAILED,
        Severity.BLOCKING,
        "reference rows + production rows = parsed rows: False",
    )
