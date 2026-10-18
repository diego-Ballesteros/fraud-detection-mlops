"""Validate the raw PaySim CSV against the schema, accumulating every finding.

The validator never stops at the first problem and never repairs one: it runs every
check, collects all the findings, and fails once at the end with the full listing. A
check that cannot run is reported as skipped, never as passed.

What each period allows. The reference period gets every check, and its findings report
counts and values. On production, everything that is a property of the RECORD is
allowed: that the declared dtypes apply, that the columns are present and in order, that
the values are within the declared domain, that there are no nulls where the schema
forbids them. Everything that is a property of the PHENOMENON is forbidden: rates,
distributions, statistics, cardinalities, correlations. Checks on production report True
or False, never a count or a proportion. Reason: a value outside the domain in
production makes phase 4 fail, and discovering it today costs nothing; but a count
informs about the phenomenon, and that crosses the cut.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import pyarrow as pa

from fraud_detection.data.loader import (
    STEP_COLUMN,
    read_csv_header,
    read_raw_csv,
    scan_lines,
    split_periods,
)
from fraud_detection.data.schema import COLUMN_SPECS, COLUMNS, EXPECTED_COLUMNS

_MAX_LISTED_VALUES = 10


class Severity(Enum):
    """Whether a failed check stops the pipeline."""

    BLOCKING = "blocking"
    INFORMATIONAL = "informational"


class Status(Enum):
    """Outcome of one check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Period(Enum):
    """Period a content check runs on; it decides what the findings may report."""

    REFERENCE = "reference"
    PRODUCTION = "production"


@dataclass(frozen=True)
class Finding:
    """Outcome of one check.

    Attributes:
        check: Name of the check, with the period and the column it applies to when
            there are.
        status: Whether the check passed, failed or could not run.
        severity: Whether a failure of this check is blocking.
        detail: What was found, in one line.
    """

    check: str
    status: Status
    severity: Severity
    detail: str


class DataContractError(Exception):
    """Raised when a validation report holds at least one blocking failure."""


@dataclass(frozen=True)
class ValidationReport:
    """Every finding of a validation run, in the order the checks ran.

    Attributes:
        source: What was validated.
        findings: One or more findings per check.
    """

    source: str
    findings: tuple[Finding, ...]

    @property
    def blocking_failures(self) -> tuple[Finding, ...]:
        """Failed findings whose severity is blocking."""
        return tuple(
            finding
            for finding in self.findings
            if finding.status is Status.FAILED and finding.severity is Severity.BLOCKING
        )

    def render(self) -> str:
        """Format the report as text: a summary line, then one line per finding.

        Returns:
            The report, one finding per line.
        """
        counts = Counter(finding.status for finding in self.findings)
        summary = (
            f"Validation of {self.source}: {counts[Status.PASSED]} passed, "
            f"{counts[Status.FAILED]} failed ({len(self.blocking_failures)} blocking), "
            f"{counts[Status.SKIPPED]} skipped"
        )
        lines = [summary]
        for finding in self.findings:
            tag = finding.status.value.upper()
            if finding.status is Status.FAILED:
                tag = f"{tag}, {finding.severity.value}"
            lines.append(f"  [{tag}] {finding.check}: {finding.detail}")
        return "\n".join(lines)

    def raise_if_blocking(self) -> None:
        """Raise with the full listing if any blocking check failed.

        Raises:
            DataContractError: If at least one finding is a blocking failure.
        """
        if self.blocking_failures:
            raise DataContractError(self.render())


def _passed(check: str, detail: str) -> Finding:
    return Finding(check, Status.PASSED, Severity.BLOCKING, detail)


def _failed(check: str, detail: str, severity: Severity = Severity.BLOCKING) -> Finding:
    return Finding(check, Status.FAILED, severity, detail)


def _skipped(check: str, detail: str) -> Finding:
    return Finding(check, Status.SKIPPED, Severity.BLOCKING, detail)


def _boolean(check: str, statement: str, holds: bool) -> Finding:
    """A finding that only says whether a statement holds: no count, no value."""
    detail = f"{statement}: {holds}"
    return _passed(check, detail) if holds else _failed(check, detail)


def _listed(values: Sequence[object]) -> str:
    shown = ", ".join(repr(value) for value in values[:_MAX_LISTED_VALUES])
    extra = len(values) - _MAX_LISTED_VALUES
    return f"{shown}, and {extra} more" if extra > 0 else shown


def check_columns(header: Sequence[str]) -> list[Finding]:
    """Check that the header holds exactly the declared columns, none missing or extra.

    Args:
        header: Column names as they appear in the file.

    Returns:
        The findings of the check: one per kind of problem, or one passed finding.
    """
    check = "columns"
    missing = [name for name in EXPECTED_COLUMNS if name not in header]
    unexpected = [name for name in header if name not in COLUMNS]
    repeated = [name for name, count in Counter(header).items() if count > 1]
    findings = []
    if missing:
        findings.append(_failed(check, f"declared but absent from the file: {_listed(missing)}"))
    if unexpected:
        findings.append(
            _failed(check, f"in the file but not declared in the schema: {_listed(unexpected)}")
        )
    if repeated:
        findings.append(_failed(check, f"repeated in the header: {_listed(repeated)}"))
    if findings:
        return findings
    if tuple(header) != EXPECTED_COLUMNS:
        # Columns are read by name, so a different order is harmless but worth knowing.
        return [
            _failed(
                check,
                f"declared columns present, in a different order: {tuple(header)}",
                Severity.INFORMATIONAL,
            )
        ]
    return [_passed(check, f"the {len(header)} declared columns, in the declared order")]


def _expected_read_dtype_matches(column: pd.Series, declared: object) -> bool:
    if isinstance(declared, pd.CategoricalDtype):
        # Categories stay open until validation; only the kind of dtype is checked here.
        return isinstance(column.dtype, pd.CategoricalDtype)
    return bool(column.dtype == declared)


def _read_checking_dtypes(
    csv_path: Path, columns: Sequence[str]
) -> tuple[list[Finding], pd.DataFrame | None]:
    """Read the columns with their declared types, reporting every column that fails."""
    check = "dtypes"
    findings: list[Finding] = []
    if not columns:
        # An empty include list makes pyarrow read every column, undeclared ones included,
        # with inferred types.
        return findings, None
    readable = list(columns)
    try:
        frame = read_raw_csv(csv_path, columns)
    except pa.ArrowInvalid:
        # pyarrow stops at the first column it cannot convert; reading each column on its
        # own names every column that fails, not only the first one.
        readable = []
        for name in columns:
            try:
                read_raw_csv(csv_path, [name])
            except pa.ArrowInvalid as error:
                findings.append(
                    _failed(f"{check}:{name}", f"cannot be read as {COLUMNS[name].dtype}: {error}")
                )
            else:
                readable.append(name)
        frame = read_raw_csv(csv_path, readable) if readable else None
    if frame is None:
        return findings, None

    mismatched = [
        name
        for name in readable
        if not _expected_read_dtype_matches(frame[name], COLUMNS[name].dtype)
    ]
    for name in mismatched:
        findings.append(
            _failed(
                f"{check}:{name}",
                f"parsed, but held as {frame[name].dtype} instead of {COLUMNS[name].dtype} "
                "(an integer column with missing values does this)",
            )
        )
    applied = [name for name in readable if name not in mismatched]
    if applied:
        findings.append(_passed(check, f"declared dtype applied to {len(applied)} columns"))
    return findings, frame


def check_domains(frame: pd.DataFrame, period: Period = Period.REFERENCE) -> list[Finding]:
    """Check that every column with a closed domain only holds values from it.

    Values are taken from the rows, never from a categorical's categories: after a
    split, the categories of one period still list the values seen in the other.

    Args:
        frame: Rows of one period, with raw (open) categories.
        period: Period of the rows. On production, a finding only says whether every
            value is within the domain; on reference, it also counts and lists the
            values outside it, and names the declared values that never occur.

    Returns:
        One finding per column with a closed domain, plus, on reference, an
        informational one for each column with declared values that never occur.
    """
    findings: list[Finding] = []
    for spec in COLUMN_SPECS:
        if spec.allowed_values is None:
            continue
        check = f"{period.value} domain:{spec.name}"
        if spec.name not in frame.columns:
            findings.append(_skipped(check, "column could not be read"))
            continue
        present = frame[spec.name].dropna()
        inside = present.isin(spec.allowed_values)
        if period is Period.PRODUCTION:
            statement = "every value within the declared domain"
            findings.append(_boolean(check, statement, bool(inside.all())))
            continue
        outside = present[~inside]
        if len(outside):
            values = sorted(outside.astype(str).unique().tolist())
            findings.append(
                _failed(check, f"{len(outside)} rows outside the domain; values: {_listed(values)}")
            )
        else:
            findings.append(
                _passed(check, f"every value in the {len(spec.allowed_values)} declared values")
            )
        observed = set(present.unique().tolist())
        unobserved = [value for value in spec.allowed_values if value not in observed]
        if unobserved:
            findings.append(
                _failed(
                    check,
                    f"declared values that never occur: {_listed(unobserved)}",
                    Severity.INFORMATIONAL,
                )
            )
    return findings


def check_nulls(frame: pd.DataFrame, period: Period = Period.REFERENCE) -> list[Finding]:
    """Check that no column holds missing values unless the schema admits them.

    Args:
        frame: Rows of one period.
        period: Period of the rows. On production, a finding only says whether forbidden
            missing values are present; on reference, it also counts them.

    Returns:
        One failed finding per column with forbidden nulls, one skipped finding per
        declared column that could not be read, and one passed finding for the rest.
    """
    findings: list[Finding] = []
    clean = []
    for spec in COLUMN_SPECS:
        check = f"{period.value} nulls:{spec.name}"
        if spec.name not in frame.columns:
            findings.append(_skipped(check, "column could not be read"))
            continue
        if spec.nullable:
            clean.append(spec.name)
        elif period is Period.PRODUCTION:
            if frame[spec.name].isna().any():
                findings.append(_boolean(check, "no forbidden missing values", False))
            else:
                clean.append(spec.name)
        else:
            missing = int(frame[spec.name].isna().sum())
            if missing:
                findings.append(_failed(check, f"{missing} missing values; the schema admits none"))
            else:
                clean.append(spec.name)
    if clean:
        check = f"{period.value} nulls"
        if period is Period.PRODUCTION:
            findings.append(
                _boolean(check, "no forbidden missing values in the remaining columns", True)
            )
        else:
            findings.append(_passed(check, f"no forbidden missing values in {len(clean)} columns"))
    return findings


def check_parsed_rows(csv_data_lines: int, parsed_rows: int) -> Finding:
    """Check that the parser returned one row per data line of the file.

    Args:
        csv_data_lines: Lines of the file after the header, counted on its raw bytes.
        parsed_rows: Rows the parser returned.

    Returns:
        The finding of the check.
    """
    if parsed_rows == csv_data_lines:
        return _passed("rows:parsed", f"{parsed_rows} rows, one per data line")
    return _failed(
        "rows:parsed",
        f"the parser returned {parsed_rows} rows for {csv_data_lines} data lines; "
        "blank lines are skipped by the parser and a quoted line break merges two",
    )


def check_periods(parsed_rows: int, reference_rows: int, production_rows: int) -> Finding:
    """Check that every parsed row falls in exactly one of the two periods.

    The finding is a boolean, with no count, because the check involves production.

    Args:
        parsed_rows: Rows the parser returned.
        reference_rows: Rows assigned to the reference period.
        production_rows: Rows assigned to the production period.

    Returns:
        The finding of the check.
    """
    holds = reference_rows + production_rows == parsed_rows
    return _boolean("rows:periods", "reference rows + production rows = parsed rows", holds)


def validate_raw_csv(csv_path: Path) -> ValidationReport:
    """Run every check on the raw CSV and return all the findings.

    Data problems never raise here; they become findings. Call ``raise_if_blocking`` on
    the result to stop the pipeline.

    Args:
        csv_path: Raw CSV file.

    Returns:
        The findings of every check, including the ones that could not run.
    """
    header = read_csv_header(csv_path)
    findings = check_columns(header)
    present = [name for name in EXPECTED_COLUMNS if name in header]
    dtype_findings, frame = _read_checking_dtypes(csv_path, present)
    findings.extend(dtype_findings)

    period_checks = [f"{period.value} {kind}" for period in Period for kind in ("domains", "nulls")]
    if frame is None:
        reason = "no column could be read"
        findings.extend(
            _skipped(check, reason) for check in (*period_checks, "rows:parsed", "rows:periods")
        )
        return ValidationReport(source=csv_path.name, findings=tuple(findings))

    csv_data_lines = max(scan_lines(csv_path).line_count - 1, 0)
    findings.append(check_parsed_rows(csv_data_lines, len(frame)))

    step_is_usable = STEP_COLUMN in frame.columns and pd.api.types.is_integer_dtype(
        frame[STEP_COLUMN].dtype
    )
    if not step_is_usable:
        reason = "the period split needs an integer 'step' column without missing values"
        findings.extend(_skipped(check, reason) for check in (*period_checks, "rows:periods"))
        return ValidationReport(source=csv_path.name, findings=tuple(findings))

    reference, production = split_periods(frame)
    for rows, period in ((reference, Period.REFERENCE), (production, Period.PRODUCTION)):
        findings.extend(check_domains(rows, period))
        findings.extend(check_nulls(rows, period))
    findings.append(check_periods(len(frame), len(reference), len(production)))
    return ValidationReport(source=csv_path.name, findings=tuple(findings))
