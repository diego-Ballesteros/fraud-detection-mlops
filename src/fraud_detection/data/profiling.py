"""Measurements of the PaySim data, returned as data for the caller to print or plot.

No function here prints: each one returns a DataFrame or a dataclass. The preparation
script prints them, and notebook 01 imports them to build its narrative.

The file-structure, memory and row-count measurements describe the raw CSV and the period
files as a whole; they are structural. Every other function is meant to receive the
reference period only, because the temporal cut reserves the production period for
phase 4.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data.loader import STEP_COLUMN, LineScan, count_parquet_rows, scan_lines
from fraud_detection.data.schema import BINARY_VALUES, TARGET_COLUMN

TYPE_COLUMN: Final[str] = "type"
AMOUNT_COLUMN: Final[str] = "amount"
FLAG_COLUMN: Final[str] = "isFlaggedFraud"
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = ("nameOrig", "nameDest")
MONEY_COLUMNS: Final[tuple[str, ...]] = (
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

BALANCE_TOLERANCE: Final[float] = 0.005
"""Largest absolute balance error still counted as zero, in currency units.

Half a cent. Amounts and balances are recorded in whole cents (``cent_resolution``
measures it), so an error between recorded values is a whole number of cents plus
floating-point noise, and the noise is orders of magnitude below a cent. Half a cent sits
in the middle of the gap between zero and one cent, with margin on both sides.
"""

SENSITIVITY_TOLERANCES: Final[tuple[float, ...]] = (1e-6, BALANCE_TOLERANCE, 0.015, 1.0)
"""Tolerances for the sensitivity table: float noise only, the declared one, one cent of
rounding absorbed, and one currency unit."""

_CENT_NOISE: Final[float] = 1e-4
"""Largest distance, in cents, between a value times 100 and the nearest integer that is
still read as a whole number of cents. For values below 1e9, the float64 noise of a value
times 100 stays below 2e-5 cents."""


def _label_groups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Row masks for all rows and for each value of the label."""
    label = frame[TARGET_COLUMN]
    groups = {"all": pd.Series(True, index=frame.index)}
    groups.update({f"{TARGET_COLUMN}={value}": label == value for value in BINARY_VALUES})
    return groups


# --- 1-3. Structure of the files ---------------------------------------------------------


@dataclass(frozen=True)
class FileStructure:
    """Structure of the raw CSV, measured on its bytes.

    Attributes:
        name: File name.
        size_bytes: Size on disk.
        lines: Line-ending counts.
        head: The first bytes of the file, undecoded.
    """

    name: str
    size_bytes: int
    lines: LineScan
    head: bytes


def measure_file_structure(csv_path: Path, head_size: int = 160) -> FileStructure:
    """Measure the name, size, line endings and first bytes of the raw CSV.

    Args:
        csv_path: Raw CSV file.
        head_size: How many leading bytes to keep.

    Returns:
        The structure of the whole file.
    """
    with csv_path.open("rb") as stream:
        head = stream.read(head_size)
    return FileStructure(
        name=csv_path.name,
        size_bytes=csv_path.stat().st_size,
        lines=scan_lines(csv_path),
        head=head,
    )


@dataclass(frozen=True)
class RowCounts:
    """Row counts of the raw CSV and of the two period files.

    Attributes:
        csv_data_lines: Lines of the CSV after the header, counted on its bytes.
        reference_rows: Rows in the reference parquet file.
        production_rows: Rows in the production parquet file.
    """

    csv_data_lines: int
    reference_rows: int
    production_rows: int

    @property
    def period_sum(self) -> int:
        """Reference rows plus production rows."""
        return self.reference_rows + self.production_rows


def count_rows(csv_path: Path, reference_path: Path, production_path: Path) -> RowCounts:
    """Count the rows of the CSV and of each period file.

    The period files are counted from their parquet footers: no production row is read.

    Args:
        csv_path: Raw CSV file.
        reference_path: Reference parquet file.
        production_path: Production parquet file.

    Returns:
        The three counts.
    """
    return RowCounts(
        csv_data_lines=max(scan_lines(csv_path).line_count - 1, 0),
        reference_rows=count_parquet_rows(reference_path),
        production_rows=count_parquet_rows(production_path),
    )


# --- 4-6. Quality ------------------------------------------------------------------------


def memory_footprint(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure the dtype and the in-memory size of every column.

    Args:
        frame: Data to measure.

    Returns:
        One row per column, plus a ``TOTAL`` row, with columns ``dtype`` and ``bytes``.
    """
    usage = frame.memory_usage(deep=True, index=False)
    table = pd.DataFrame({"dtype": frame.dtypes.astype(str), "bytes": usage})
    total = pd.DataFrame({"dtype": [""], "bytes": [int(usage.sum())]}, index=["TOTAL"])
    return pd.concat([table, total])


def inferred_memory_footprint(csv_path: Path) -> pd.DataFrame:
    """Measure the CSV as pandas reads it with no dtypes given, inferring them.

    This is the counterfactual the declared dtypes are compared against; the loader never
    reads this way. The frame is dropped on return and only the table leaves.

    Args:
        csv_path: Raw CSV file.

    Returns:
        The table of ``memory_footprint`` for the inferred read.
    """
    return memory_footprint(pd.read_csv(csv_path))


def memory_comparison(inferred: pd.DataFrame, declared: pd.DataFrame) -> pd.DataFrame:
    """Put the inferred and the declared memory footprints side by side.

    Args:
        inferred: ``memory_footprint`` of the read with inferred dtypes.
        declared: ``memory_footprint`` of the read with the declared dtypes.

    Returns:
        One row per column and a ``TOTAL`` row, with both dtypes, both sizes and the
        ratio of declared to inferred bytes.
    """
    table = inferred.add_prefix("inferred_").join(declared.add_prefix("declared_"))
    table["declared/inferred"] = table["declared_bytes"] / table["inferred_bytes"]
    return table


def null_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count the missing values of every column.

    Args:
        frame: Data to measure.

    Returns:
        One row per column with the count and the percentage of missing values.
    """
    nulls = frame.isna().sum()
    return pd.DataFrame({"nulls": nulls, "percent": 100 * nulls / len(frame)})


@dataclass(frozen=True)
class DuplicateRows:
    """Rows identical to another row across every column.

    The dataset has no transaction identifier, so there is no narrower key: a duplicate
    is a row whose eleven values all repeat another row's.

    Attributes:
        rows: Rows measured.
        redundant_copies: Rows identical to an earlier row.
        rows_in_duplicate_groups: Rows that have at least one identical row, counting
            every member of each group.
    """

    rows: int
    redundant_copies: int
    rows_in_duplicate_groups: int


def duplicate_rows(frame: pd.DataFrame) -> DuplicateRows:
    """Count the rows that exactly duplicate another row across every column.

    Args:
        frame: Data to measure.

    Returns:
        The duplicate counts.
    """
    redundant = int(frame.duplicated(keep="first").sum())
    in_groups = int(frame.duplicated(keep=False).sum()) if redundant else 0
    return DuplicateRows(
        rows=len(frame), redundant_copies=redundant, rows_in_duplicate_groups=in_groups
    )


# --- 7-10. The phenomenon -----------------------------------------------------------------


def class_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Count the rows of each label value.

    Args:
        frame: Data to measure.

    Returns:
        One row per label value and a ``total`` row, with the count and the proportion.
    """
    counts = frame[TARGET_COLUMN].value_counts().reindex(BINARY_VALUES, fill_value=0)
    counts.loc["total"] = counts.sum()
    return pd.DataFrame({"rows": counts, "proportion": counts / len(frame)})


@dataclass(frozen=True)
class LabelByType:
    """Label counts per transaction type.

    Attributes:
        counts: Rows per type and label value, with totals, in absolute counts.
        types_without_positives: Types that occur and never carry the positive label.
        unobserved_types: Declared types with no row at all.
    """

    counts: pd.DataFrame
    types_without_positives: tuple[str, ...]
    unobserved_types: tuple[str, ...]


def label_by_type(frame: pd.DataFrame) -> LabelByType:
    """Cross-tabulate the label against the transaction type, in absolute counts.

    Args:
        frame: Data to measure.

    Returns:
        The table and the types where the positive class never occurs.
    """
    counts = (
        frame.groupby([TYPE_COLUMN, TARGET_COLUMN], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=BINARY_VALUES, fill_value=0)
    )
    # A plain index: a CategoricalIndex refuses the "total" row, which is no category.
    counts.index = counts.index.astype(str)
    counts["total"] = counts.sum(axis=1)
    observed = counts[counts["total"] > 0]
    positive = BINARY_VALUES[1]
    without_positives = tuple(str(name) for name in observed.index[observed[positive] == 0])
    unobserved = tuple(str(name) for name in counts.index[counts["total"] == 0])
    counts.loc["total"] = counts.sum()
    return LabelByType(counts, without_positives, unobserved)


@dataclass(frozen=True)
class StepDistribution:
    """Distribution of rows over ``step``.

    Attributes:
        first: Smallest observed step.
        last: Largest observed step.
        distinct: Number of distinct observed steps.
        missing: Steps between ``first`` and the period's last step with no row.
        rows_per_step: Rows per step over that whole range, zeros included.
        summary: Count, mean, standard deviation, minimum, quartiles and maximum of
            ``rows_per_step``.
    """

    first: int
    last: int
    distinct: int
    missing: tuple[int, ...]
    rows_per_step: pd.Series
    summary: pd.Series


def step_distribution(
    frame: pd.DataFrame, period_last_step: int = settings.REFERENCE_LAST_STEP
) -> StepDistribution:
    """Measure the range of ``step``, its gaps and the number of rows per step.

    Args:
        frame: Data to measure.
        period_last_step: Last step of the period the frame belongs to.

    Returns:
        The distribution, with rows per step over the range from the first observed step
        to ``period_last_step``.
    """
    step = frame[STEP_COLUMN]
    observed = step.value_counts().sort_index()
    first = int(step.min())
    expected = range(first, period_last_step + 1)
    rows_per_step = observed.reindex(expected, fill_value=0)
    return StepDistribution(
        first=first,
        last=int(step.max()),
        distinct=len(observed),
        missing=tuple(int(value) for value in rows_per_step.index[rows_per_step == 0]),
        rows_per_step=rows_per_step,
        summary=rows_per_step.describe(),
    )


def identifier_repetition(
    frame: pd.DataFrame, columns: Sequence[str] = IDENTIFIER_COLUMNS
) -> pd.DataFrame:
    """Measure the cardinality of each identifier and how often its values repeat.

    Args:
        frame: Data to measure.
        columns: Identifier columns.

    Returns:
        One row per identifier: rows, distinct values, values that occur more than once,
        rows those values cover, and the largest number of occurrences of one value.
    """
    records = {}
    for name in columns:
        counts = frame[name].value_counts()
        repeated = counts[counts > 1]
        records[name] = {
            "rows": len(frame),
            "distinct": len(counts),
            "values_repeated": len(repeated),
            "rows_with_repeated_value": int(repeated.sum()),
            "max_occurrences": int(counts.max()),
        }
    return pd.DataFrame.from_dict(records, orient="index")


# --- 11-14. Leakage hunt ------------------------------------------------------------------


@dataclass(frozen=True)
class FlagProfile:
    """What the rows flagged by the system rule have in common.

    Attributes:
        flagged_rows: Rows where the flag is positive.
        by_label: Rows per flag value and label value.
        types_when_flagged: Rows per transaction type among the flagged rows.
        amount_stats: Count, minimum, median and maximum of the amount, for the flagged
            rows and for the rest.
    """

    flagged_rows: int
    by_label: pd.DataFrame
    types_when_flagged: pd.Series
    amount_stats: pd.DataFrame


def flag_profile(frame: pd.DataFrame) -> FlagProfile:
    """Profile the system flag: its count, its cross-table with the label, and its rows.

    Args:
        frame: Data to measure.

    Returns:
        The profile of the flagged rows against the rest.
    """
    flagged = frame[FLAG_COLUMN] == BINARY_VALUES[1]
    by_label = (
        frame.groupby([FLAG_COLUMN, TARGET_COLUMN])
        .size()
        .unstack(fill_value=0)
        .reindex(index=BINARY_VALUES, columns=BINARY_VALUES, fill_value=0)
    )
    amount = frame[AMOUNT_COLUMN]
    amount_stats = pd.DataFrame(
        {
            group: {
                "rows": int(mask.sum()),
                "min": amount[mask].min(),
                "median": amount[mask].median(),
                "max": amount[mask].max(),
            }
            for group, mask in (("flagged", flagged), ("not flagged", ~flagged))
        }
    ).T
    return FlagProfile(
        flagged_rows=int(flagged.sum()),
        by_label=by_label,
        types_when_flagged=frame.loc[flagged, TYPE_COLUMN].value_counts(),
        amount_stats=amount_stats,
    )


def balance_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the two balance-arithmetic errors of every row.

    ``origin_error = oldbalanceOrg - amount - newbalanceOrig`` and
    ``destination_error = oldbalanceDest + amount - newbalanceDest``. Both are zero when
    the recorded balances follow from the amount.

    Args:
        frame: Data to measure.

    Returns:
        One row per input row, with columns ``origin_error`` and ``destination_error``.
    """
    amount = frame[AMOUNT_COLUMN]
    return pd.DataFrame(
        {
            "origin_error": frame["oldbalanceOrg"] - amount - frame["newbalanceOrig"],
            "destination_error": frame["oldbalanceDest"] + amount - frame["newbalanceDest"],
        }
    )


def balance_consistency(frame: pd.DataFrame, tolerance: float = BALANCE_TOLERANCE) -> pd.DataFrame:
    """Measure how often each balance error is zero, overall and per label value.

    Args:
        frame: Data to measure.
        tolerance: Largest absolute error counted as zero.

    Returns:
        One row per group (all rows, then each label value): its rows, and for each side
        the rows with a zero error and their share.
    """
    is_zero = balance_errors(frame).abs() <= tolerance
    records = {}
    for group, mask in _label_groups(frame).items():
        zero = is_zero[mask]
        records[group] = {
            "rows": int(mask.sum()),
            "origin_zero": int(zero["origin_error"].sum()),
            "origin_share": zero["origin_error"].mean(),
            "destination_zero": int(zero["destination_error"].sum()),
            "destination_share": zero["destination_error"].mean(),
        }
    return pd.DataFrame.from_dict(records, orient="index")


def balance_tolerance_sensitivity(
    frame: pd.DataFrame, tolerances: Sequence[float] = SENSITIVITY_TOLERANCES
) -> pd.DataFrame:
    """Measure the share of zero balance errors per group under several tolerances.

    Args:
        frame: Data to measure.
        tolerances: Tolerances to compare.

    Returns:
        One row per tolerance; one column per side and group with the share of rows
        whose error is zero.
    """
    absolute = balance_errors(frame).abs()
    groups = _label_groups(frame)
    records = {}
    for tolerance in tolerances:
        is_zero = absolute <= tolerance
        records[tolerance] = {
            f"{side.removesuffix('_error')} {group}": is_zero.loc[mask, side].mean()
            for side in absolute.columns
            for group, mask in groups.items()
        }
    table = pd.DataFrame.from_dict(records, orient="index")
    table.index.name = "tolerance"
    return table


def balance_error_magnitude(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe the absolute balance errors per side and per label value.

    Args:
        frame: Data to measure.

    Returns:
        One row per side and label value, with the count, mean, standard deviation,
        minimum, quartiles and maximum of the absolute error.
    """
    absolute = balance_errors(frame).abs()
    absolute[TARGET_COLUMN] = frame[TARGET_COLUMN]
    described = absolute.groupby(TARGET_COLUMN).describe()
    return described.stack(level=0, future_stack=True).swaplevel().sort_index()


def cent_resolution(frame: pd.DataFrame, columns: Sequence[str] = MONEY_COLUMNS) -> pd.DataFrame:
    """Measure the share of values recorded as a whole number of cents.

    Args:
        frame: Data to measure.
        columns: Money columns.

    Returns:
        One row per column: rows that are not whole cents and their share.
    """
    records = {}
    for name in columns:
        cents = frame[name] * 100
        not_whole = (cents - cents.round()).abs() > _CENT_NOISE
        records[name] = {"not_whole_cents": int(not_whole.sum()), "share": not_whole.mean()}
    return pd.DataFrame.from_dict(records, orient="index")


def _destination_both_zero(frame: pd.DataFrame) -> pd.Series:
    """Rows where both destination balances are zero while the amount is positive."""
    return (
        (frame["oldbalanceDest"] == 0) & (frame["newbalanceDest"] == 0) & (frame[AMOUNT_COLUMN] > 0)
    )


def zero_destination_balances(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure how often both destination balances are zero while money moves.

    Counts rows with ``oldbalanceDest == 0``, ``newbalanceDest == 0`` and ``amount > 0``.

    Args:
        frame: Data to measure.

    Returns:
        One row per group (all rows, then each label value): its rows, the matching
        rows and their share.
    """
    matches = _destination_both_zero(frame)
    records = {
        group: {
            "rows": int(mask.sum()),
            "both_zero_with_amount": int(matches[mask].sum()),
            "share": matches[mask].mean(),
        }
        for group, mask in _label_groups(frame).items()
    }
    return pd.DataFrame.from_dict(records, orient="index")


def label_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlate every numeric column with the label.

    An input to the leakage discussion, not a selection criterion. Pearson measures the
    linear relation; Spearman, the monotonic one, which is less dominated by the heavy
    tails of the money columns. Spearman is computed by its definition, the Pearson
    correlation of average ranks: pandas' ``method="spearman"`` imports scipy at call
    time, and scipy is not a dependency of the project.

    Args:
        frame: Data to measure.

    Returns:
        One row per numeric column other than the label, with both coefficients.
    """
    numeric = [
        name
        for name in frame.columns
        if name != TARGET_COLUMN and pd.api.types.is_numeric_dtype(frame[name])
    ]
    label = frame[TARGET_COLUMN]
    return pd.DataFrame(
        {
            "pearson": frame[numeric].corrwith(label),
            "spearman": frame[numeric].rank().corrwith(label.rank()),
        }
    )


# --- Beyond the requested list: what measurements 9 and 12 made suspicious ----------------


def balance_consistency_by_type(
    frame: pd.DataFrame, tolerance: float = BALANCE_TOLERANCE
) -> pd.DataFrame:
    """Measure measurements 12 and 13 per transaction type and label value.

    The origin formula assumes that money leaves the origin account, which CASH_IN
    contradicts by construction, and the positive class only occurs in some types. The
    overall shares of ``balance_consistency`` and ``zero_destination_balances`` mix both
    effects; this table separates them.

    Args:
        frame: Data to measure.
        tolerance: Largest absolute error counted as zero.

    Returns:
        One row per observed type and label value: its rows, the share with a zero origin
        error, the share with a zero destination error, and the share with both
        destination balances zero while the amount is positive.
    """
    is_zero = balance_errors(frame).abs() <= tolerance
    table = pd.DataFrame(
        {
            TYPE_COLUMN: frame[TYPE_COLUMN],
            TARGET_COLUMN: frame[TARGET_COLUMN],
            "origin_zero_share": is_zero["origin_error"],
            "destination_zero_share": is_zero["destination_error"],
            "destination_both_zero_share": _destination_both_zero(frame),
        }
    )
    grouped = table.groupby([TYPE_COLUMN, TARGET_COLUMN], observed=True)
    shares = grouped.mean()
    shares.insert(0, "rows", grouped.size())
    return shares


@dataclass(frozen=True)
class LabelByStep:
    """How the positive class spreads over ``step``.

    Attributes:
        positives_per_step: Positive rows per step over the period, zeros included.
        summary: Count, mean, standard deviation, minimum, quartiles and maximum of
            ``positives_per_step``.
        volume_correlation: Pearson correlation, across steps, between all rows per step
            and positive rows per step.
        steps_all_positive: Steps with at least one row where every row is positive.
        positives_in_those_steps: Positive rows that fall in those steps.
        total_positives: Positive rows in the period.
    """

    positives_per_step: pd.Series
    summary: pd.Series
    volume_correlation: float
    steps_all_positive: int
    positives_in_those_steps: int
    total_positives: int


def label_by_step(
    frame: pd.DataFrame, period_last_step: int = settings.REFERENCE_LAST_STEP
) -> LabelByStep:
    """Measure the positive class per step against the volume of rows per step.

    Args:
        frame: Data to measure.
        period_last_step: Last step of the period the frame belongs to.

    Returns:
        The positives per step and how they relate to the volume per step.
    """
    step = frame[STEP_COLUMN]
    expected = range(int(step.min()), period_last_step + 1)
    rows = step.value_counts().reindex(expected, fill_value=0)
    positive = frame[TARGET_COLUMN] == BINARY_VALUES[1]
    positives = step[positive].value_counts().reindex(expected, fill_value=0)
    all_positive = (rows > 0) & (positives == rows)
    return LabelByStep(
        positives_per_step=positives,
        summary=positives.describe(),
        volume_correlation=float(rows.corr(positives)),
        steps_all_positive=int(all_positive.sum()),
        positives_in_those_steps=int(positives[all_positive].sum()),
        total_positives=int(positive.sum()),
    )


def amount_equals_origin_balance(
    frame: pd.DataFrame, tolerance: float = BALANCE_TOLERANCE
) -> pd.DataFrame:
    """Measure how often the amount equals the origin's balance before the transaction.

    Both columns are declared PRE_TRANSACTION. A zero origin error with a zero balance
    after the transaction implies this equality, so this measures how much of the origin
    error's signal is already present before the transaction.

    Args:
        frame: Data to measure.
        tolerance: Largest absolute difference counted as equal.

    Returns:
        One row per observed type and label value: its rows, the rows where the amount
        equals ``oldbalanceOrg``, and their share.
    """
    equal = (frame[AMOUNT_COLUMN] - frame["oldbalanceOrg"]).abs() <= tolerance
    table = pd.DataFrame(
        {TYPE_COLUMN: frame[TYPE_COLUMN], TARGET_COLUMN: frame[TARGET_COLUMN], "equal": equal}
    )
    grouped = table.groupby([TYPE_COLUMN, TARGET_COLUMN], observed=True)["equal"]
    return pd.DataFrame(
        {"rows": grouped.size(), "amount_equals_balance": grouped.sum(), "share": grouped.mean()}
    )
