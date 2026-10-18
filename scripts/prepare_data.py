"""Prepare the PaySim data: validate the raw CSV, write the two period files, and measure.

Run from the repository root::

    uv run python scripts/prepare_data.py

Steps, in order: measure the structure of the raw file; validate it, collecting every
finding and stopping if any is blocking; load it with the declared dtypes and compare its
memory against an inferred read; split it at the temporal cut and write one parquet file
per period; count the rows; and measure the reference period.

Every measurement after the row counts runs on the reference period only. The temporal
cut reserves the production period for phase 4, and this script never reads its rows
back: production is counted from its parquet footer.
"""

import time
from typing import Final

import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data import loader, profiling, validator

PRIOR_INFERRED_LINE_COUNT: Final[int] = 6_362_621
"""Line count inferred in an earlier turn, never measured (docs/ERRORS_AND_LEARNINGS.md,
ERR-001 and ERR-002). This script measures it."""

PERIODICITY_WINDOW: Final[int] = 48
MAX_LISTED_STEPS: Final[int] = 30


def _section(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


def _show(table: pd.DataFrame | pd.Series) -> None:
    print(table.to_string())


def _report_file_structure() -> None:
    _section("1-2. FILE STRUCTURE (whole file: structural)")
    structure = profiling.measure_file_structure(loader.RAW_CSV_PATH)
    lines = structure.lines
    print(f"file name             : {structure.name}")
    print(f"size in bytes         : {structure.size_bytes}")
    print(f"LF bytes (\\n)         : {lines.lf}")
    print(f"CRLF pairs (\\r\\n)     : {lines.crlf}")
    print(f"lone CR bytes (\\r)    : {lines.lone_cr}")
    print(f"ends with a newline   : {lines.ends_with_newline}")
    print(f"line count (measured) : {lines.line_count}")
    print(f"line count (inferred) : {PRIOR_INFERRED_LINE_COUNT}")
    difference = lines.line_count - PRIOR_INFERRED_LINE_COUNT
    print(f"inference confirmed   : {difference == 0} (measured - inferred = {difference})")
    first_line_end = structure.head.index(b"\n") + 1
    print(f"first {len(structure.head)} bytes, raw:")
    print(f"  {structure.head!r}")
    print("last 6 bytes of the first line, hex:")
    print(f"  {structure.head[first_line_end - 6 : first_line_end].hex(' ')}")


def _validate() -> bool:
    _section("VALIDATION (structure on the whole file; content on the reference period)")
    report = validator.validate_raw_csv(loader.RAW_CSV_PATH)
    print(report.render())
    return not report.blocking_failures


def _load_and_write() -> None:
    _section("LOAD: MEMORY BEFORE AND AFTER (whole file, both periods: structural)")
    started = time.perf_counter()
    inferred = profiling.inferred_memory_footprint(loader.RAW_CSV_PATH)
    elapsed = time.perf_counter() - started
    print(f"before: pandas.read_csv with inferred dtypes ({elapsed:.1f} s)")
    started = time.perf_counter()
    frame = loader.load_csv(loader.RAW_CSV_PATH)
    elapsed = time.perf_counter() - started
    print(f"after : loader.load_csv with the declared dtypes ({elapsed:.1f} s)")
    _show(profiling.memory_comparison(inferred, profiling.memory_footprint(frame)))

    _section("WRITE: ONE PARQUET FILE PER PERIOD")
    reference, production = loader.split_periods(frame)
    del frame
    loader.write_periods(reference, production)
    for path in (loader.REFERENCE_PATH, loader.PRODUCTION_PATH):
        print(f"{path.relative_to(settings.ROOT_DIR)}: {path.stat().st_size} bytes")


def _report_row_counts() -> None:
    _section("3. ROW COUNTS (CSV counted on its bytes; periods counted from parquet footers)")
    counts = profiling.count_rows(
        loader.RAW_CSV_PATH, loader.REFERENCE_PATH, loader.PRODUCTION_PATH
    )
    print(f"total (CSV data lines)          : {counts.csv_data_lines}")
    print(f"reference (step <= {settings.REFERENCE_LAST_STEP})        : {counts.reference_rows}")
    print(f"production (step >= {settings.PRODUCTION_FIRST_STEP})       : {counts.production_rows}")
    print(f"reference + production          : {counts.period_sum}")
    print(f"sum equals total                : {counts.period_sum == counts.csv_data_lines}")


def _report_quality(reference: pd.DataFrame) -> None:
    _section("4. DTYPES AS READ AND MEMORY (reference period)")
    _show(profiling.memory_footprint(reference))

    _section("5. NULLS PER COLUMN (reference period)")
    _show(profiling.null_counts(reference))

    _section("6. DUPLICATE ROWS (reference period; identical across all 11 columns)")
    duplicates = profiling.duplicate_rows(reference)
    print(f"rows measured                     : {duplicates.rows}")
    print(f"rows identical to an earlier row  : {duplicates.redundant_copies}")
    print(f"rows in a group of identical rows : {duplicates.rows_in_duplicate_groups}")


def _report_phenomenon(reference: pd.DataFrame) -> None:
    _section("7. CLASS BALANCE (reference period)")
    _show(profiling.class_balance(reference))

    _section("8. LABEL BY TRANSACTION TYPE, ABSOLUTE COUNTS (reference period)")
    by_type = profiling.label_by_type(reference)
    _show(by_type.counts)
    print(f"types where the positive class never occurs: {list(by_type.types_without_positives)}")
    print(f"declared types with no row at all          : {list(by_type.unobserved_types)}")

    _section("9. DISTRIBUTION OF STEP (reference period)")
    steps = profiling.step_distribution(reference)
    print(f"first observed step : {steps.first}")
    print(f"last observed step  : {steps.last}")
    print(f"distinct steps      : {steps.distinct}")
    shown = list(steps.missing[:MAX_LISTED_STEPS])
    more = len(steps.missing) - len(shown)
    print(
        f"steps in [{steps.first}, {settings.REFERENCE_LAST_STEP}] with no row: "
        f"{len(steps.missing)} -> {shown}{f' ... and {more} more' if more > 0 else ''}"
    )
    print("rows per step over that range, zeros included:")
    _show(steps.summary)
    window = steps.rows_per_step.iloc[:PERIODICITY_WINDOW]
    print(f"rows per step, first {PERIODICITY_WINDOW} steps of the range (two blocks of 24):")
    half = PERIODICITY_WINDOW // 2
    blocks = pd.DataFrame(
        {
            "step": window.index[:half],
            "rows": window.to_numpy()[:half],
            "step+24": window.index[half:],
            "rows+24": window.to_numpy()[half:],
        }
    )
    print(blocks.to_string(index=False))

    _section("10. IDENTIFIER CARDINALITY AND REPETITION (reference period)")
    _show(profiling.identifier_repetition(reference))


def _report_leakage(reference: pd.DataFrame) -> None:
    _section("11. THE SYSTEM FLAG isFlaggedFraud (reference period)")
    flag = profiling.flag_profile(reference)
    print(f"rows with isFlaggedFraud == 1: {flag.flagged_rows}")
    print("rows by isFlaggedFraud (rows) and isFraud (columns):")
    _show(flag.by_label)
    print("transaction types among the flagged rows:")
    _show(flag.types_when_flagged)
    print("amount among the flagged rows against the rest:")
    _show(flag.amount_stats)

    _section("12. BALANCE ARITHMETIC (reference period)")
    print("recording resolution: values that are not a whole number of cents")
    _show(profiling.cent_resolution(reference))
    print(
        f"\ndeclared tolerance: |error| <= {profiling.BALANCE_TOLERANCE} counts as zero "
        "(half a cent)"
    )
    print("error_origin  = oldbalanceOrg  - amount - newbalanceOrig")
    print("error_destino = oldbalanceDest + amount - newbalanceDest")
    _show(profiling.balance_consistency(reference))
    print("\nsensitivity: share of rows with a zero error under each tolerance")
    _show(profiling.balance_tolerance_sensitivity(reference))
    print("\nmagnitude of the absolute error, per side and per isFraud value")
    _show(profiling.balance_error_magnitude(reference))

    _section("13. DESTINATION BALANCES BOTH ZERO WHILE amount > 0 (reference period)")
    _show(profiling.zero_destination_balances(reference))

    _section("14. CORRELATION OF THE NUMERIC COLUMNS WITH isFraud (reference period)")
    print("An input to the leakage discussion, not a selection criterion.")
    print("Spearman = Pearson correlation of average ranks.")
    _show(profiling.label_correlations(reference))


def _report_extra(reference: pd.DataFrame) -> None:
    _section("EXTRA A. MEASUREMENTS 12 AND 13 PER TRANSACTION TYPE (reference period)")
    _show(profiling.balance_consistency_by_type(reference))

    _section("EXTRA B. SUSPECTED: step. POSITIVES PER STEP AGAINST VOLUME (reference period)")
    by_step = profiling.label_by_step(reference)
    print("positive rows per step over the period, zeros included:")
    _show(by_step.summary)
    print(f"correlation across steps, rows vs positives : {by_step.volume_correlation:.4f}")
    print(f"steps where every row is positive           : {by_step.steps_all_positive}")
    print(
        f"positives in those steps                    : {by_step.positives_in_those_steps} "
        f"of {by_step.total_positives}"
    )
    window = by_step.positives_per_step.iloc[:PERIODICITY_WINDOW]
    print(f"positives per step, first {PERIODICITY_WINDOW} steps: {window.tolist()}")

    _section("EXTRA C. SUSPECTED: oldbalanceOrg WITH amount. amount == oldbalanceOrg (reference)")
    print(f"equal within the declared tolerance, {profiling.BALANCE_TOLERANCE}")
    _show(profiling.amount_equals_origin_balance(reference))


def main() -> int:
    """Validate, write and measure; return the process exit code.

    Returns:
        0 when every step ran, 1 when the validation found a blocking problem.
    """
    started = time.perf_counter()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", "{:.10g}".format)

    _report_file_structure()
    if not _validate():
        print("\nBlocking findings: nothing was written and nothing was measured.")
        return 1
    _load_and_write()
    _report_row_counts()

    reference = loader.read_reference()
    _report_quality(reference)
    _report_phenomenon(reference)
    _report_leakage(reference)
    _report_extra(reference)
    print(f"\nTotal time: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
