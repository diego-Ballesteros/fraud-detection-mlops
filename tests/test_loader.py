"""Tests of fraud_detection.data.loader.

Two kinds of test, and they measure different things:

- Fast tests on a synthetic CSV built here with the shape of the schema. They exercise
  the loader's logic on rows chosen by hand: parsing, the temporal cut with a row on each
  side of the boundary, and the parquet round trip.
- One slow test on the real files, counts only: the data lines of the real CSV against
  the rows of the two parquet files written by scripts/prepare_data.py. It is skipped
  where the files are absent, as in CI, because data/ is gitignored.
"""

from pathlib import Path

import pyarrow as pa
import pytest

from fraud_detection.config import settings
from fraud_detection.data import loader
from fraud_detection.data.schema import COLUMN_SPECS, EXPECTED_COLUMNS, TRANSACTION_TYPES

LAST = settings.REFERENCE_LAST_STEP
FIRST = settings.PRODUCTION_FIRST_STEP

# Columns in schema order. The boundary steps sit right on each side of the cut.
SYNTHETIC_ROWS: list[tuple[object, ...]] = [
    (1, "PAYMENT", 9839.64, "C1231006815", 170136.0, 160296.36, "M1979787155", 0.0, 0.0, 0, 0),
    (LAST - 1, "TRANSFER", 181.0, "C1305486145", 181.0, 0.0, "C553264065", 0.0, 0.0, 1, 0),
    (LAST, "CASH_OUT", 181.0, "C840083671", 181.0, 0.0, "C38997010", 21182.0, 0.0, 1, 0),
    (LAST, "CASH_IN", 1500.5, "C1", 10.0, 1510.5, "C2", 0.0, 0.0, 0, 0),
    (FIRST, "DEBIT", 50.0, "C3", 100.0, 50.0, "C4", 0.0, 50.0, 0, 0),
    (FIRST, "TRANSFER", 4953893.08, "C5", 4953893.08, 4953893.08, "C6", 0.0, 0.0, 1, 1),
    (FIRST + 300, "PAYMENT", 12.34, "C7", 12.34, 0.0, "M8", 0.0, 0.0, 0, 0),
]


def _write_csv(path: Path, rows: list[tuple[object, ...]]) -> Path:
    # CRLF terminators, like the real file.
    lines = [",".join(EXPECTED_COLUMNS)] + [",".join(str(value) for value in row) for row in rows]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path


def _with_step(row: tuple[object, ...], step: int) -> tuple[object, ...]:
    return (step, *row[1:])


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    return _write_csv(tmp_path / "synthetic.csv", SYNTHETIC_ROWS)


@pytest.fixture
def period_paths(synthetic_csv: Path, tmp_path: Path) -> tuple[Path, Path]:
    paths = (tmp_path / "reference.parquet", tmp_path / "production.parquet")
    reference, production = loader.split_periods(loader.load_csv(synthetic_csv))
    loader.write_periods(reference, production, *paths)
    return paths


def test_periods_do_not_cross_the_temporal_cut(period_paths: tuple[Path, Path]) -> None:
    reference_path, production_path = period_paths
    reference = loader.read_reference(reference_path)
    production = loader.read_production(production_path)
    assert reference["step"].max() <= LAST
    assert production["step"].min() >= FIRST
    # Control: the boundary steps exist on their own side, so neither assertion above can
    # pass on an empty or one-sided period.
    assert LAST in set(reference["step"])
    assert FIRST in set(production["step"])


def test_period_rows_sum_to_the_csv_rows(
    synthetic_csv: Path, period_paths: tuple[Path, Path]
) -> None:
    csv_rows = loader.scan_lines(synthetic_csv).line_count - 1
    # Control: the line scan counts exactly the rows written.
    assert csv_rows == len(SYNTHETIC_ROWS)
    assert sum(loader.count_parquet_rows(path) for path in period_paths) == csv_rows


def test_parquet_dtypes_match_the_schema(period_paths: tuple[Path, Path]) -> None:
    reference_path, production_path = period_paths
    for frame in (loader.read_reference(reference_path), loader.read_production(production_path)):
        mismatched = {
            spec.name: (str(frame[spec.name].dtype), str(spec.dtype))
            for spec in COLUMN_SPECS
            if frame[spec.name].dtype != spec.dtype
        }
        assert mismatched == {}
        # Equality between unordered categoricals ignores the order of the categories.
        assert tuple(frame["type"].cat.categories) == TRANSACTION_TYPES


def test_synthetic_rows_cover_every_declared_type() -> None:
    # Guards the fixture: the dtype test checks categories that the rows actually use.
    assert {row[1] for row in SYNTHETIC_ROWS} == set(TRANSACTION_TYPES)


def test_out_of_range_integer_is_rejected_not_wrapped(tmp_path: Path) -> None:
    # pandas.read_csv(dtype={"step": "int16"}) reads 40000 as -25536 without an error.
    path = _write_csv(tmp_path / "overflow.csv", [_with_step(SYNTHETIC_ROWS[0], 40000)])
    with pytest.raises(pa.ArrowInvalid, match="int16"):
        loader.read_raw_csv(path)


def test_an_empty_column_list_reads_nothing(synthetic_csv: Path) -> None:
    # pyarrow alone would read every column, with inferred types (ERR-008).
    with pytest.raises(ValueError, match="No column requested"):
        loader.read_raw_csv(synthetic_csv, [])


def test_out_of_domain_type_is_refused_not_nulled(tmp_path: Path) -> None:
    row = SYNTHETIC_ROWS[0]
    path = _write_csv(tmp_path / "domain.csv", [(row[0], "REFUND", *row[2:])])
    raw = loader.read_raw_csv(path)
    # The parse keeps the value, so the validator can see it.
    assert raw["type"].astype(str).tolist() == ["REFUND"]
    with pytest.raises(ValueError, match="missing values"):
        loader.apply_declared_dtypes(raw)


def test_read_reference_refuses_a_production_row(synthetic_csv: Path, tmp_path: Path) -> None:
    both_periods = loader.load_csv(synthetic_csv)
    path = tmp_path / "contaminated.parquet"
    both_periods.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="past the reference period"):
        loader.read_reference(path)


requires_real_files = pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (loader.RAW_CSV_PATH, loader.REFERENCE_PATH, loader.PRODUCTION_PATH)
    ),
    reason="real CSV or period files absent (data/ is gitignored); run scripts/prepare_data.py",
)


@requires_real_files
def test_real_period_files_hold_every_csv_row() -> None:
    # Real files, counts only: production is counted from its footer, no row is read.
    csv_rows = loader.scan_lines(loader.RAW_CSV_PATH).line_count - 1
    reference_rows = loader.count_parquet_rows(loader.REFERENCE_PATH)
    production_rows = loader.count_parquet_rows(loader.PRODUCTION_PATH)
    assert reference_rows > 0
    assert production_rows > 0
    assert reference_rows + production_rows == csv_rows
