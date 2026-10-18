"""Load the raw PaySim CSV with the declared dtypes and split it into two physical periods.

The reference period (``step <= REFERENCE_LAST_STEP``) and the production period
(``step >= PRODUCTION_FIRST_STEP``) are written to two separate parquet files on purpose.
With a single file, respecting the temporal cut depends on every reader applying the
right filter, which is level 3 in docs/METHODOLOGY.md, section 6.5. With two files,
reading production requires naming it, and that shows up in the diff.

The CSV is parsed by pyarrow with an explicit type per column, never by
``pandas.read_csv``. Measured with pandas 2.3.3, ``read_csv(dtype={"step": "int16"})``
turns 40000 into -25536 without an error, with both the C and the pyarrow engines;
pyarrow's own parser rejects the value and names the column.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from fraud_detection.config import settings
from fraud_detection.data.schema import COLUMN_SPECS, COLUMNS, EXPECTED_COLUMNS

RAW_CSV_PATH: Final[Path] = settings.RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv"
REFERENCE_PATH: Final[Path] = settings.PROCESSED_DATA_DIR / "reference.parquet"
PRODUCTION_PATH: Final[Path] = settings.PROCESSED_DATA_DIR / "production.parquet"

STEP_COLUMN: Final[str] = "step"

_STRING_DTYPE: Final = pd.StringDtype("pyarrow")
# A CSV parse yields arrow ``string``; pandas writes StringDtype("pyarrow") to parquet as
# ``large_string``. Both map back to the declared dtype.
_ARROW_TO_PANDAS: Final = {pa.string(): _STRING_DTYPE, pa.large_string(): _STRING_DTYPE}
_SCAN_CHUNK_BYTES: Final[int] = 1 << 24


@dataclass(frozen=True)
class LineScan:
    r"""Line-ending counts of a file, measured on its raw bytes.

    Attributes:
        lf: Number of ``\n`` bytes, each one ending a line.
        crlf: How many of those ``\n`` are preceded by ``\r``.
        lone_cr: ``\r`` bytes not followed by ``\n``.
        ends_with_newline: Whether the last byte of the file is ``\n``.
    """

    lf: int
    crlf: int
    lone_cr: int
    ends_with_newline: bool

    @property
    def line_count(self) -> int:
        """Number of lines, counting a final line that has no line terminator."""
        unterminated_last_line = self.lf > 0 and not self.ends_with_newline
        return self.lf + int(unterminated_last_line)


def scan_lines(path: Path) -> LineScan:
    """Count the line endings of a file by reading its bytes, without parsing it.

    Args:
        path: File to scan.

    Returns:
        The counts of line-ending bytes.
    """
    lf = crlf = carriage_returns = 0
    previous_last = b""
    with path.open("rb") as stream:
        while chunk := stream.read(_SCAN_CHUNK_BYTES):
            lf += chunk.count(b"\n")
            crlf += chunk.count(b"\r\n")
            carriage_returns += chunk.count(b"\r")
            # A CRLF split across two chunks is invisible to either chunk's count.
            if previous_last == b"\r" and chunk.startswith(b"\n"):
                crlf += 1
            previous_last = chunk[-1:]
    return LineScan(
        lf=lf,
        crlf=crlf,
        lone_cr=carriage_returns - crlf,
        ends_with_newline=previous_last == b"\n",
    )


def read_csv_header(csv_path: Path) -> tuple[str, ...]:
    """Read the column names from the first line of the CSV, without parsing any row.

    Args:
        csv_path: Raw CSV file.

    Returns:
        The header fields, in file order.
    """
    with csv_path.open(encoding="utf-8", newline="") as stream:
        first_line = stream.readline()
    return tuple(first_line.rstrip("\r\n").split(","))


def _arrow_read_type(dtype: object) -> pa.DataType:
    """Arrow type a declared dtype is parsed as."""
    if isinstance(dtype, pd.CategoricalDtype):
        # Open dictionary: a value outside the declared categories survives to the
        # validator. Casting to the declared categories instead turns it into NaN without
        # an error (measured with pandas 2.3.3), which would be a silent imputation.
        return pa.dictionary(pa.int32(), pa.string())
    if isinstance(dtype, pd.StringDtype):
        return pa.string()
    return pa.from_numpy_dtype(dtype)


def read_raw_csv(csv_path: Path, columns: Sequence[str] = EXPECTED_COLUMNS) -> pd.DataFrame:
    """Parse the CSV with each column's declared type, checked by pyarrow.

    An integer outside the declared width, a decimal in an integer column or text in a
    numeric column raises instead of being wrapped or coerced. Only an empty field is
    read as missing. Categorical columns keep open categories, so the result is raw:
    ``apply_declared_dtypes`` turns it into the declared dtypes once it has been
    validated.

    Args:
        csv_path: Raw CSV file.
        columns: Declared columns to read; the default is all of them.

    Returns:
        One column per requested name, in the requested order.

    Raises:
        pyarrow.ArrowInvalid: If a value cannot be converted to its column's type, or a
            requested column is not in the file.
    """
    convert_options = pa_csv.ConvertOptions(
        column_types={name: _arrow_read_type(COLUMNS[name].dtype) for name in columns},
        include_columns=list(columns),
        null_values=[""],
        strings_can_be_null=True,
    )
    table = pa_csv.read_csv(csv_path, convert_options=convert_options)
    return table.to_pandas(types_mapper=_ARROW_TO_PANDAS.get)


def apply_declared_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every column its declared dtype, refusing any cast that would create nulls.

    Only categorical and string columns are cast. Numeric columns must already hold
    their declared dtype, because a narrowing numeric cast wraps out-of-range values
    without an error. Categorical columns are recoded with ``set_categories``, never with
    ``astype``: unordered categoricals with the same categories compare equal whatever
    their order, so ``astype`` to the declared dtype is a no-op and keeps the order of
    appearance (measured with pandas 2.3.3).

    Args:
        frame: Data with exactly the declared columns, in the declared order.

    Returns:
        A new frame with the declared dtypes.

    Raises:
        ValueError: If the columns are not the declared ones, a numeric column has a
            dtype other than the declared one, or a cast would create missing values.
    """
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Columns {tuple(frame.columns)} differ from the declared {EXPECTED_COLUMNS}"
        )
    columns: dict[str, pd.Series] = {}
    for spec in COLUMN_SPECS:
        column = frame[spec.name]
        if not isinstance(spec.dtype, pd.CategoricalDtype | pd.StringDtype):
            if column.dtype != spec.dtype:
                raise ValueError(
                    f"{spec.name!r} holds {column.dtype}, declared {spec.dtype}; numeric "
                    "columns are never cast here"
                )
            columns[spec.name] = column
            continue
        if isinstance(spec.dtype, pd.CategoricalDtype):
            cast = column.astype("category").cat.set_categories(
                spec.dtype.categories, ordered=spec.dtype.ordered
            )
        else:
            cast = column.astype(spec.dtype)
        created_nulls = int(cast.isna().sum()) - int(column.isna().sum())
        if created_nulls:
            raise ValueError(
                f"Casting {spec.name!r} to {spec.dtype} would turn {created_nulls} values "
                "into missing values"
            )
        columns[spec.name] = cast
    return pd.DataFrame(columns)


def load_csv(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the whole CSV with the declared dtypes.

    Call it only after the validator passed: this function raises on the first problem
    it meets, while the validator reports all of them.

    Args:
        csv_path: Raw CSV file.

    Returns:
        Both periods, every column with its declared dtype.
    """
    return apply_declared_dtypes(read_raw_csv(csv_path))


def split_periods(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into the reference and the production period by ``step``.

    Production is selected by its own comparison against ``PRODUCTION_FIRST_STEP``, not
    as the complement of reference. A row that falls in neither period is then dropped
    visibly, because the validator's partition check counts it, instead of landing on
    one side by default.

    Args:
        frame: Rows from both periods.

    Returns:
        The reference rows and the production rows, each with a fresh index.
    """
    step = frame[STEP_COLUMN]
    reference = frame.loc[step <= settings.REFERENCE_LAST_STEP].reset_index(drop=True)
    production = frame.loc[step >= settings.PRODUCTION_FIRST_STEP].reset_index(drop=True)
    return reference, production


def write_periods(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    reference_path: Path = REFERENCE_PATH,
    production_path: Path = PRODUCTION_PATH,
) -> None:
    """Write each period to its own parquet file, overwriting any previous version.

    Args:
        reference: Rows with ``step <= REFERENCE_LAST_STEP``.
        production: Rows with ``step >= PRODUCTION_FIRST_STEP``.
        reference_path: Destination of the reference period.
        production_path: Destination of the production period.
    """
    for frame, path in ((reference, reference_path), (production, production_path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, engine="pyarrow", index=False)


def _read_period(path: Path) -> pd.DataFrame:
    """Read a period file and give it the declared dtypes.

    The pandas metadata stored in the file is ignored: with it, pyarrow rebuilds
    StringDtype("pyarrow") as StringDtype("python") even when given a types_mapper.
    """
    table = pq.read_table(path)
    frame = table.to_pandas(types_mapper=_ARROW_TO_PANDAS.get, ignore_metadata=True)
    return apply_declared_dtypes(frame)


def read_reference(path: Path = REFERENCE_PATH) -> pd.DataFrame:
    """Read the reference period (``step <= REFERENCE_LAST_STEP``) with the declared dtypes.

    This is the period for training and for fitting anything: the model, the
    preprocessor, a threshold or a statistic.

    Args:
        path: Reference parquet file.

    Returns:
        The reference rows.

    Raises:
        ValueError: If the file holds a row past the reference period, which would mean
            production rows leaked into the file used for fitting.
    """
    frame = _read_period(path)
    last_step = int(frame[STEP_COLUMN].max()) if len(frame) else None
    if last_step is not None and last_step > settings.REFERENCE_LAST_STEP:
        raise ValueError(
            f"{path} holds step {last_step}, past the reference period "
            f"(step <= {settings.REFERENCE_LAST_STEP})"
        )
    return frame


def read_production(path: Path = PRODUCTION_PATH) -> pd.DataFrame:
    """Read the production period (``step >= PRODUCTION_FIRST_STEP``) with the declared dtypes.

    Reading this file before phase 4 (monitoring) violates the temporal cut: no
    production row may influence the fitting of anything, whether the model, the
    preprocessor, a threshold or a statistic (CLAUDE.md). Until then the only
    sanctioned access is ``count_parquet_rows``, which reads the file footer and no row.

    Args:
        path: Production parquet file.

    Returns:
        The production rows.
    """
    return _read_period(path)


def count_parquet_rows(path: Path) -> int:
    """Count the rows of a parquet file from its footer metadata, without reading any row.

    Args:
        path: Parquet file.

    Returns:
        The number of rows the file declares.
    """
    return int(pq.read_metadata(path).num_rows)
