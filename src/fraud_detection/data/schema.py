"""Declarative data contract for the PaySim transactions CSV.

This module only declares: one ``ColumnSpec`` per column, in the order of the CSV header,
and the views derived from those declarations. Nothing here reads or transforms data.

The central field is ``availability``: when a column exists relative to the transaction.
CLAUDE.md makes it a hard rule that a column whose prior existence nobody can explain is
leakage, so a column that cannot be placed before the transaction from its name and the
dataset's public documentation is declared POST_TRANSACTION by default.

Every exposed object is immutable: specs are frozen dataclasses, the column mapping is a
``MappingProxyType``, and the derived collections are tuples and frozensets. Rebinding a
module attribute (``schema.TARGET_COLUMN = ...``) is not prevented by Python; that gap is
the same one ADR 0001 records for class attributes in ``config.py``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype


class Availability(Enum):
    """When a column exists relative to the transaction it describes."""

    PRE_TRANSACTION = "pre_transaction"
    """Existed before the transaction; information available when the decision is made."""

    POST_TRANSACTION = "post_transaction"
    """Only exists after the transaction has executed."""

    SYSTEM_OUTPUT = "system_output"
    """Output of another decision system, not an input."""

    TARGET = "target"
    """The label."""

    IDENTIFIER = "identifier"
    """Identifies an entity; not a measurement of the phenomenon."""


@dataclass(frozen=True)
class ColumnSpec:
    """Contract for one column of the CSV.

    Attributes:
        name: Column name exactly as it appears in the CSV header.
        dtype: Dtype the column holds in memory and in the processed parquet files.
        availability: When the column exists relative to the transaction.
        justification: One line on why ``availability`` holds.
        allowed_values: Closed domain of the column, or None when it has none.
        nullable: Whether the contract admits missing values.
    """

    name: str
    dtype: np.dtype[Any] | ExtensionDtype
    availability: Availability
    justification: str
    allowed_values: tuple[str, ...] | tuple[int, ...] | None = None
    nullable: bool = False


# The public dataset documentation spells two of these "CASH-IN" and "CASH-OUT"; the CSV
# itself uses an underscore, and the contract follows the CSV.
TRANSACTION_TYPES: Final[tuple[str, ...]] = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
BINARY_VALUES: Final[tuple[int, ...]] = (0, 1)

_IDENTIFIER_DTYPE: Final = pd.StringDtype("pyarrow")

COLUMN_SPECS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec(
        name="step",
        dtype=np.dtype("int16"),
        availability=Availability.PRE_TRANSACTION,
        justification=(
            "Simulation hour (1 step = 1 hour) at which the transaction is requested; "
            "the clock is known at decision time."
        ),
    ),
    ColumnSpec(
        name="type",
        dtype=pd.CategoricalDtype(categories=TRANSACTION_TYPES, ordered=False),
        availability=Availability.PRE_TRANSACTION,
        justification="Kind of operation the customer requests; part of the request itself.",
        allowed_values=TRANSACTION_TYPES,
    ),
    ColumnSpec(
        name="amount",
        dtype=np.dtype("float64"),
        availability=Availability.PRE_TRANSACTION,
        justification="Amount the customer asks to move; part of the request itself.",
    ),
    ColumnSpec(
        name="nameOrig",
        dtype=_IDENTIFIER_DTYPE,
        availability=Availability.IDENTIFIER,
        justification="Identifier of the customer who starts the transaction.",
    ),
    ColumnSpec(
        name="oldbalanceOrg",
        dtype=np.dtype("float64"),
        availability=Availability.PRE_TRANSACTION,
        justification=(
            "Documented as the origin account's balance before the transaction; it is the "
            "requesting customer's own balance."
        ),
    ),
    ColumnSpec(
        name="newbalanceOrig",
        dtype=np.dtype("float64"),
        availability=Availability.POST_TRANSACTION,
        justification=(
            "Documented as the origin account's balance after the transaction; it only "
            "exists once the transaction has executed."
        ),
    ),
    ColumnSpec(
        name="nameDest",
        dtype=_IDENTIFIER_DTYPE,
        availability=Availability.IDENTIFIER,
        justification=(
            "Identifier of the recipient; per the documentation, a leading 'M' marks a merchant."
        ),
    ),
    ColumnSpec(
        name="oldbalanceDest",
        dtype=np.dtype("float64"),
        availability=Availability.PRE_TRANSACTION,
        justification=(
            "Documented as the recipient's balance before the transaction; the documentation "
            "adds that merchants ('M') carry no balance information."
        ),
    ),
    ColumnSpec(
        name="newbalanceDest",
        dtype=np.dtype("float64"),
        availability=Availability.POST_TRANSACTION,
        justification=(
            "Documented as the recipient's balance after the transaction; it only exists "
            "once the transaction has executed."
        ),
    ),
    ColumnSpec(
        name="isFraud",
        dtype=np.dtype("int8"),
        availability=Availability.TARGET,
        justification="Label: whether a fraudulent agent of the simulation made the transaction.",
        allowed_values=BINARY_VALUES,
    ),
    ColumnSpec(
        name="isFlaggedFraud",
        dtype=np.dtype("int8"),
        availability=Availability.SYSTEM_OUTPUT,
        justification=(
            "Documented as the output of the business rule that flags attempts to transfer "
            "more than 200,000 in a single transaction."
        ),
        allowed_values=BINARY_VALUES,
    ),
)

# --- Views derived from the declarations; none of them is written by hand. -------------

COLUMNS: Final[Mapping[str, ColumnSpec]] = MappingProxyType(
    {spec.name: spec for spec in COLUMN_SPECS}
)
"""Spec of every column, by name."""

EXPECTED_COLUMNS: Final[tuple[str, ...]] = tuple(spec.name for spec in COLUMN_SPECS)
"""Every column the CSV must have, in header order."""

MODEL_ELIGIBLE_COLUMNS: Final[frozenset[str]] = frozenset(
    spec.name for spec in COLUMN_SPECS if spec.availability is Availability.PRE_TRANSACTION
)
"""Columns that may enter the model: exactly the PRE_TRANSACTION ones."""

# Unpacking into one name fails at import unless exactly one column is the target.
(_target_name,) = (spec.name for spec in COLUMN_SPECS if spec.availability is Availability.TARGET)
TARGET_COLUMN: Final[str] = _target_name
"""Name of the label column."""
