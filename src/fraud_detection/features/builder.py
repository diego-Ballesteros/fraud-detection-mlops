"""Derived features, built from the model-eligible columns and nothing else.

Each derivation receives a frame with only its own input columns, so reading any other
column raises ``KeyError`` instead of quietly working: a derivation cannot reach a
POST_TRANSACTION column, nor another derivation's input.

``day_of_week`` is not built: the weekly-structure measurement discarded it (ADR 0006).
Its edges, a 7-day position that is not a calendar day and a first day of 23 hours, are in
docs/DATA_DICTIONARY.md, section 2.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraud_detection.data.schema import EXPECTED_COLUMNS, MODEL_ELIGIBLE_COLUMNS

SIGNATURE_FEATURE: Final[str] = "amount_equals_origin_balance"
HOUR_SIN: Final[str] = "hour_sin"
HOUR_COS: Final[str] = "hour_cos"
HOURS_PER_DAY: Final[int] = 24

ELIGIBLE_COLUMNS: Final[tuple[str, ...]] = tuple(
    name for name in EXPECTED_COLUMNS if name in MODEL_ELIGIBLE_COLUMNS
)
"""The model-eligible columns, in schema order."""

DERIVATION_INPUTS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {SIGNATURE_FEATURE: ("amount", "oldbalanceOrg"), HOUR_SIN: ("step",), HOUR_COS: ("step",)}
)
"""The eligible columns each derived feature is computed from."""


def amount_equals_origin_balance(frame: pd.DataFrame) -> pd.Series:
    """Whether the transaction takes the whole origin balance.

    ``(amount > 0) & (amount == oldbalanceOrg)``. This is the only definition of the
    simulator's signature: the feature, the baseline rule and the measurements in
    ``profiling`` all call it. A movement of zero over a balance of zero takes nothing, so
    it is excluded (ADR 0006).

    Args:
        frame: Rows with at least ``amount`` and ``oldbalanceOrg``.

    Returns:
        One boolean per row, named ``amount_equals_origin_balance``.
    """
    amount = frame["amount"]
    signature = (amount > 0) & (amount == frame["oldbalanceOrg"])
    return signature.rename(SIGNATURE_FEATURE)


def cyclic_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode the hour of the day, ``step % 24``, as a point on the unit circle.

    Hour 23 and hour 0 end up next to each other, which a raw ordinal hour does not
    allow (ADR 0008).

    Args:
        frame: Rows with at least ``step``.

    Returns:
        Two columns, ``hour_sin`` and ``hour_cos``, on the index of ``frame``.
    """
    angle = 2 * np.pi * (frame["step"] % HOURS_PER_DAY) / HOURS_PER_DAY
    return pd.DataFrame({HOUR_SIN: np.sin(angle), HOUR_COS: np.cos(angle)}, index=frame.index)


class DerivedFeatures(TransformerMixin, BaseEstimator):
    """Keep only the model-eligible columns and add the derived features.

    The transformer is stateless: ``fit`` learns nothing, so it cannot carry information
    from one dataset to another.
    """

    def __init__(self, signature: bool = True) -> None:
        """Configure which derived features to add.

        Args:
            signature: Whether to add ``amount_equals_origin_balance``. The
                ``no_signature`` configuration turns it off (ADR 0007).
        """
        self.signature = signature

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DerivedFeatures":  # noqa: N803
        """Learn nothing; the transformer is stateless.

        Args:
            X: Rows, ignored.
            y: Labels, ignored.

        Returns:
            The transformer itself.
        """
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report the transformer as fitted: being stateless, it always is."""
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Select the eligible columns and append the derived ones.

        Args:
            X: Rows holding at least every model-eligible column.

        Returns:
            The eligible columns, then the derived features.

        Raises:
            KeyError: If an eligible column is missing.
        """
        eligible = X[list(ELIGIBLE_COLUMNS)]
        parts = [eligible]
        if self.signature:
            inputs = list(DERIVATION_INPUTS[SIGNATURE_FEATURE])
            parts.append(amount_equals_origin_balance(eligible[inputs]).to_frame())
        parts.append(cyclic_hour(eligible[list(DERIVATION_INPUTS[HOUR_SIN])]))
        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Name the output columns, in order.

        Args:
            input_features: Ignored; the output depends only on ``signature``.

        Returns:
            The eligible columns, then the derived features.
        """
        derived = [SIGNATURE_FEATURE] if self.signature else []
        return np.asarray([*ELIGIBLE_COLUMNS, *derived, HOUR_SIN, HOUR_COS], dtype=object)
