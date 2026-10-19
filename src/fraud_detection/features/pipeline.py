"""The preprocessing Pipeline, built new for each named configuration.

``build_preprocessor`` is a factory, not a built object: a built object invites reuse
across folds, and a reused object is an object already fitted (ADR 0008).

Eligible against selected (ADR 0006). Eligible is a property of the data: a column can be
used without breaking the causal contract, and ``schema.MODEL_ELIGIBLE_COLUMNS`` derives
it from the schema. Selected is a property of the experiment: what one configuration uses.
It lives here, as a map from configuration to features rather than a constant, because
ADR 0007 requires an ablation. ``step`` is eligible and never selected: the Pipeline
consumes it to derive the hour and does not expose it.

The ColumnTransformer gives each branch a frame with only that branch's columns, so a
branch that reads another branch's column raises ``KeyError`` instead of working.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from fraud_detection.data.schema import TRANSACTION_TYPES
from fraud_detection.features.builder import (
    DERIVATION_INPUTS,
    HOUR_COS,
    HOUR_SIN,
    SIGNATURE_FEATURE,
    DerivedFeatures,
)

TYPE_COLUMN: Final[str] = "type"


@dataclass(frozen=True)
class FeatureConfiguration:
    """The features one configuration selects.

    Attributes:
        money: Money columns: ``log1p``, then robust scaling.
        signature: Whether ``amount_equals_origin_balance`` is built and selected.
    """

    money: tuple[str, ...]
    signature: bool

    @property
    def categorical(self) -> tuple[str, ...]:
        """Categorical columns, one-hot encoded."""
        return (TYPE_COLUMN,)

    @property
    def derived(self) -> tuple[str, ...]:
        """Derived features, passed through as the builder makes them."""
        signature = (SIGNATURE_FEATURE,) if self.signature else ()
        return (*signature, HOUR_SIN, HOUR_COS)

    @property
    def selected(self) -> tuple[str, ...]:
        """Every column one of the ColumnTransformer's branches receives."""
        return (*self.money, *self.categorical, *self.derived)

    @property
    def raw_inputs(self) -> tuple[str, ...]:
        """CSV columns the configuration reads, directly or through a derivation."""
        names = [*self.money, *self.categorical]
        for feature in self.derived:
            names.extend(DERIVATION_INPUTS[feature])
        return tuple(dict.fromkeys(names))


CONFIGURATIONS: Final[Mapping[str, FeatureConfiguration]] = MappingProxyType(
    {
        "full": FeatureConfiguration(
            money=("amount", "oldbalanceOrg", "oldbalanceDest"), signature=True
        ),
        # The ablation of ADR 0007: no oldbalanceOrg and no amount_equals_origin_balance.
        "no_signature": FeatureConfiguration(money=("amount", "oldbalanceDest"), signature=False),
    }
)
"""Selected features per configuration name."""


def log1p_non_negative(values: pd.DataFrame) -> pd.DataFrame:
    """Apply ``log1p`` to money columns, refusing negative values.

    ``log1p`` turns a value below -1 into NaN without an error, and a value between -1 and
    0 into a meaningless number. The schema declares these columns non-negative, and the
    split of work is this: the validator catches the malformed record, before any data is
    written; this guard is the second line, and catches someone using the Pipeline on data
    that never went through the validator.

    Args:
        values: Money columns.

    Returns:
        ``log1p`` of every value, with the same columns and index.

    Raises:
        ValueError: If any value is negative.
    """
    if (np.asarray(values) < 0).any():
        raise ValueError("Negative values in the money columns; log1p is defined for >= 0")
    return np.log1p(values)


def _money_branch() -> Pipeline:
    return Pipeline(
        [
            ("log1p", FunctionTransformer(log1p_non_negative, feature_names_out="one-to-one")),
            ("scale", RobustScaler()),
        ]
    )


def build_preprocessor(configuration: str) -> Pipeline:
    """Build a new, unfitted preprocessing Pipeline for a named configuration.

    Two steps: ``derive`` keeps the model-eligible columns and adds the derived features;
    ``columns`` sends each selected column to exactly one branch (money, categorical,
    derived) and drops the rest. Every call returns new objects.

    Args:
        configuration: A key of ``CONFIGURATIONS``.

    Returns:
        The unfitted Pipeline, with pandas output.

    Raises:
        KeyError: If the configuration does not exist.
    """
    selection = CONFIGURATIONS[configuration]
    columns = ColumnTransformer(
        transformers=[
            ("money", _money_branch(), list(selection.money)),
            (
                "categorical",
                OneHotEncoder(
                    categories=[list(TRANSACTION_TYPES)],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                list(selection.categorical),
            ),
            ("derived", "passthrough", list(selection.derived)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [("derive", DerivedFeatures(signature=selection.signature)), ("columns", columns)]
    )
    return pipeline.set_output(transform="pandas")
