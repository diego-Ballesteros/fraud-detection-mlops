"""The simulator's signature as a scikit-learn classifier: the project's baseline (ADR 0007).

It is a rule, not a model. ``fit`` records nothing from the data, so the same rule is
evaluated whatever it is fitted on, which is what makes it a baseline and not a
competitor. Being an estimator lets phase 2 evaluate it with the same harness as the
models, so the comparison is between objects of the same kind.
"""

from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from fraud_detection.data.schema import BINARY_VALUES
from fraud_detection.features.builder import (
    DERIVATION_INPUTS,
    SIGNATURE_FEATURE,
    amount_equals_origin_balance,
)

BASELINE_TYPES: Final[tuple[str, ...]] = ("TRANSFER", "CASH_OUT")
"""Transaction types the rule applies to.

Kept although the equality never occurs in the other types of the reference period: that
is a fact of the reference period, and dropping a condition because it is redundant in
training would rest on training describing serving, which is what the temporal cut exists
to prevent (ADR 0007).
"""


class SignatureRule(ClassifierMixin, BaseEstimator):
    """Flag ``type in (TRANSFER, CASH_OUT)`` and ``amount_equals_origin_balance``.

    ``classes_`` is a class attribute fixed by the schema's label domain, not something
    ``fit`` learns.
    """

    classes_: Final = np.asarray(BINARY_VALUES)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SignatureRule":  # noqa: N803
        """Record nothing: the rule does not depend on the data it is fitted on.

        Args:
            X: Rows, ignored.
            y: Labels, ignored.

        Returns:
            The rule itself, unchanged.
        """
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report the rule as fitted: it has nothing to learn."""
        return True

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Flag the rows the rule covers.

        Args:
            X: Rows with at least ``type``, ``amount`` and ``oldbalanceOrg``.

        Returns:
            1 for a flagged row and 0 otherwise.
        """
        signature = amount_equals_origin_balance(X[list(DERIVATION_INPUTS[SIGNATURE_FEATURE])])
        flagged = X["type"].isin(BASELINE_TYPES) & signature
        return np.asarray(flagged, dtype=np.int8)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Give each row a probability of 1 or 0: the rule has no intermediate scores.

        Args:
            X: Rows with at least ``type``, ``amount`` and ``oldbalanceOrg``.

        Returns:
            One row per input row, with the probabilities of classes 0 and 1.
        """
        positive = self.predict(X).astype(float)
        return np.column_stack([1.0 - positive, positive])
