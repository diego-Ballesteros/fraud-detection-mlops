"""The models of phase 2A, with default hyperparameters, as factories the harness calls.

Every factory returns a new, unfitted object, and every model pairs with its own
preprocessor factory: the ``full`` Pipeline for the models, and a passthrough for the
signature rule, which reads ``type``, ``amount`` and ``oldbalanceOrg`` as they come.

"Default hyperparameters" means each library's defaults. Only two settings are given, and
neither changes what is learned: the seed, from ``settings.RANDOM_STATE``, and ``n_jobs``,
to use every core where the library allows it. Logistic regression with the default lbfgs
solver has no ``n_jobs`` effect for a binary target; its parallelism comes from the
OpenMP and BLAS thread pools.

lightgbm is imported inside its factory, not at the top. On a Windows machine without the
Visual C++ Redistributable, ``import lightgbm`` alone fails to load ``lib_lightgbm.dll``:
its import table names ``MSVCP140.dll`` and ``VCOMP140.DLL``, and scikit-learn's wheel
ships and loads both under those exact names (numpy's copy is renamed with a hash, so
importing numpy does not help). This module imports scikit-learn at the top, so by the time
the factory runs both are loaded. Moving the import to the top would put it before
scikit-learn in import order, and a test that builds the model in a clean interpreter
would fail on such a machine (ADR 0003 records the condition; docs/DEPENDENCIES.md says
which lightgbm versions it affects).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from fraud_detection.config import settings
from fraud_detection.features.pipeline import build_preprocessor
from fraud_detection.models.baseline import SignatureRule
from fraud_detection.models.evaluation import EstimatorFactory

EVALUATED_CONFIGURATION: Final[str] = "full"
"""The feature configuration of every run in phase 2A; ``no_signature`` comes later."""

ALL_CORES: Final[int] = -1


def full_preprocessor() -> Pipeline:
    """The ``full`` preprocessing Pipeline, new and unfitted."""
    return build_preprocessor(EVALUATED_CONFIGURATION)


def raw_rows() -> FunctionTransformer:
    """A passthrough step: the signature rule reads the raw columns."""
    return FunctionTransformer()


def majority_class() -> DummyClassifier:
    """Predict the majority class for every row: score 0 for fraud everywhere.

    It is the floor and a control: with every score tied, average precision is exactly the
    prevalence of the validation fold, so any other value means the harness is wrong.
    """
    return DummyClassifier(strategy="most_frequent")


def signature_rule() -> SignatureRule:
    """The baseline of ADR 0007."""
    return SignatureRule()


def logistic_regression() -> LogisticRegression:
    """Logistic regression with scikit-learn's defaults."""
    return LogisticRegression(random_state=settings.RANDOM_STATE)


def random_forest() -> RandomForestClassifier:
    """Random forest with scikit-learn's defaults, including ``max_depth=None``."""
    return RandomForestClassifier(n_jobs=ALL_CORES, random_state=settings.RANDOM_STATE)


def lightgbm_classifier() -> Any:
    """LightGBM with its defaults; ``verbose=-1`` only silences its log.

    Returns:
        A new ``lightgbm.LGBMClassifier``.
    """
    # Deferred on purpose: see the module docstring.
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_jobs=ALL_CORES, random_state=settings.RANDOM_STATE, verbose=-1)


@dataclass(frozen=True)
class ModelSpec:
    """A model as the harness sees it.

    Attributes:
        name: Identifier, used as the MLflow run's model tag.
        preprocessor: Factory of the preprocessing step.
        model: Factory of the classifier.
        learns_from_labels: Whether ``fit`` uses the labels. The constant and the rule do
            not, so shuffling their labels cannot change them.
    """

    name: str
    preprocessor: EstimatorFactory
    model: EstimatorFactory
    learns_from_labels: bool


MODEL_SPECS: Final[Mapping[str, ModelSpec]] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            ModelSpec("majority_class", full_preprocessor, majority_class, False),
            ModelSpec("signature_rule", raw_rows, signature_rule, False),
            ModelSpec("logistic_regression", full_preprocessor, logistic_regression, True),
            ModelSpec("random_forest", full_preprocessor, random_forest, True),
            ModelSpec("lightgbm", full_preprocessor, lightgbm_classifier, True),
        )
    }
)
"""Every model of phase 2A, by name, in the order they are reported."""
