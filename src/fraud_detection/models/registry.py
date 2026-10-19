"""The chosen model: its configuration, the one serialized artifact, and its registration.

ADR 0010 chose trial 5 of the search (LightGBM) by the tie rule of docs/EVALUATION.md. Its
hyperparameters are written here at full precision, because the study file (``optuna.db``)
is not versioned and a clean clone must rebuild the same model.

The artifact is one ``Pipeline``, preprocessing and model together, built by
``evaluation.compose``, the same function the harness fits every fold with. Notebook 03,
the monitoring and the app load this one object; none of them rebuilds a transformation.
It is serialized with the standard library's ``pickle``, the same format the registry copy
uses, and loading it needs this package installed, because the Pipeline refers to its
classes and functions by name.

Registration builds no MLflow client at import, and refuses to run unless the active
tracking URI is the one ``tracking.mlflow_setup.configure_mlflow`` set: a client built
before configuration resolves against the local store (the "Model Version not found" of
the previous project).
"""

import pickle
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from fraud_detection.config import settings
from fraud_detection.models.estimators import full_preprocessor, lightgbm_classifier
from fraud_detection.models.evaluation import compose
from fraud_detection.tracking.mlflow_setup import TrackingTarget, redact_uri

CHOSEN_STUDY: Final[str] = "search_full_temporal"
CHOSEN_TRIAL: Final[int] = 5

CHOSEN_PARAMS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "n_estimators": 98,
        "learning_rate": 0.027010527749605478,
        "num_leaves": 56,
        "max_depth": 7,
        "min_child_samples": 249,
        "reg_lambda": 0.009962513222055111,
        "scale_pos_weight": 34.890188454913854,
    }
)
"""Trial 5's LightGBM hyperparameters, as the study stored them (ADR 0010)."""

ARTIFACT_PATH: Final[Path] = settings.ROOT_DIR / "artifacts" / "fraud_detection_pipeline.pkl"
REGISTERED_MODEL_NAME: Final[str] = "fraud-detection-pipeline"
MODEL_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
)
"""The raw columns the artifact reads; every other column of a row is ignored."""


def chosen_model(random_state: int = settings.RANDOM_STATE) -> Any:
    """The chosen LightGBM, new and unfitted.

    Args:
        random_state: Seed; only the robustness check changes it.

    Returns:
        A ``lightgbm.LGBMClassifier`` with ``CHOSEN_PARAMS``.
    """
    return lightgbm_classifier(**CHOSEN_PARAMS, random_state=random_state)


def build_chosen_pipeline() -> Pipeline:
    """The artifact's Pipeline, unfitted: the ``full`` preprocessor and the chosen model.

    Returns:
        The Pipeline, built by the same function the harness uses.
    """
    return compose(full_preprocessor, chosen_model)


def save_pipeline(pipeline: Pipeline, path: Path = ARTIFACT_PATH) -> int:
    """Serialize the fitted Pipeline.

    Args:
        pipeline: The fitted Pipeline.
        path: Destination file; its folder is created if needed.

    Returns:
        The file size in bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(pipeline, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path.stat().st_size


def load_pipeline(path: Path = ARTIFACT_PATH) -> Pipeline:
    """Load the serialized Pipeline. Only load files this project wrote: pickle runs code.

    Args:
        path: The artifact file.

    Returns:
        The fitted Pipeline.

    Raises:
        TypeError: If the file does not hold a scikit-learn Pipeline.
    """
    with path.open("rb") as stream:
        pipeline = pickle.load(stream)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"{path} holds {type(pipeline).__name__}, not a Pipeline")
    return pipeline


def register_pipeline(
    pipeline: Pipeline,
    input_example: pd.DataFrame,
    *,
    target: TrackingTarget,
    tags: Mapping[str, str],
    params: Mapping[str, Any],
) -> tuple[str, str]:
    """Log the fitted Pipeline as an MLflow model and register a new version of it.

    Args:
        pipeline: The fitted Pipeline.
        input_example: A few raw rows with ``MODEL_INPUT_COLUMNS``; the signature is
            inferred from them and from the Pipeline's probabilities.
        target: What ``configure_mlflow`` returned; proves MLflow was configured.
        tags: Tags for the run and the model version.
        params: Parameters to log on the run.

    Returns:
        The MLflow run ID and the registered version.

    Raises:
        RuntimeError: If the active tracking URI is not ``target``'s.
    """
    active = redact_uri(mlflow.get_tracking_uri())
    if active != target.tracking_uri:
        raise RuntimeError(
            f"MLflow points at {active}, not at the configured {target.tracking_uri}: "
            "call configure_mlflow before registering"
        )
    signature = infer_signature(input_example, pipeline.predict_proba(input_example))
    with mlflow.start_run(run_name=f"register__{REGISTERED_MODEL_NAME}") as run:
        mlflow.set_tags(dict(tags))
        mlflow.log_params({name: str(value)[:500] for name, value in params.items()})
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="pickle",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            pyfunc_predict_fn="predict_proba",
        )
        version = str(info.registered_model_version)
        client = mlflow.MlflowClient()
        for name, value in tags.items():
            client.set_model_version_tag(REGISTERED_MODEL_NAME, version, name, value)
        return str(run.info.run_id), version
