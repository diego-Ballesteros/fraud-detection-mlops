"""The chosen model: its configuration, the one serialized artifact, and its registration.

ADR 0010 chose trial 5 of the search (LightGBM) by the tie rule of docs/EVALUATION.md. Its
hyperparameters are written here at full precision, because the study file (``optuna.db``)
is not versioned and a clean clone must rebuild the same model.

The artifact is one ``Pipeline``, preprocessing and model together, built by
``evaluation.compose``, the same function the harness fits every fold with. Notebooks 03
and 04, the interpretability and the monitoring scripts load this one object; none of them
rebuilds a transformation.
It is serialized with the standard library's ``pickle``, the same format the registry copy
uses, and loading it needs this package installed, because the Pipeline refers to its
classes and functions by name.

Registration builds no MLflow client at import, and refuses to run unless the active
tracking URI is the one ``tracking.mlflow_setup.configure_mlflow`` set: a client built
before configuration resolves against the local store (the "Model Version not found" of
the previous project).

A registered version points at the run that logged it. ``log_version_artifacts`` attaches
files to that run, under their own folder, so the version carries its own explanation: the
interpretability figures of phase 3 travel with version 1. It refuses unless the version was
registered from the very artifact the files describe, by its sha256.
"""

import pickle
from collections.abc import Mapping, Sequence
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
ARTIFACT_VERSION: Final[str] = "1"
"""The registry version of ``ARTIFACT_PATH`` (ADR 0010)."""

ARTIFACT_SHA256_TAG: Final[str] = "artifact_sha256"
"""Version tag ``scripts/register_model.py`` writes: the sha256 of the registered artifact."""

INTERPRETABILITY_FOLDER: Final[str] = "interpretability"
"""Folder of the version's run where its interpretability figures and tables go."""
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


def log_version_artifacts(
    paths: Sequence[Path],
    *,
    target: TrackingTarget,
    version: str,
    artifact_sha256: str,
    folder: str = INTERPRETABILITY_FOLDER,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Attach files to the run that registered one version, under one folder of that run.

    Every check runs before the first upload, so a refusal leaves the run as it was. The
    run's parameters, metrics and logged model are not touched; only files under ``folder``
    are added, and a file with the same name there is replaced.

    Args:
        paths: Files to attach, at least one.
        target: What ``configure_mlflow`` returned; proves MLflow was configured.
        version: The registered version of ``REGISTERED_MODEL_NAME``.
        artifact_sha256: The sha256 of the artifact the files describe; it must equal the
            version's ``ARTIFACT_SHA256_TAG``.
        folder: Folder of the run that receives the files.
        tags: Tags to set on the version once the files are attached.

    Returns:
        The ID of the run the files went to.

    Raises:
        RuntimeError: If the active tracking URI is not ``target``'s.
        ValueError: If ``paths`` is empty, or the version was registered from another
            artifact or without the sha256 tag.
        FileNotFoundError: If a path is not a file.
    """
    active = redact_uri(mlflow.get_tracking_uri())
    if active != target.tracking_uri:
        raise RuntimeError(
            f"MLflow points at {active}, not at the configured {target.tracking_uri}: "
            "call configure_mlflow before attaching files"
        )
    # An empty list would attach nothing and still report success (ERR-008).
    if not paths:
        raise ValueError("No file to attach")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Not files: {missing}")
    client = mlflow.MlflowClient()
    model_version = client.get_model_version(REGISTERED_MODEL_NAME, version)
    registered = model_version.tags.get(ARTIFACT_SHA256_TAG)
    if registered != artifact_sha256:
        raise ValueError(
            f"Version {version} was registered from artifact {registered}, "
            f"not from {artifact_sha256}: the files would describe another model"
        )
    for path in paths:
        client.log_artifact(model_version.run_id, str(path), artifact_path=folder)
    for name, value in (tags or {}).items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, name, value)
    return str(model_version.run_id)
