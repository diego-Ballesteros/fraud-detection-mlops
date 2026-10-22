"""Tests of fraud_detection.models.registry: the chosen configuration, artifact and registry."""

import pickle
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraud_detection.config import Settings, settings
from fraud_detection.models import registry
from fraud_detection.models.registry import (
    ARTIFACT_PATH,
    ARTIFACT_SHA256_TAG,
    CHOSEN_PARAMS,
    INTERPRETABILITY_FOLDER,
    MODEL_INPUT_COLUMNS,
    REGISTERED_MODEL_NAME,
    build_chosen_pipeline,
    load_pipeline,
    log_version_artifacts,
    register_pipeline,
    save_pipeline,
)
from fraud_detection.models.search_space import FLOATS, INTS
from fraud_detection.tracking.mlflow_setup import TrackingTarget, configure_mlflow
from tests.test_evaluation import synthetic_frame
from tests.test_mlflow_setup import MLFLOW_STATE_ENV


def _rows(**overrides: object) -> pd.DataFrame:
    row = {"step": 30, "type": "PAYMENT", "amount": 120.0, "oldbalanceOrg": 900.0}
    row |= {"oldbalanceDest": 0.0, **overrides}
    return pd.DataFrame([row])[list(MODEL_INPUT_COLUMNS)]


@pytest.fixture
def fitted() -> Pipeline:
    frame = synthetic_frame()
    return build_chosen_pipeline().fit(frame.drop(columns=["isFraud"]), frame["isFraud"])


def test_the_chosen_params_lie_inside_the_search_space() -> None:
    for name, value in CHOSEN_PARAMS.items():
        low, high, _ = INTS.get(f"lgbm_{name}") or FLOATS[f"lgbm_{name}"]
        assert low <= value <= high, name


def test_the_artifact_has_the_harness_shape_and_the_chosen_params() -> None:
    pipeline = build_chosen_pipeline()
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    assert list(pipeline.named_steps["preprocess"].named_steps) == ["derive", "columns"]
    params = pipeline.named_steps["model"].get_params()
    assert {name: params[name] for name in CHOSEN_PARAMS} == dict(CHOSEN_PARAMS)
    assert (params["random_state"], params["n_jobs"]) == (settings.RANDOM_STATE, -1)


def test_a_saved_pipeline_loads_and_predicts_the_same(fitted: Pipeline, tmp_path: Path) -> None:
    path = tmp_path / "pipeline.pkl"
    size = save_pipeline(fitted, path)
    assert size == path.stat().st_size > 0
    rows = synthetic_frame().drop(columns=["isFraud"])
    assert np.array_equal(load_pipeline(path).predict_proba(rows), fitted.predict_proba(rows))


def test_a_file_that_is_not_a_pipeline_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps({"not": "a pipeline"}))
    with pytest.raises(TypeError, match="not a Pipeline"):
        load_pipeline(path)


@pytest.mark.skipif(
    not ARTIFACT_PATH.is_file(),
    reason="The artifact is written by scripts/register_model.py and is not in this checkout.",
)
def test_the_serialized_artifact_predicts_on_synthetic_rows() -> None:
    pipeline = load_pipeline(ARTIFACT_PATH)
    signature = _rows(type="TRANSFER", amount=5000.0, oldbalanceOrg=5000.0)
    legitimate = _rows()
    probabilities = pipeline.predict_proba(pd.concat([signature, legitimate]))
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    # A row that takes the whole balance is the simulator's fraud signature.
    assert probabilities[0, 1] > probabilities[1, 1]


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TrackingTarget]:
    for name in MLFLOW_STATE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # the sqlite store writes artifacts under ./mlruns
    config = Settings(
        _env_file=None,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'registry.db').as_posix()}",
        mlflow_tracking_username="someone",
        mlflow_tracking_password="not-a-real-secret",
    )
    yield configure_mlflow("registry-test", config=config)
    mlflow.set_tracking_uri(None)


def test_registration_refuses_a_target_mlflow_is_not_pointing_at(
    fitted: Pipeline, store: TrackingTarget
) -> None:
    elsewhere = TrackingTarget("sqlite:///elsewhere.db", "e", "1", local=False)
    with pytest.raises(RuntimeError, match="configure_mlflow"):
        register_pipeline(fitted, _rows(), target=elsewhere, tags={}, params={})
    assert mlflow.search_runs(output_format="list") == []


def test_a_registered_version_loads_back_and_predicts_the_same(
    fitted: Pipeline, store: TrackingTarget
) -> None:
    example = synthetic_frame()[list(MODEL_INPUT_COLUMNS)].head(5).astype({"type": object})
    run_id, version = register_pipeline(
        fitted, example, target=store, tags={"chosen_trial": "5"}, params={"n": 1}
    )
    loaded = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    rows = synthetic_frame()[list(MODEL_INPUT_COLUMNS)]
    assert np.array_equal(loaded.predict_proba(rows), fitted.predict_proba(rows))
    model_version = mlflow.MlflowClient().get_model_version(REGISTERED_MODEL_NAME, version)
    assert model_version.tags["chosen_trial"] == "5"
    assert model_version.run_id == run_id
    assert registry.CHOSEN_TRIAL == 5


def _registered(fitted: Pipeline, store: TrackingTarget, sha256: str) -> tuple[str, str]:
    example = synthetic_frame()[list(MODEL_INPUT_COLUMNS)].head(5).astype({"type": object})
    return register_pipeline(
        fitted, example, target=store, tags={ARTIFACT_SHA256_TAG: sha256}, params={}
    )


def _files(tmp_path: Path) -> list[Path]:
    folder = tmp_path / "figures"
    folder.mkdir()
    paths = [folder / "waterfall_a.png", folder / "orderings.csv"]
    for number, path in enumerate(paths):
        path.write_bytes(f"content {number}".encode())
    return paths


def test_files_attach_to_the_run_of_the_version(
    fitted: Pipeline, store: TrackingTarget, tmp_path: Path
) -> None:
    run_id, version = _registered(fitted, store, "abc123")
    paths = _files(tmp_path)
    attached_to = log_version_artifacts(
        paths,
        target=store,
        version=version,
        artifact_sha256="abc123",
        tags={"interpretability": INTERPRETABILITY_FOLDER},
    )
    assert attached_to == run_id
    client = mlflow.MlflowClient()
    listed = sorted(item.path for item in client.list_artifacts(run_id, INTERPRETABILITY_FOLDER))
    assert listed == sorted(f"{INTERPRETABILITY_FOLDER}/{path.name}" for path in paths)
    for path in paths:
        local = client.download_artifacts(
            run_id, f"{INTERPRETABILITY_FOLDER}/{path.name}", str(tmp_path / "back")
        )
        assert Path(local).read_bytes() == path.read_bytes()
    tags = client.get_model_version(REGISTERED_MODEL_NAME, version).tags
    assert tags["interpretability"] == INTERPRETABILITY_FOLDER
    # The logged model is still where it was, and still loads.
    loaded = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    rows = synthetic_frame()[list(MODEL_INPUT_COLUMNS)]
    assert np.array_equal(loaded.predict_proba(rows), fitted.predict_proba(rows))


def test_files_are_refused_for_a_version_of_another_artifact(
    fitted: Pipeline, store: TrackingTarget, tmp_path: Path
) -> None:
    run_id, version = _registered(fitted, store, "abc123")
    with pytest.raises(ValueError, match="another model"):
        log_version_artifacts(
            _files(tmp_path), target=store, version=version, artifact_sha256="def456"
        )
    assert mlflow.MlflowClient().list_artifacts(run_id, INTERPRETABILITY_FOLDER) == []


def test_attaching_refuses_before_uploading_anything(
    fitted: Pipeline, store: TrackingTarget, tmp_path: Path
) -> None:
    run_id, version = _registered(fitted, store, "abc123")
    paths = _files(tmp_path)
    elsewhere = TrackingTarget("sqlite:///elsewhere.db", "e", "1", local=False)
    with pytest.raises(RuntimeError, match="configure_mlflow"):
        log_version_artifacts(paths, target=elsewhere, version=version, artifact_sha256="abc123")
    with pytest.raises(ValueError, match="No file"):
        log_version_artifacts([], target=store, version=version, artifact_sha256="abc123")
    # One missing file stops the call before the existing one is uploaded.
    with pytest.raises(FileNotFoundError):
        log_version_artifacts(
            [paths[0], tmp_path / "missing.png"],
            target=store,
            version=version,
            artifact_sha256="abc123",
        )
    assert mlflow.MlflowClient().list_artifacts(run_id, INTERPRETABILITY_FOLDER) == []
