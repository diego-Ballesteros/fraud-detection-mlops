"""Tests of fraud_detection.models.tracking: run logging, data identity and git tags."""

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fraud_detection.config import Settings
from fraud_detection.models.estimators import MODEL_SPECS
from fraud_detection.models.evaluation import FoldResult
from fraud_detection.models.fit_guard import GuardedRun
from fraud_detection.models.tracking import (
    file_sha256,
    git_tags,
    log_guarded_run,
    reference_identity,
    run_tags,
)
from fraud_detection.tracking.mlflow_setup import configure_mlflow
from tests.test_mlflow_setup import MLFLOW_STATE_ENV


def fold(number: int, pr_auc: float) -> FoldResult:
    return FoldResult(
        fold=number,
        train_rows=100 * number,
        train_positives=number,
        validation_rows=50,
        validation_positives=2,
        train_steps=(1, 10 * number),
        validation_steps=(10 * number + 1, 10 * number + 10),
        train_signature_rows=number,
        train_signature_positives=number,
        label_shuffled=False,
        fit_seconds=float(number),
        predict_seconds=0.5,
        fit_warnings=(),
        metrics={"pr_auc": pr_auc, "recall_at_p0.99": 0.5},
    )


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    for name in MLFLOW_STATE_ENV:
        monkeypatch.delenv(name, raising=False)
    uri = f"sqlite:///{(tmp_path / 'store.db').as_posix()}"
    config = Settings(
        _env_file=None,
        mlflow_tracking_uri=uri,
        mlflow_tracking_username="someone",
        mlflow_tracking_password="not-a-real-secret",
    )
    configure_mlflow("tracking-test", config=config)
    yield uri
    mlflow.set_tracking_uri(None)


def test_a_run_has_its_folds_aggregates_and_tags(store: str) -> None:
    run = GuardedRun(
        folds=(fold(1, 0.2), fold(2, 0.4), fold(3, 0.9)),
        aborted_fold=None,
        aborted_after_seconds=None,
        error=None,
        wall_seconds=12.0,
    )
    run_id = log_guarded_run(
        run,
        run_name="probe",
        tags={"configuration": "full", "splitter": "temporal_expanding"},
        params={"C": 1.0},
    )
    client = MlflowClient(tracking_uri=store)
    logged = client.get_run(run_id)
    history = client.get_metric_history(run_id, "pr_auc")
    assert sorted((point.step, point.value) for point in history) == [(1, 0.2), (2, 0.4), (3, 0.9)]
    assert logged.data.metrics["pr_auc_mean"] == pytest.approx(0.5)
    # Sample standard deviation of 0.2, 0.4, 0.9: sqrt(0.13) = 0.3606.
    assert logged.data.metrics["pr_auc_std"] == pytest.approx(0.3605551, rel=1e-6)
    assert logged.data.metrics["train_rows"] == 300  # the last step logged
    assert logged.data.tags["configuration"] == "full"
    assert logged.data.tags["complete"] == "true"
    assert logged.data.tags["folds_finished"] == "3"
    assert logged.data.params["C"] == "1.0"


def test_an_aborted_run_says_so(store: str) -> None:
    run = GuardedRun(
        folds=(fold(1, 0.2),),
        aborted_fold=2,
        aborted_after_seconds=1200.4,
        error=None,
        wall_seconds=1300.0,
    )
    run_id = log_guarded_run(run, run_name="aborted", tags={}, params={})
    tags = MlflowClient(tracking_uri=store).get_run(run_id).data.tags
    assert tags["complete"] == "false"
    assert tags["aborted_fold"] == "2"
    assert tags["aborted_after_seconds"] == "1200.4"
    assert "pr_auc_std" not in MlflowClient(tracking_uri=store).get_run(run_id).data.metrics


def test_reference_identity_reads_the_footer_and_hashes_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "reference.parquet"
    pd.DataFrame({"step": [3, 1, 400, 7], "amount": [1.0, 2.0, 3.0, 4.0]}).to_parquet(path)
    identity = reference_identity(path)
    assert (identity.rows, identity.first_step, identity.last_step) == (4, 1, 400)
    assert identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity.tags()["data.reference_steps"] == "1-400"
    assert file_sha256(path) == identity.sha256


def test_a_file_without_statistics_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bare.parquet"
    pd.DataFrame({"step": [1, 2]}).to_parquet(path, write_statistics=False)
    with pytest.raises(ValueError, match="no statistics"):
        reference_identity(path)


def test_the_git_tags_name_the_head_and_say_it_is_not_this_code() -> None:
    tags = git_tags()
    assert re.fullmatch(r"[0-9a-f]{40}", tags["git_head_at_run"])
    assert tags["git_worktree_dirty"] in {"true", "false"}
    assert "does not contain it" in tags["git_head_note"]


def test_run_tags_say_what_was_evaluated() -> None:
    tags = run_tags(
        MODEL_SPECS["lightgbm"],
        "temporal_expanding",
        shuffle_training_labels=True,
        shared={"k": "v"},
    )
    assert tags == {
        "k": "v",
        "model": "lightgbm",
        "configuration": "full",
        "preprocessor": "full_preprocessor",
        "splitter": "temporal_expanding",
        "deciding_splitter": "true",
        "label_shuffle": "train_only",
    }
    rule = run_tags(
        MODEL_SPECS["signature_rule"], "stratified_kfold", shuffle_training_labels=False, shared={}
    )
    assert (rule["preprocessor"], rule["deciding_splitter"], rule["label_shuffle"]) == (
        "raw_rows",
        "false",
        "none",
    )


def test_a_nested_run_hangs_from_the_active_run(store: str) -> None:
    run = GuardedRun(
        folds=(fold(1, 0.2),),
        aborted_fold=None,
        aborted_after_seconds=None,
        error=None,
        wall_seconds=1.0,
    )
    with mlflow.start_run(run_name="parent") as parent:
        child_id = log_guarded_run(run, run_name="child", tags={}, params={}, nested=True)
    tags = MlflowClient(tracking_uri=store).get_run(child_id).data.tags
    assert tags["mlflow.parentRunId"] == parent.info.run_id
