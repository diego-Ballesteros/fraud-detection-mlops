"""Log a cross-validation to MLflow as one run: tags, parameters, per-fold metrics, aggregates.

Call ``tracking.mlflow_setup.configure_mlflow`` first; this module never decides where runs
go. No model is logged as an artifact and nothing is registered in the Model Registry.

Every run carries the identity of the data it measured (the reference rows, their step
range and the sha256 of the parquet file) and the git HEAD at the moment it ran. The
Executor does not commit, so that HEAD is the commit the work started from, not the
commit that will contain the code that produced the run; the run says so in a tag.
"""

import hashlib
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mlflow
import pyarrow.parquet as pq

from fraud_detection.config import settings
from fraud_detection.data.loader import REFERENCE_PATH, STEP_COLUMN
from fraud_detection.models.estimators import EVALUATED_CONFIGURATION, ModelSpec
from fraud_detection.models.evaluation import (
    DECIDING_SPLITTER,
    SPLITTERS,
    FoldResult,
    aggregate,
    load_evaluation_frame,
)
from fraud_detection.models.fit_guard import FIT_TIME_LIMIT_SECONDS, GuardedRun, run_guarded

GIT_HEAD_NOTE: Final[str] = (
    "HEAD when the run started. The code that produced this run was not committed at "
    "that moment (see git_worktree_dirty); a human commits it later, so this SHA does not "
    "contain it."
)
_HASH_CHUNK_BYTES: Final[int] = 1 << 24


@dataclass(frozen=True)
class DataIdentity:
    """What identifies the reference data a run measured.

    Attributes:
        rows: Rows in the reference parquet file, from its footer.
        first_step: Lowest step, from the file's column statistics.
        last_step: Highest step, from the file's column statistics.
        sha256: Hash of the file's bytes.
    """

    rows: int
    first_step: int
    last_step: int
    sha256: str

    def tags(self) -> dict[str, str]:
        """The identity as MLflow tags."""
        return {
            "data.reference_rows": str(self.rows),
            "data.reference_steps": f"{self.first_step}-{self.last_step}",
            "data.reference_sha256": self.sha256,
        }


def file_sha256(path: Path) -> str:
    """Hash a file's bytes in chunks.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal sha256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def reference_identity(path: Path = REFERENCE_PATH) -> DataIdentity:
    """Identify the reference file from its footer and its bytes, without reading any row.

    Args:
        path: Reference parquet file.

    Returns:
        Row count, step range and sha256.

    Raises:
        ValueError: If a row group lacks statistics for ``step``.
    """
    metadata = pq.read_metadata(path)
    column = metadata.schema.names.index(STEP_COLUMN)
    lows, highs = [], []
    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(column).statistics
        if statistics is None or not statistics.has_min_max:
            raise ValueError(f"Row group {group} of {path} has no statistics for {STEP_COLUMN}")
        lows.append(int(statistics.min))
        highs.append(int(statistics.max))
    return DataIdentity(
        rows=int(metadata.num_rows),
        first_step=min(lows),
        last_step=max(highs),
        sha256=file_sha256(path),
    )


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=settings.ROOT_DIR, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def git_tags() -> dict[str, str]:
    """The git HEAD at run time, whether the work tree had uncommitted changes, and the note.

    Returns:
        Tags ``git_head_at_run``, ``git_worktree_dirty`` and ``git_head_note``.
    """
    return {
        "git_head_at_run": _git("rev-parse", "HEAD"),
        "git_worktree_dirty": str(bool(_git("status", "--porcelain"))).lower(),
        "git_head_note": GIT_HEAD_NOTE,
    }


def fold_metrics(fold: FoldResult) -> dict[str, float]:
    """Every number of a fold that goes to MLflow, scores and sizes alike.

    Args:
        fold: One fold's result.

    Returns:
        Metric name to value.
    """
    sizes = {
        "train_rows": fold.train_rows,
        "train_positives": fold.train_positives,
        "validation_rows": fold.validation_rows,
        "validation_positives": fold.validation_positives,
        "train_signature_rows": fold.train_signature_rows,
        "train_signature_positives": fold.train_signature_positives,
        "fit_seconds": fold.fit_seconds,
        "predict_seconds": fold.predict_seconds,
    }
    return {**fold.metrics, **{name: float(value) for name, value in sizes.items()}}


def log_guarded_run(
    run: GuardedRun,
    *,
    run_name: str,
    tags: Mapping[str, str],
    params: Mapping[str, Any],
) -> str:
    """Log a guarded cross-validation as one MLflow run.

    Per-fold values are logged with ``step`` equal to the fold number; each metric present
    in every finished fold is also logged as ``<name>_mean`` and ``<name>_std`` (sample
    standard deviation across folds). An aborted or failed run is logged too, with the
    folds that finished and tags that say what happened.

    Args:
        run: The guarded run.
        run_name: Display name of the MLflow run.
        tags: Tags to set: configuration, splitter, model, data identity, git.
        params: Parameters to log, such as the model's hyperparameters.

    Returns:
        The MLflow run ID.
    """
    status = {
        "complete": str(run.complete).lower(),
        "folds_finished": str(len(run.folds)),
    }
    if run.aborted_fold is not None:
        status["aborted_fold"] = str(run.aborted_fold)
        status["aborted_after_seconds"] = f"{run.aborted_after_seconds:.1f}"
    if run.error is not None:
        status["error"] = run.error[-4000:]
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({**tags, **status})
        mlflow.log_params({name: str(value)[:500] for name, value in params.items()})
        for fold in run.folds:
            mlflow.log_metrics(fold_metrics(fold), step=fold.fold)
        summary: dict[str, float] = {"wall_seconds": run.wall_seconds}
        for name, value in aggregate(run.folds).items():
            summary[f"{name}_mean"] = value.mean
            if value.folds > 1:
                summary[f"{name}_std"] = value.std
        mlflow.log_metrics(summary)
        return str(active.info.run_id)


def base_tags(identity: DataIdentity) -> dict[str, str]:
    """Tags every run of a script shares: data identity, git, and the turn.

    Args:
        identity: The reference data the runs measure.

    Returns:
        Tag name to value.
    """
    return {**identity.tags(), **git_tags(), "turn": "2A"}


def run_tags(
    spec: ModelSpec, splitter: str, *, shuffle_training_labels: bool, shared: Mapping[str, str]
) -> dict[str, str]:
    """The tags of one run: what was evaluated and how, on top of the shared ones.

    Args:
        spec: The model.
        splitter: Name of the splitter, a key of ``SPLITTERS``.
        shuffle_training_labels: Whether the training labels were permuted.
        shared: Tags from ``base_tags``.

    Returns:
        Tag name to value.
    """
    return {
        **shared,
        "model": spec.name,
        "configuration": EVALUATED_CONFIGURATION,
        "preprocessor": spec.preprocessor.__name__,
        "splitter": splitter,
        "deciding_splitter": str(splitter == DECIDING_SPLITTER).lower(),
        "label_shuffle": "train_only" if shuffle_training_labels else "none",
    }


def evaluate_and_log(
    spec: ModelSpec,
    splitter: str,
    *,
    shared_tags: Mapping[str, str],
    shuffle_training_labels: bool = False,
    fit_time_limit: float = FIT_TIME_LIMIT_SECONDS,
) -> tuple[GuardedRun, str]:
    """Cross-validate one model on the reference period, guarded, and log it as a run.

    Args:
        spec: The model.
        splitter: Name of the splitter, a key of ``SPLITTERS``.
        shared_tags: Tags from ``base_tags``.
        shuffle_training_labels: Permute the training labels of every fold.
        fit_time_limit: Seconds a single fit may run before it is aborted.

    Returns:
        The guarded run and its MLflow run ID.
    """
    run = run_guarded(
        load_evaluation_frame,
        SPLITTERS[splitter](),
        spec,
        shuffle_training_labels=shuffle_training_labels,
        fit_time_limit=fit_time_limit,
    )
    params = {
        **spec.model().get_params(),
        "fit_time_limit_seconds": fit_time_limit,
        "cpu_count": os.cpu_count(),
    }
    suffix = "__shuffled" if shuffle_training_labels else ""
    run_id = log_guarded_run(
        run,
        run_name=f"{spec.name}__{splitter}{suffix}",
        tags=run_tags(
            spec, splitter, shuffle_training_labels=shuffle_training_labels, shared=shared_tags
        ),
        params=params,
    )
    return run, run_id
