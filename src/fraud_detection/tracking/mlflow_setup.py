"""MLflow configuration: tracking URI, credentials and experiment, set only when asked.

Importing this module builds no client, changes no global state and opens no connection.
In the previous project an ``MlflowClient()`` built before configuration resolved against
the local store and raised "Model Version not found" against a registry that was perfectly
reachable. Here nothing MLflow-related happens until ``configure_mlflow`` is called, and a
test imports the module in a clean interpreter to prove it.

MLflow reads its credentials from ``os.environ``. pydantic-settings reads ``.env`` into the
``settings`` instance and does **not** export it to ``os.environ`` (finding (g) of phase 0).
``configure_mlflow`` therefore exports the credentials from ``settings`` before its first
MLflow call, instead of assuming someone already put them there.

The local store is anchored to the project root. MLflow's own default is ``mlflow.db`` in
the current working directory, so the same script would log to a different store depending
on where it is launched from (the reason ADR 0001 resolves paths from the module).
"""

import os
import traceback
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import mlflow

from fraud_detection.config import Settings, settings

TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"
USERNAME_ENV: Final[str] = "MLFLOW_TRACKING_USERNAME"
PASSWORD_ENV: Final[str] = "MLFLOW_TRACKING_PASSWORD"

EXPERIMENT_NAME: Final[str] = "fraud-detection"
"""One experiment for the project; runs carry the model, splitter and configuration as tags."""

LOCAL_TRACKING_URI: Final[str] = f"sqlite:///{(settings.ROOT_DIR / 'mlflow.db').as_posix()}"
"""The local store, at the project root and gitignored, whatever the working directory."""


class TrackingNotConfiguredError(RuntimeError):
    """The remote tracking server was requested but ``settings`` does not describe it.

    Attributes:
        missing: Names of the environment variables that are not set.
    """

    def __init__(self, missing: tuple[str, ...]) -> None:
        """Record which variables are missing.

        Args:
            missing: Names of the environment variables that are not set.
        """
        super().__init__(f"Not set in .env or the environment: {', '.join(missing)}")
        self.missing = missing


@dataclass(frozen=True)
class TrackingTarget:
    """Where runs are logged after ``configure_mlflow``. Holds no credential.

    Attributes:
        tracking_uri: The tracking URI, without any user information it might embed.
        experiment_name: Name of the active experiment.
        experiment_id: Identifier the tracking server gave the experiment.
        local: Whether this is the local store rather than the server in ``settings``.
    """

    tracking_uri: str
    experiment_name: str
    experiment_id: str
    local: bool


def redact_uri(uri: str) -> str:
    """Drop any ``user:password@`` part from a URI, so the result is safe to print.

    Args:
        uri: A tracking URI.

    Returns:
        The same URI without user information.
    """
    parts = urlsplit(uri)
    if parts.username is None and parts.password is None:
        return uri
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port is not None else host
    return urlunsplit(parts._replace(netloc=netloc))


def configure_mlflow(
    experiment_name: str = EXPERIMENT_NAME,
    *,
    local: bool = False,
    config: Settings = settings,
) -> TrackingTarget:
    """Point MLflow at a tracking store and an experiment, exporting credentials first.

    With ``local=False`` the URI and both credentials come from ``config`` and all three
    are written to ``os.environ`` before any MLflow call. With ``local=True`` the store is
    ``LOCAL_TRACKING_URI`` and no credential is needed. Either way the URI is also exported,
    so a child process sees the same store. The function is idempotent: a second call with
    the same arguments leaves the same environment, URI and experiment.

    The first network call to a remote server is the experiment lookup, so a server that
    cannot be reached, or that rejects the credentials, raises here and not at the first
    logged metric.

    Args:
        experiment_name: Experiment to activate; created if it does not exist.
        local: Use the local store instead of the server in ``config``.
        config: Source of the URI and the credentials.

    Returns:
        The store and experiment now active, without credentials.

    Raises:
        TrackingNotConfiguredError: If ``local`` is False and ``config`` lacks the URI, the
            username or the password.
    """
    if local:
        uri = LOCAL_TRACKING_URI
    else:
        # Settings ignores empty values (env_ignore_empty), so an empty entry arrives as None.
        remote_uri = config.mlflow_tracking_uri
        username = config.mlflow_tracking_username
        password = config.mlflow_tracking_password
        if remote_uri is None or username is None or password is None:
            given = (
                (TRACKING_URI_ENV, remote_uri),
                (USERNAME_ENV, username),
                (PASSWORD_ENV, password),
            )
            raise TrackingNotConfiguredError(tuple(name for name, value in given if value is None))
        uri = remote_uri
        os.environ[USERNAME_ENV] = username
        os.environ[PASSWORD_ENV] = password.get_secret_value()
    os.environ[TRACKING_URI_ENV] = uri
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(experiment_name)
    return TrackingTarget(
        tracking_uri=redact_uri(uri),
        experiment_name=experiment.name,
        experiment_id=str(experiment.experiment_id),
        local=local,
    )


def configure_with_local_fallback(
    experiment_name: str = EXPERIMENT_NAME, *, config: Settings = settings
) -> tuple[TrackingTarget, str | None]:
    """Try the server in ``config``; if that fails for any reason, use the local store.

    The fallback is never silent: the caller gets the full traceback of the failure and
    ``TrackingTarget.local`` says where the runs went. The harness itself needs no network.

    Args:
        experiment_name: Experiment to activate.
        config: Source of the URI and the credentials.

    Returns:
        The active target, and the traceback of the remote failure or None.
    """
    try:
        return configure_mlflow(experiment_name, config=config), None
    except Exception:
        failure = traceback.format_exc()
    return configure_mlflow(experiment_name, local=True), failure
