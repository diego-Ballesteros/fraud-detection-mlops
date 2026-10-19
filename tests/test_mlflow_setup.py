"""Tests of fraud_detection.tracking.mlflow_setup.

The import test runs in a clean interpreter with the network and MLflow's client and store
constructors replaced by recorders. It first runs on synthetic modules that do build a
client, resolve a store, open a connection or change MLflow's state at import: if the
probe cannot see those, a clean result on the real module proves nothing.
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pytest
from pydantic import SecretStr

from fraud_detection.config import Settings
from fraud_detection.tracking import mlflow_setup
from fraud_detection.tracking.mlflow_setup import (
    PASSWORD_ENV,
    TRACKING_URI_ENV,
    USERNAME_ENV,
    TrackingNotConfiguredError,
    configure_mlflow,
    configure_with_local_fallback,
    redact_uri,
)

MLFLOW_STATE_ENV = (TRACKING_URI_ENV, USERNAME_ENV, PASSWORD_ENV, "MLFLOW_EXPERIMENT_ID")

PROBE = r"""
import importlib, json, os, socket, sys

events = []

def deny(name):
    def blocked(*args, **kwargs):
        events.append(name)
        raise OSError(f"network blocked by the test: {name}")
    return blocked

socket.socket.connect = deny("socket.connect")
socket.socket.connect_ex = deny("socket.connect_ex")
socket.create_connection = deny("socket.create_connection")
socket.getaddrinfo = deny("socket.getaddrinfo")

import mlflow
import mlflow.tracking._tracking_service.utils as store_utils
from mlflow.tracking.client import MlflowClient

original_init = MlflowClient.__init__
def spy_init(self, *args, **kwargs):
    events.append("MlflowClient")
    original_init(self, *args, **kwargs)
MlflowClient.__init__ = spy_init

original_get_store = store_utils._get_store
def spy_get_store(*args, **kwargs):
    events.append("tracking store")
    return original_get_store(*args, **kwargs)
store_utils._get_store = spy_get_store

def mlflow_env():
    return {k: v for k, v in os.environ.items() if k.startswith("MLFLOW_")}

env_before, uri_set_before = mlflow_env(), store_utils.is_tracking_uri_set()
try:
    importlib.import_module(sys.argv[1])
    error = None
except Exception as exc:
    error = type(exc).__name__
print(json.dumps({
    "events": events,
    "env_changed": mlflow_env() != env_before,
    "uri_set": store_utils.is_tracking_uri_set() != uri_set_before,
    "error": error,
}))
"""

BAD_MODULES = {
    "builds_client": "import mlflow\nclient = mlflow.MlflowClient()\n",
    "resolves_store": "import mlflow\nmlflow.search_experiments()\n",
    "opens_connection": "import urllib.request\nurllib.request.urlopen('http://example.invalid')\n",
    "sets_uri": "import mlflow\nmlflow.set_tracking_uri('sqlite:///probe.db')\n",
    "exports_env": "import os\nos.environ['MLFLOW_TRACKING_USERNAME'] = 'x'\n",
}


def probe_import(module: str, tmp_path: Path) -> dict[str, Any]:
    """Import ``module`` in a clean interpreter and report what it touched."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("MLFLOW_")}
    env["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, module],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=180,
        check=True,
    )
    result: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    return result


def test_the_import_probe_sees_every_violation(tmp_path: Path) -> None:
    for name, source in BAD_MODULES.items():
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    seen = {name: probe_import(name, tmp_path) for name in BAD_MODULES}
    assert "MlflowClient" in seen["builds_client"]["events"]
    assert "tracking store" in seen["resolves_store"]["events"]
    # http.client goes through socket.create_connection, which fails before getaddrinfo.
    assert any(event.startswith("socket.") for event in seen["opens_connection"]["events"])
    assert seen["sets_uri"]["uri_set"] is True
    assert seen["exports_env"]["env_changed"] is True


@pytest.mark.parametrize(
    "module", ["fraud_detection.tracking.mlflow_setup", "fraud_detection.models.tracking"]
)
def test_importing_builds_no_client_and_touches_nothing(module: str, tmp_path: Path) -> None:
    assert probe_import(module, tmp_path) == {
        "events": [],
        "env_changed": False,
        "uri_set": False,
        "error": None,
    }


@pytest.fixture
def clean_mlflow_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # monkeypatch restores each variable, including deleting one the code under test sets.
    for name in MLFLOW_STATE_ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    mlflow.set_tracking_uri(None)


def remote_settings(tmp_path: Path) -> Settings:
    """Settings that name a store and credentials; the store is a local file."""
    return Settings(
        _env_file=None,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'store.db').as_posix()}",
        mlflow_tracking_username="someone",
        mlflow_tracking_password=SecretStr("not-a-real-secret"),
    )


def _state() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in MLFLOW_STATE_ENV} | {
        "uri": mlflow.get_tracking_uri()
    }


@pytest.mark.usefixtures("clean_mlflow_state")
def test_configuration_is_idempotent(tmp_path: Path) -> None:
    config = remote_settings(tmp_path)
    first = configure_mlflow("probe", config=config)
    state = _state()
    second = configure_mlflow("probe", config=config)
    assert second == first
    assert _state() == state
    assert state[USERNAME_ENV] == "someone"
    assert state[PASSWORD_ENV] == "not-a-real-secret"
    assert [e.name for e in mlflow.search_experiments() if e.name == "probe"] == ["probe"]


@pytest.mark.usefixtures("clean_mlflow_state")
def test_credentials_are_exported_before_the_first_mlflow_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str | None, str | None]] = []

    def record(name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            seen.append((name, os.environ.get(USERNAME_ENV), os.environ.get(PASSWORD_ENV)))
            return SimpleNamespace(name=args[0] if args else "", experiment_id="1")

        return call

    monkeypatch.setattr(mlflow, "set_tracking_uri", record("set_tracking_uri"))
    monkeypatch.setattr(mlflow, "set_experiment", record("set_experiment"))
    configure_mlflow("probe", config=remote_settings(tmp_path))
    assert seen == [
        ("set_tracking_uri", "someone", "not-a-real-secret"),
        ("set_experiment", "someone", "not-a-real-secret"),
    ]


@pytest.mark.usefixtures("clean_mlflow_state")
def test_a_missing_variable_is_named(tmp_path: Path) -> None:
    config = Settings(_env_file=None, mlflow_tracking_username="someone")
    with pytest.raises(TrackingNotConfiguredError) as raised:
        configure_mlflow("probe", config=config)
    assert raised.value.missing == (TRACKING_URI_ENV, PASSWORD_ENV)
    # Nothing was exported for a configuration that was refused.
    assert os.environ.get(USERNAME_ENV) is None


def test_the_target_never_holds_a_credential(tmp_path: Path) -> None:
    fields = set(vars(mlflow_setup.TrackingTarget("u", "e", "1", False)))
    assert fields == {"tracking_uri", "experiment_name", "experiment_id", "local"}


def test_user_information_is_redacted_from_the_uri() -> None:
    assert (
        redact_uri("https://user:token@dagshub.com/o/r.mlflow") == "https://dagshub.com/o/r.mlflow"
    )
    assert redact_uri("https://user@host:8443/x") == "https://host:8443/x"
    assert redact_uri("https://dagshub.com/o/r.mlflow") == "https://dagshub.com/o/r.mlflow"
    assert redact_uri("sqlite:///C:/x/mlflow.db") == "sqlite:///C:/x/mlflow.db"


def test_the_local_store_is_anchored_to_the_project_root() -> None:
    expected = "sqlite:///" + (mlflow_setup.settings.ROOT_DIR / "mlflow.db").as_posix()
    assert expected == mlflow_setup.LOCAL_TRACKING_URI


@pytest.mark.usefixtures("clean_mlflow_state")
def test_a_remote_failure_falls_back_to_the_local_store_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = f"sqlite:///{(tmp_path / 'local.db').as_posix()}"
    monkeypatch.setattr(mlflow_setup, "LOCAL_TRACKING_URI", local)
    target, failure = configure_with_local_fallback("probe", config=Settings(_env_file=None))
    assert target.local is True
    assert target.tracking_uri == local
    assert failure is not None
    assert "TrackingNotConfiguredError" in failure
    # Control: a configured server is used and reports no failure.
    target, failure = configure_with_local_fallback("probe", config=remote_settings(tmp_path))
    assert target.local is False
    assert failure is None
