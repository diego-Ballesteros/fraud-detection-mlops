"""Tests of fraud_detection.monitoring.report: Evidently agrees with the project's instruments.

Two instruments that measure the same thing on the same bins must give the same number. On
the discrete features Evidently and ``drift`` bin by value, so their PSI must coincide; at
the fixed threshold, Evidently's precision and recall must be the project's, including rows
whose score equals the threshold exactly.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from evidently.core.report import Snapshot

from fraud_detection.monitoring.drift import reference_profiles
from fraud_detection.monitoring.performance import confusion
from fraud_detection.monitoring.report import (
    build_report,
    classification_values,
    drift_values,
    report_frame,
    save_report,
)

FEATURES = ["money", "flag", "hour"]
DISCRETE = ["flag", "hour"]
THRESHOLD = 0.4


def _features(rng: np.random.Generator, size: int, shift: float, flag_share: float) -> pd.DataFrame:
    hour = rng.integers(0, 24, size)
    return pd.DataFrame(
        {
            "money": rng.normal(shift, 1, size),
            "flag": (rng.random(size) < flag_share).astype(float),
            # 24 hours, 21 distinct float values: the shape of hour_sin.
            "hour": np.round(np.sin(2 * np.pi * hour / 24), 1),
        }
    )


def _scored(rng: np.random.Generator, features: pd.DataFrame) -> pd.DataFrame:
    labels = (rng.random(len(features)) < 0.1).astype(int)
    # Scores on a coarse grid, so many rows sit exactly on the threshold.
    scores = np.clip(np.round(0.3 * labels + rng.random(len(features)) * 0.6, 1), 0, 1)
    return report_frame(features, labels, scores)


@pytest.fixture(scope="module")
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    reference = _scored(rng, _features(rng, 6_000, 0.0, 0.05))
    production = _scored(rng, _features(rng, 4_000, 0.7, 0.15))
    return reference, production


@pytest.fixture(scope="module")
def snapshot(frames: tuple[pd.DataFrame, pd.DataFrame]) -> Snapshot:
    reference, production = frames
    return build_report(reference, production, FEATURES, DISCRETE, threshold=THRESHOLD)


def test_telemetry_is_switched_off_before_evidently_loads() -> None:
    assert "DO_NOT_TRACK" in os.environ


def test_report_frame_adds_the_label_and_the_score() -> None:
    features = pd.DataFrame({"a": [1.0, 2.0]})
    frame = report_frame(features, np.array([0, 1]), np.array([0.2, 0.9]))
    assert frame.columns.tolist() == ["a", "isFraud", "score"]
    assert features.columns.tolist() == ["a"]  # the input is not modified


def test_evidently_psi_equals_ours_on_the_discrete_features(
    frames: tuple[pd.DataFrame, pd.DataFrame], snapshot: Snapshot
) -> None:
    reference, production = frames
    theirs = drift_values(snapshot)
    assert sorted(theirs.index) == sorted(FEATURES)
    profiles = reference_profiles(reference[FEATURES], DISCRETE)
    for name in DISCRETE:
        ours = profiles[name].psi(production[name].to_numpy())
        assert theirs[name] == pytest.approx(ours, rel=1e-9, abs=1e-12), name
    # Control: the comparison can fail. The flag moved from 5% to 15% and has to show it.
    assert theirs["flag"] > 0.1


def test_evidently_precision_and_recall_equal_ours_at_the_threshold(
    frames: tuple[pd.DataFrame, pd.DataFrame], snapshot: Snapshot
) -> None:
    _, production = frames
    scores = production["score"].to_numpy()
    assert (scores == THRESHOLD).sum() > 0, "no row on the threshold: the test checks nothing"
    ours = confusion(production["isFraud"].to_numpy(), scores >= THRESHOLD)
    theirs = classification_values(snapshot)
    assert theirs["precision"] == pytest.approx(ours.precision, abs=1e-12)
    assert theirs["recall"] == pytest.approx(ours.recall, abs=1e-12)
    # Control: with ">" instead of ">=" the numbers would differ.
    strict = confusion(production["isFraud"].to_numpy(), scores > THRESHOLD)
    assert (strict.precision, strict.recall) != (ours.precision, ours.recall)


def test_the_report_is_written_as_html(snapshot: Snapshot, tmp_path: Path) -> None:
    path = tmp_path / "deliverables" / "report.html"
    size = save_report(snapshot, path)
    assert size == path.stat().st_size > 0
    assert path.read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype html")
