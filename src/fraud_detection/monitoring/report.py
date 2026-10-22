"""The Evidently report: the HTML deliverable of the monitoring, in ``reports/deliverables/``.

One ``Report``, with current = production and reference = the reference period:

- ``DataDriftPreset`` over the 11 model features, with PSI for every column.
- ``ClassificationPreset`` at the fixed threshold (``performance.OPERATING_THRESHOLD``), on
  the artifact's scores. Evidently flags a row when its probability is at least the threshold
  (read in evidently 0.7.23), the project's convention; it receives the probabilities and the
  threshold, never flags computed here, so a test can check that the two agree. The reference
  side of this part is in-sample, because the artifact was fitted on those rows: context,
  never a baseline.

**Column types are declared, not inferred.** The money features are numerical and the other
eight categorical. Evidently bins a numerical column with more than 20 distinct values with
equal-width bins over reference and current together; ``hour_sin`` and ``hour_cos`` hold 21
and 22 distinct float values (symmetric hours differ in the last bits), so left to inference
they would take that path. Declared categorical, Evidently compares their value shares
exactly, as ``drift`` does, and the two instruments must agree on those eight features. On the
three money features the bins differ by design (``drift`` takes them from the reference
only), and so do the values.

**Telemetry.** Evidently ships a telemetry client. In 0.7.23 only its UI, service and
collector import it, not report generation (read in the code, not measured). ``DO_NOT_TRACK``,
the switch that client reads, is set before Evidently is imported anyway.
"""

import os

os.environ.setdefault("DO_NOT_TRACK", "1")

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import ClassificationPreset, DataDriftPreset

from fraud_detection.data.schema import TARGET_COLUMN
from fraud_detection.monitoring.drift import PSI_MODERATE
from fraud_detection.monitoring.performance import OPERATING_THRESHOLD

SCORE_COLUMN: Final[str] = "score"
"""The artifact's probability of fraud, the column Evidently thresholds."""

_METRIC_TYPE_PREFIX: Final[str] = "evidently:metric_v2:"


def report_frame(features: pd.DataFrame, labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """The model features with the label and the score, as Evidently reads them.

    Args:
        features: The model features of some rows.
        labels: Their 0/1 labels.
        scores: The artifact's score of each row.

    Returns:
        A new frame: the features, then the label and the score.
    """
    frame = features.copy()
    frame[TARGET_COLUMN] = np.asarray(labels, dtype=np.int64)
    frame[SCORE_COLUMN] = np.asarray(scores, dtype=float)
    return frame


def build_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: Sequence[str],
    discrete: Sequence[str],
    threshold: float = OPERATING_THRESHOLD,
) -> Snapshot:
    """Run the report: drift of every feature by PSI, and classification at the threshold.

    Args:
        reference: ``report_frame`` of the reference period.
        production: ``report_frame`` of the production period.
        features: The feature columns whose drift is measured.
        discrete: Which of them are categorical; the others are numerical.
        threshold: Where Evidently cuts the score.

    Returns:
        The computed snapshot.
    """
    definition = DataDefinition(
        numerical_columns=[name for name in features if name not in discrete] + [SCORE_COLUMN],
        categorical_columns=[name for name in features if name in discrete] + [TARGET_COLUMN],
        classification=[
            BinaryClassification(target=TARGET_COLUMN, prediction_probas=SCORE_COLUMN, pos_label=1)
        ],
    )
    report = Report(
        [
            DataDriftPreset(columns=list(features), method="psi", threshold=PSI_MODERATE),
            ClassificationPreset(probas_threshold=threshold),
        ]
    )
    return report.run(
        Dataset.from_pandas(production, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )


def _metrics(snapshot: Snapshot, kind: str) -> list[dict[str, Any]]:
    wanted = _METRIC_TYPE_PREFIX + kind
    return [item for item in snapshot.dict()["metrics"] if item["config"]["type"] == wanted]


def drift_values(snapshot: Snapshot) -> pd.Series:
    """Evidently's PSI of each feature, current against reference.

    Args:
        snapshot: From ``build_report``.

    Returns:
        PSI by feature name.
    """
    items = _metrics(snapshot, "ValueDrift")
    values = {item["config"]["column"]: float(item["value"]) for item in items}
    return pd.Series(values, name="evidently_psi", dtype=float)


def classification_values(snapshot: Snapshot) -> dict[str, float]:
    """Evidently's precision and recall of the positive class on the current data.

    Args:
        snapshot: From ``build_report``.

    Returns:
        ``precision`` and ``recall``.

    Raises:
        ValueError: If the snapshot does not hold exactly one of each.
    """
    found = {}
    for name in ("Precision", "Recall"):
        items = _metrics(snapshot, name)
        if len(items) != 1:
            raise ValueError(f"Expected one {name} metric, found {len(items)}")
        found[name.lower()] = float(items[0]["value"])
    return found


def save_report(snapshot: Snapshot, path: Path) -> int:
    """Write the report as one HTML file.

    Args:
        snapshot: From ``build_report``.
        path: Destination; its folder is created if needed.

    Returns:
        The file size in bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return path.stat().st_size
