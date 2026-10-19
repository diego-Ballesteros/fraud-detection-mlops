"""The evaluation harness: two splitters, the decision metrics, and one loop over folds.

Declared before any result was seen (ADR 0009):

- **Decision metric:** recall at precision >= 0.99. The global PR-AUC is saturated by the
  region the simulator's signature covers, so it does not separate models; what separates
  them are the frauds the signature rule leaves out. **Secondary:** recall at precision >=
  0.95, in case no model reaches the 0.99 floor. **Context, always reported:** PR-AUC,
  ROC-AUC and Brier score.
- **Practical significance:** a recall difference below 0.005 makes two models the same
  model.
- **Splitters:** ``ExpandingStepSplit`` decides. ``StratifiedKFold`` is measured beside it
  to quantify the optimism of ignoring time; it is not an alternative.

The harness receives **factories** of the preprocessor and the model, never built objects:
a built object invites reuse across folds, and a reused object is an object already
fitted. Each fold calls both factories and fits a fresh Pipeline.

The features handed to the Pipeline never contain the label column, so no preprocessor can
read it. With ``shuffle_training_labels`` the labels of the training part of each fold are
permuted before fitting, and the validation labels stay intact. The fold runs through the
same code either way, so the shuffled-target test validates this harness and not another.
"""

import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from fraud_detection.config import settings
from fraud_detection.data.loader import STEP_COLUMN, read_reference
from fraud_detection.data.schema import TARGET_COLUMN
from fraud_detection.features.builder import (
    DERIVATION_INPUTS,
    ELIGIBLE_COLUMNS,
    SIGNATURE_FEATURE,
    amount_equals_origin_balance,
)
from fraud_detection.models.baseline import SignatureRule

DECISION_PRECISION: Final[float] = 0.99
"""Precision floor of the decision metric, recall at precision >= 0.99."""

SECONDARY_PRECISION: Final[float] = 0.95
"""Precision floor of the secondary metric, in case no model reaches the decision floor."""

PRECISION_FLOORS: Final[tuple[float, ...]] = (DECISION_PRECISION, SECONDARY_PRECISION)

PRACTICAL_SIGNIFICANCE: Final[float] = 0.005
"""Recall difference below which two models are the same model: about 22 of the 4,477
reference frauds."""

TEMPORAL_FOLDS: Final[int] = 5
VALIDATION_WINDOW_STEPS: Final[int] = 40
FIRST_VALIDATION_STEP: Final[int] = (
    settings.REFERENCE_LAST_STEP - TEMPORAL_FOLDS * VALIDATION_WINDOW_STEPS + 1
)
"""201: derived so that the last validation window ends on the last reference step."""

STRATIFIED_FOLDS: Final[int] = 5

TEMPORAL: Final[str] = "temporal_expanding"
STRATIFIED: Final[str] = "stratified_kfold"
DECIDING_SPLITTER: Final[str] = TEMPORAL

EVALUATION_COLUMNS: Final[tuple[str, ...]] = (*ELIGIBLE_COLUMNS, TARGET_COLUMN)
"""What the harness reads: the model-eligible columns and the label."""

# A floor given as a float is read as the simplest fraction within this denominator, so
# 0.99 is exactly 99/100 and the comparison against tp / flagged is done in integers.
_MAX_FLOOR_DENOMINATOR: Final[int] = 10**6

PERMUTATIONS: Final[int] = 20
"""Draws per permutation test and per random-classifier null (ADR 0009, status note)."""

FIT_STARTED: Final[str] = "fit_started"
FIT_FINISHED: Final[str] = "fit_finished"
FOLD_DONE: Final[str] = "fold_done"


# --- Splitters --------------------------------------------------------------------------


class Splitter(Protocol):
    """Anything that yields (train positions, validation positions) like scikit-learn."""

    def split(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: np.ndarray | None = None,
        groups: None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield the positional indices of each fold."""
        ...


@dataclass(frozen=True)
class StepWindow:
    """A closed range of steps, ``first <= step <= last``."""

    first: int
    last: int


@dataclass(frozen=True)
class ExpandingStepSplit:
    """Expanding-window folds cut on whole steps, never inside one.

    ``TimeSeriesSplit`` cuts by row position, so a fold border can fall inside a step: the
    model would train on part of an hour and validate on the rest. Here every border is a
    step border. Fold k validates on the k-th window of ``window`` steps and trains on
    every row before that window, which is the production situation: train on all the
    history, predict the next period. With the defaults::

        fold 1: train 1-200, validate 201-240
        ...
        fold 5: train 1-360, validate 361-400

    Rows past the reference period are refused, not skipped.

    Attributes:
        first_validation_step: First step of the first validation window.
        window: Steps per validation window.
        n_splits: Number of folds.
    """

    first_validation_step: int = FIRST_VALIDATION_STEP
    window: int = VALIDATION_WINDOW_STEPS
    n_splits: int = TEMPORAL_FOLDS

    def __post_init__(self) -> None:
        """Refuse windows with no history before them or past the reference period."""
        if self.window < 1 or self.n_splits < 1:
            raise ValueError("window and n_splits must be positive")
        if self.first_validation_step < 2:
            raise ValueError("The first validation window needs at least one step before it")
        last = self.validation_windows[-1].last
        if last > settings.REFERENCE_LAST_STEP:
            raise ValueError(
                f"Validation reaches step {last}, past the reference period "
                f"(step <= {settings.REFERENCE_LAST_STEP})"
            )

    @property
    def validation_windows(self) -> tuple[StepWindow, ...]:
        """The validation window of each fold, in order."""
        starts = (self.first_validation_step + k * self.window for k in range(self.n_splits))
        return tuple(StepWindow(start, start + self.window - 1) for start in starts)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # noqa: N803
        """Return the number of folds.

        Args:
            X: Ignored.
            y: Ignored.
            groups: Ignored.

        Returns:
            ``n_splits``.
        """
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: np.ndarray | None = None,
        groups: None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield the training and validation positions of each fold.

        Args:
            X: Rows with a ``step`` column, in any order.
            y: Ignored.
            groups: Ignored.

        Yields:
            Positions of the training rows (every step before the window) and of the
            validation rows (the window), both sorted.

        Raises:
            ValueError: If a row is past the reference period, or a fold would be empty.
        """
        steps = X[STEP_COLUMN].to_numpy()
        if len(steps) and int(steps.max()) > settings.REFERENCE_LAST_STEP:
            raise ValueError(
                f"Rows past step {settings.REFERENCE_LAST_STEP}: the production period "
                "never enters a split"
            )
        for window in self.validation_windows:
            train = np.flatnonzero(steps < window.first)
            validation = np.flatnonzero((steps >= window.first) & (steps <= window.last))
            if not len(train) or not len(validation):
                raise ValueError(f"Empty fold for validation window {window}")
            yield train, validation


def stratified_kfold() -> StratifiedKFold:
    """The contrast splitter: 5 stratified folds, shuffled with ``settings.RANDOM_STATE``.

    It ignores time on purpose. It is measured to quantify the optimism of doing so, and
    never decides.

    Returns:
        A new ``StratifiedKFold``.
    """
    return StratifiedKFold(
        n_splits=STRATIFIED_FOLDS, shuffle=True, random_state=settings.RANDOM_STATE
    )


SPLITTERS: Final[Mapping[str, Callable[[], Splitter]]] = MappingProxyType(
    {TEMPORAL: ExpandingStepSplit, STRATIFIED: stratified_kfold}
)
"""Splitter factories by name. ``DECIDING_SPLITTER`` is the one that decides."""


# --- Metrics ----------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    """The rows one score threshold flags.

    Attributes:
        threshold: A row is flagged when its score is at least this value.
        flagged: Rows flagged.
        true_positives: Flagged rows that are positive.
        positives: Positive rows in the whole sample.
    """

    threshold: float
    flagged: int
    true_positives: int
    positives: int

    @property
    def precision(self) -> float:
        """Share of flagged rows that are positive."""
        return self.true_positives / self.flagged

    @property
    def recall(self) -> float:
        """Share of positive rows that are flagged."""
        return self.true_positives / self.positives


def _as_fraction(value: float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(value).limit_denominator(_MAX_FLOOR_DENOMINATOR)


def _threshold_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every distinct score as a threshold, highest first, with its flagged and TP counts.

    Rows with equal scores are always flagged together: no threshold separates them.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(f"labels {y.shape} and scores {s.shape} must be equal-length vectors")
    if np.isnan(s).any():
        raise ValueError("Scores contain NaN; a NaN cannot be ranked")
    order = np.argsort(-s, kind="stable")
    ranked = s[order]
    true_positives = np.cumsum(y[order], dtype=np.int64)
    if not len(ranked) or true_positives[-1] == 0:
        raise ValueError("No positive row: recall is undefined")
    last_of_each_score = np.append(np.flatnonzero(ranked[1:] != ranked[:-1]), len(ranked) - 1)
    return (
        ranked[last_of_each_score],
        last_of_each_score.astype(np.int64) + 1,
        true_positives[last_of_each_score],
    )


def recall_at_precision(
    labels: np.ndarray, scores: np.ndarray, min_precision: float | Fraction
) -> OperatingPoint | None:
    """The point with the largest recall among those with precision >= ``min_precision``.

    Tie policy, declared before measuring:

    1. A threshold flags every row whose score is at least the threshold, so rows with
       equal scores are flagged together. A tie in score is never split, because no
       threshold can split it.
    2. When several thresholds reach the floor, the largest recall wins. Thresholds with
       the same precision are therefore resolved by recall: the lower threshold, which
       flags more frauds at that precision, is reported.
    3. When several thresholds share that largest recall (they differ only in legitimate
       rows flagged), the highest one is reported: it flags the fewest rows and has the
       highest precision.
    4. If no threshold reaches the floor, there is no point: the caller reports recall 0
       and marks the floor as not attained.

    Precision is compared exactly, in integers: a point whose precision equals the floor
    counts as reaching it.

    Args:
        labels: 0/1 labels.
        scores: Scores, higher meaning more likely positive.
        min_precision: The floor; a float is read as the simplest nearby fraction (0.99 is
            99/100), a ``Fraction`` is used as given.

    Returns:
        The operating point, or None if no threshold reaches the floor.

    Raises:
        ValueError: If there is no positive, a score is NaN, or the shapes differ.
    """
    thresholds, flagged, true_positives = _threshold_curve(labels, scores)
    floor = _as_fraction(min_precision)
    meets = true_positives * floor.denominator >= flagged * floor.numerator
    if not meets.any():
        return None
    best = true_positives[meets].max()
    # Thresholds run from highest to lowest, so the first match is the highest threshold.
    index = int(np.flatnonzero(meets & (true_positives == best))[0])
    return OperatingPoint(
        threshold=float(thresholds[index]),
        flagged=int(flagged[index]),
        true_positives=int(true_positives[index]),
        positives=int(true_positives[-1]),
    )


def precision_at_recall(
    labels: np.ndarray, scores: np.ndarray, min_recall: float | Fraction
) -> OperatingPoint:
    """The point with the largest precision among those with recall >= ``min_recall``.

    Same rule for tied scores as ``recall_at_precision``. Among thresholds with the same
    largest precision, the one with the larger recall (the lower threshold) is reported.
    Recall is compared exactly, in integers.

    Args:
        labels: 0/1 labels.
        scores: Scores, higher meaning more likely positive.
        min_recall: The recall to reach, at most 1.

    Returns:
        The operating point. One always exists: the lowest threshold has recall 1.

    Raises:
        ValueError: If ``min_recall`` exceeds 1, or for the reasons of
            ``recall_at_precision``.
    """
    floor = _as_fraction(min_recall)
    if floor > 1:
        raise ValueError(f"Recall cannot reach {min_recall}")
    thresholds, flagged, true_positives = _threshold_curve(labels, scores)
    positives = true_positives[-1]
    reaches = np.flatnonzero(true_positives * floor.denominator >= positives * floor.numerator)
    precision = true_positives[reaches] / flagged[reaches]
    index = int(reaches[np.flatnonzero(precision == precision.max())[-1]])
    return OperatingPoint(
        threshold=float(thresholds[index]),
        flagged=int(flagged[index]),
        true_positives=int(true_positives[index]),
        positives=int(positives),
    )


@dataclass(frozen=True)
class RankingHead:
    """What the top of a ranking holds.

    Attributes:
        k: Rank asked for.
        flagged: Rows whose score is at least the k-th highest score; more than ``k`` when
            the k-th score is tied, because a tie is never split.
        positives: Positive rows among them.
        marked_positives: Positive rows among them that are also marked (for example, rows
            with the signature).
    """

    k: int
    flagged: int
    positives: int
    marked_positives: int


def ranking_head(
    labels: np.ndarray, scores: np.ndarray, marked: np.ndarray, ks: tuple[int, ...]
) -> tuple[RankingHead, ...]:
    """Count positives, and marked positives, among the top ``k`` rows of a ranking.

    Same tie rule as ``recall_at_precision``: the cut is the threshold of the k-th row, so
    every row tied with it is in.

    Args:
        labels: 0/1 labels.
        scores: Scores, higher meaning more likely positive.
        marked: Boolean mark per row.
        ks: Ranks to report, each between 1 and the number of rows.

    Returns:
        One ``RankingHead`` per ``k``, in the order given.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    mark = np.asarray(marked).astype(bool)
    ranked = np.sort(s)[::-1]
    heads = []
    for k in ks:
        flagged = s >= ranked[k - 1]
        heads.append(
            RankingHead(
                k=k,
                flagged=int(flagged.sum()),
                positives=int(y[flagged].sum()),
                marked_positives=int(y[flagged & mark].sum()),
            )
        )
    return tuple(heads)


def average_precision_by_block(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Split average precision into the contribution of each block of tied scores.

    Average precision is ``sum over thresholds of (R_n - R_(n-1)) * P_n``, and each
    threshold is one block of tied scores, so each block's contribution can be read
    directly. The contributions add up to scikit-learn's ``average_precision_score``.

    Args:
        labels: 0/1 labels.
        scores: Scores, higher meaning more likely positive.

    Returns:
        One row per block, highest score first: ``threshold``, ``rows`` and ``positives``
        in the block, ``flagged`` and ``true_positives`` down to and including it, the
        ``precision`` and ``recall`` there, and the block's ``contribution``.
    """
    thresholds, flagged, true_positives = _threshold_curve(labels, scores)
    positives = true_positives[-1]
    block_positives = np.diff(true_positives, prepend=0)
    precision = true_positives / flagged
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "rows": np.diff(flagged, prepend=0),
            "positives": block_positives,
            "flagged": flagged,
            "true_positives": true_positives,
            "precision": precision,
            "recall": true_positives / positives,
            "contribution": block_positives / positives * precision,
        }
    )


def floor_key(floor: float) -> str:
    """Name fragment of a precision floor in metric names, e.g. ``p0.99``."""
    return f"p{floor:.2f}"


def score_fold(
    labels: np.ndarray, scores: np.ndarray, baseline_flags: np.ndarray
) -> dict[str, float]:
    """Every metric of one validation fold.

    - ``recall_at_p0.99`` and ``recall_at_p0.95``: the decision and secondary metrics,
      0 when the floor is not reached; ``attained_p*`` says whether it was, and
      ``precision_at_p*`` and ``threshold_at_p*`` locate the point when it exists.
    - ``pr_auc`` (average precision: a constant score gives exactly the prevalence),
      ``roc_auc``, ``brier`` and ``prevalence``: context.
    - ``baseline_precision`` and ``baseline_recall``: the signature rule's point on the
      same rows. ``precision_at_baseline_recall`` and ``recall_at_baseline_precision``
      place that point against the scored model's PR curve: the model dominates the rule
      where the first exceeds ``baseline_precision`` or the second ``baseline_recall``.

    Args:
        labels: True 0/1 labels of the validation rows.
        scores: The model's scores for the positive class.
        baseline_flags: The signature rule's 0/1 flags on the same rows.

    Returns:
        Metric name to value.
    """
    y = np.asarray(labels)
    metrics: dict[str, float] = {
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
    }
    for floor in PRECISION_FLOORS:
        key = floor_key(floor)
        point = recall_at_precision(y, scores, floor)
        metrics[f"recall_at_{key}"] = point.recall if point else 0.0
        metrics[f"attained_{key}"] = float(point is not None)
        if point is not None:
            metrics[f"precision_at_{key}"] = point.precision
            metrics[f"threshold_at_{key}"] = point.threshold
    flags = np.asarray(baseline_flags).astype(bool)
    flagged = int(flags.sum())
    caught = int(y[flags].sum())
    positives = int(y.sum())
    metrics["baseline_recall"] = caught / positives
    if flagged:
        metrics["baseline_precision"] = caught / flagged
        at_recall = precision_at_recall(y, scores, Fraction(caught, positives))
        metrics["precision_at_baseline_recall"] = at_recall.precision
        at_precision = recall_at_precision(y, scores, Fraction(caught, flagged))
        metrics["recall_at_baseline_precision"] = at_precision.recall if at_precision else 0.0
    return metrics


# --- The fold loop ----------------------------------------------------------------------

EstimatorFactory = Callable[[], Any]
"""A zero-argument callable returning a new, unfitted scikit-learn estimator."""


@dataclass(frozen=True)
class FoldResult:
    """What one fold measured.

    Attributes:
        fold: Fold number, from 1.
        train_rows: Rows the model was fitted on.
        train_positives: Positive labels among them, as given to ``fit``.
        validation_rows: Rows scored.
        validation_positives: True positive labels among them.
        train_steps: Lowest and highest step in the training rows.
        validation_steps: Lowest and highest step in the validation rows.
        train_signature_rows: Training rows where ``amount_equals_origin_balance`` holds.
        train_signature_positives: How many of those carry a positive label as given to
            ``fit``; with shuffled labels, how many positives fell on them by chance.
        label_shuffled: Whether the training labels were permuted.
        fit_seconds: Wall time of ``fit``, preprocessor included.
        predict_seconds: Wall time of scoring the validation rows.
        fit_warnings: Distinct warnings raised during ``fit``.
        metrics: The output of ``score_fold``.
    """

    fold: int
    train_rows: int
    train_positives: int
    validation_rows: int
    validation_positives: int
    train_steps: tuple[int, int]
    validation_steps: tuple[int, int]
    train_signature_rows: int
    train_signature_positives: int
    label_shuffled: bool
    fit_seconds: float
    predict_seconds: float
    fit_warnings: tuple[str, ...]
    metrics: dict[str, float]


@dataclass(frozen=True)
class FoldEvent:
    """Progress of ``cross_validate``: a fit started, a fit finished, or a fold is done."""

    kind: str
    fold: int
    result: FoldResult | None = None


@dataclass(frozen=True)
class Aggregate:
    """A metric across folds: mean and sample standard deviation (ddof=1)."""

    mean: float
    std: float
    folds: int


def aggregate(folds: tuple[FoldResult, ...]) -> dict[str, Aggregate]:
    """Mean and sample standard deviation of every metric present in every fold.

    A metric missing from some fold (the threshold of a floor that one fold did not reach)
    is left out instead of being averaged over fewer folds.

    Args:
        folds: Fold results.

    Returns:
        Metric name to its aggregate; empty if there are no folds.
    """
    if not folds:
        return {}
    shared = set.intersection(*(set(fold.metrics) for fold in folds))
    extra = {
        "fit_seconds": [fold.fit_seconds for fold in folds],
        "predict_seconds": [fold.predict_seconds for fold in folds],
    }
    values = {name: [fold.metrics[name] for fold in folds] for name in sorted(shared)} | extra
    return {
        name: Aggregate(
            mean=float(np.mean(series)),
            std=float(np.std(series, ddof=1)) if len(series) > 1 else float("nan"),
            folds=len(series),
        )
        for name, series in values.items()
    }


def load_evaluation_frame() -> pd.DataFrame:
    """Read the reference period and keep only the columns the harness uses.

    Returns:
        The model-eligible columns and the label, every row of the reference period.
    """
    return read_reference()[list(EVALUATION_COLUMNS)]


def fold_table(folds: tuple[FoldResult, ...], metrics: tuple[str, ...] = ()) -> pd.DataFrame:
    """One row per fold: step ranges, rows and positives on each side, fit time, metrics.

    Args:
        folds: Fold results.
        metrics: Names from each fold's ``metrics`` to add as columns.

    Returns:
        The table, indexed by fold number.
    """
    rows = [
        {
            "fold": fold.fold,
            "train_steps": f"{fold.train_steps[0]}-{fold.train_steps[1]}",
            "train_rows": fold.train_rows,
            "train_positives": fold.train_positives,
            "validation_steps": f"{fold.validation_steps[0]}-{fold.validation_steps[1]}",
            "validation_rows": fold.validation_rows,
            "validation_positives": fold.validation_positives,
            "fit_seconds": round(fold.fit_seconds, 1),
            **{name: fold.metrics.get(name, float("nan")) for name in metrics},
        }
        for fold in folds
    ]
    return pd.DataFrame(rows).set_index("fold")


def features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate the label from everything else, so no preprocessor can read it.

    Args:
        frame: Rows with the label column.

    Returns:
        The frame without the label, and the labels as an integer array.
    """
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN].to_numpy(dtype=np.int64)


def derived_seeds(count: int) -> tuple[int, ...]:
    """Seeds for repeated draws: ``settings.RANDOM_STATE + i`` for ``i`` in ``0 .. count - 1``.

    The first is ``settings.RANDOM_STATE`` itself, so draw 0 reproduces every single-draw
    result measured before. numpy turns an integer seed into a stream through
    ``SeedSequence``, which hashes it, so consecutive integers give unrelated streams.

    Args:
        count: How many seeds.

    Returns:
        The seeds, in order.
    """
    return tuple(settings.RANDOM_STATE + offset for offset in range(count))


def shuffled_labels(labels: np.ndarray, seed: int = settings.RANDOM_STATE) -> np.ndarray:
    """Permute labels; the class counts stay exactly equal.

    Args:
        labels: Labels to permute.
        seed: Seed of the permutation; one of ``derived_seeds`` when repeating the test.

    Returns:
        A new array with the same values in a different order.
    """
    return np.random.default_rng(seed).permutation(labels)


def _step_range(frame: pd.DataFrame) -> tuple[int, int]:
    steps = frame[STEP_COLUMN]
    return int(steps.min()), int(steps.max())


def _warning_text(caught: list[warnings.WarningMessage]) -> tuple[str, ...]:
    texts = (f"{item.category.__name__}: {str(item.message)[:300]}" for item in caught)
    return tuple(dict.fromkeys(texts))


def compose(preprocessor: EstimatorFactory, model: EstimatorFactory) -> Pipeline:
    """Build the one Pipeline shape of the project: a fresh preprocessor, then a fresh model.

    The harness fits every fold with it and ``models.registry`` builds the artifact with it,
    so the evaluated object and the served object have the same structure by construction.

    Args:
        preprocessor: Factory of the preprocessing step.
        model: Factory of the classifier.

    Returns:
        ``Pipeline([("preprocess", ...), ("model", ...)])``, unfitted.
    """
    return Pipeline([("preprocess", preprocessor()), ("model", model())])


@dataclass(frozen=True)
class _FittedFold:
    """A fold after fitting and scoring: what both evaluate_fold and score_one_fold need."""

    estimator: Pipeline
    train: pd.DataFrame
    train_labels: np.ndarray
    validation: pd.DataFrame
    validation_labels: np.ndarray
    scores: np.ndarray
    fit_seconds: float
    predict_seconds: float
    fit_warnings: tuple[str, ...]


def _fit_and_score(
    features: pd.DataFrame,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    preprocessor: EstimatorFactory,
    model: EstimatorFactory,
    *,
    fold: int,
    shuffle_training_labels: bool,
    shuffle_seed: int,
    on_event: Callable[[FoldEvent], None] | None,
) -> _FittedFold:
    """Fit a fresh Pipeline on the training rows and score the validation rows."""
    notify = on_event or (lambda _event: None)
    train = features.iloc[train_index]
    validation = features.iloc[validation_index]
    train_labels = labels[train_index]
    if shuffle_training_labels:
        train_labels = shuffled_labels(train_labels, shuffle_seed)
    if set(np.unique(train_labels).tolist()) != {0, 1}:
        raise ValueError(f"Fold {fold}: the training labels must hold both classes")

    estimator = compose(preprocessor, model)
    notify(FoldEvent(FIT_STARTED, fold))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        estimator.fit(train, train_labels)
        fit_seconds = time.perf_counter() - started
    notify(FoldEvent(FIT_FINISHED, fold))

    started = time.perf_counter()
    probabilities = estimator.predict_proba(validation)
    predict_seconds = time.perf_counter() - started
    return _FittedFold(
        estimator=estimator,
        train=train,
        train_labels=train_labels,
        validation=validation,
        validation_labels=labels[validation_index],
        scores=probabilities[:, list(estimator.classes_).index(1)],
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        fit_warnings=_warning_text(caught),
    )


def evaluate_fold(
    features: pd.DataFrame,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    preprocessor: EstimatorFactory,
    model: EstimatorFactory,
    *,
    fold: int,
    shuffle_training_labels: bool = False,
    shuffle_seed: int = settings.RANDOM_STATE,
    on_event: Callable[[FoldEvent], None] | None = None,
) -> FoldResult:
    """Fit a fresh Pipeline on the training rows and score the validation rows.

    Args:
        features: Every row, without the label column.
        labels: Every row's 0/1 label.
        train_index: Positions of the training rows.
        validation_index: Positions of the validation rows.
        preprocessor: Factory of the preprocessing step; called once, here.
        model: Factory of the classifier; called once, here.
        fold: Fold number, for the result and the events.
        shuffle_training_labels: Permute the training labels before fitting.
        shuffle_seed: Seed of that permutation.
        on_event: Called with ``FIT_STARTED`` and ``FIT_FINISHED`` around ``fit``.

    Returns:
        The fold's sizes, timings and metrics.

    Raises:
        ValueError: If the training labels do not hold both classes.
    """
    fitted = _fit_and_score(
        features,
        labels,
        train_index,
        validation_index,
        preprocessor,
        model,
        fold=fold,
        shuffle_training_labels=shuffle_training_labels,
        shuffle_seed=shuffle_seed,
        on_event=on_event,
    )
    signature_inputs = list(DERIVATION_INPUTS[SIGNATURE_FEATURE])
    signature = amount_equals_origin_balance(fitted.train[signature_inputs]).to_numpy()
    baseline_flags = SignatureRule().predict(fitted.validation)
    return FoldResult(
        fold=fold,
        train_rows=len(fitted.train),
        train_positives=int(fitted.train_labels.sum()),
        validation_rows=len(fitted.validation),
        validation_positives=int(fitted.validation_labels.sum()),
        train_steps=_step_range(fitted.train),
        validation_steps=_step_range(fitted.validation),
        train_signature_rows=int(signature.sum()),
        train_signature_positives=int(fitted.train_labels[signature].sum()),
        label_shuffled=shuffle_training_labels,
        fit_seconds=fitted.fit_seconds,
        predict_seconds=fitted.predict_seconds,
        fit_warnings=fitted.fit_warnings,
        metrics=score_fold(fitted.validation_labels, fitted.scores, baseline_flags),
    )


@dataclass(frozen=True)
class FoldScores:
    """One fold refitted to inspect its ranking, by the same code path as ``evaluate_fold``.

    Attributes:
        estimator: The fitted Pipeline.
        validation: The validation rows, without the label.
        labels: Their true labels.
        scores: The model's scores for the positive class.
    """

    estimator: Pipeline
    validation: pd.DataFrame
    labels: np.ndarray
    scores: np.ndarray


def score_one_fold(
    frame: pd.DataFrame,
    splitter: Splitter,
    fold: int,
    preprocessor: EstimatorFactory,
    model: EstimatorFactory,
    *,
    shuffle_training_labels: bool = False,
    shuffle_seed: int = settings.RANDOM_STATE,
) -> FoldScores:
    """Refit one fold and keep its scores, which ``cross_validate`` does not return.

    It runs the same fit-and-score step as ``evaluate_fold``, so the scores are those that
    produced the fold's metrics: diagnostics read the ranking of the evaluated model, not of
    a copy.

    Args:
        frame: Rows with the label column.
        splitter: The splitter of the evaluation.
        fold: Fold number, from 1.
        preprocessor: Factory of the preprocessing step.
        model: Factory of the classifier.
        shuffle_training_labels: Permute the training labels, as in the permutation test.
        shuffle_seed: Seed of that permutation.

    Returns:
        The fitted Pipeline, the validation rows, their labels and their scores.
    """
    features, labels = features_and_labels(frame)
    train_index, validation_index = list(splitter.split(features, labels))[fold - 1]
    fitted = _fit_and_score(
        features,
        labels,
        train_index,
        validation_index,
        preprocessor,
        model,
        fold=fold,
        shuffle_training_labels=shuffle_training_labels,
        shuffle_seed=shuffle_seed,
        on_event=None,
    )
    return FoldScores(fitted.estimator, fitted.validation, fitted.validation_labels, fitted.scores)


def cross_validate(
    frame: pd.DataFrame,
    splitter: Splitter,
    preprocessor: EstimatorFactory,
    model: EstimatorFactory,
    *,
    shuffle_training_labels: bool = False,
    shuffle_seed: int = settings.RANDOM_STATE,
    on_event: Callable[[FoldEvent], None] | None = None,
) -> tuple[FoldResult, ...]:
    """Evaluate a model on every fold of a splitter.

    Args:
        frame: Rows with the label column and every column the preprocessor reads.
        splitter: Yields the folds; ``ExpandingStepSplit`` or ``stratified_kfold()``.
        preprocessor: Factory of the preprocessing step, called once per fold.
        model: Factory of the classifier, called once per fold.
        shuffle_training_labels: Permute the training labels of every fold before fitting;
            validation labels are never touched.
        shuffle_seed: Seed of the permutation, the same for every fold.
        on_event: Called around each fit and with ``FOLD_DONE`` after each fold.

    Returns:
        One result per fold, in order.
    """
    notify = on_event or (lambda _event: None)
    features, labels = features_and_labels(frame)
    results = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features, labels), start=1
    ):
        result = evaluate_fold(
            features,
            labels,
            train_index,
            validation_index,
            preprocessor,
            model,
            fold=fold,
            shuffle_training_labels=shuffle_training_labels,
            shuffle_seed=shuffle_seed,
            on_event=notify,
        )
        results.append(result)
        notify(FoldEvent(FOLD_DONE, fold, result))
    return tuple(results)
