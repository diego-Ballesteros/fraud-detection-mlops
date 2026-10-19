"""Tests of fraud_detection.models.evaluation: splitters, metrics and the fold loop.

Expected values are computed by hand from the rows written here. Every structural check
on the splitter also runs on a splitter known to break it, so a clean result means
something (docs/METHODOLOGY.md, section 7.6).
"""

from fractions import Fraction
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer

from fraud_detection import settings
from fraud_detection.data.schema import COLUMNS, TARGET_COLUMN
from fraud_detection.models import estimators, evaluation
from fraud_detection.models.evaluation import (
    Aggregate,
    ExpandingStepSplit,
    FoldEvent,
    OperatingPoint,
    StepWindow,
    aggregate,
    cross_validate,
    precision_at_recall,
    recall_at_precision,
    score_fold,
)

# --- A synthetic reference period ------------------------------------------------------------


def synthetic_frame(seed: int = 0) -> pd.DataFrame:
    """Steps 1..400 with 1 to 3 rows each, rows in random order, positives in every window.

    Positives at steps with step % 10 == 0, the first row of each such step. Those at steps
    with step % 20 == 0 take the whole balance, so the signature rule flags two of the four
    positives of every validation window, and no legitimate row.
    """
    rng = np.random.default_rng(seed)
    steps = np.repeat(np.arange(1, 401), np.arange(1, 401) % 3 + 1)
    first_of_step = np.r_[True, steps[1:] != steps[:-1]]
    positive = (steps % 10 == 0) & first_of_step
    amount = rng.uniform(10, 1000, len(steps)).round(2)
    balance = amount + rng.uniform(1, 500, len(steps)).round(2)
    takes_all = positive & (steps % 20 == 0)
    balance[takes_all] = amount[takes_all]
    frame = pd.DataFrame(
        {
            "step": steps,
            "type": np.where(positive, "TRANSFER", "PAYMENT"),
            "amount": amount,
            "oldbalanceOrg": balance,
            "oldbalanceDest": rng.uniform(0, 100, len(steps)).round(2),
            TARGET_COLUMN: positive.astype(int),
        }
    )
    frame = frame.astype({name: COLUMNS[name].dtype for name in frame.columns})
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_frame()


# --- The temporal splitter -------------------------------------------------------------------


def fold_violations(steps: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[str]:
    """Every way a fold can break the temporal contract, one line per problem."""
    problems = []
    for number, (train, validation) in enumerate(folds, start=1):
        shared = set(steps[train]) & set(steps[validation])
        if shared:
            problems.append(f"fold {number}: steps in both parts {sorted(shared)[:3]}")
        start = int(steps[validation].min())
        if int(steps[train].max()) >= start:
            problems.append(f"fold {number}: training reaches step {steps[train].max()}")
        missing = set(steps[steps < start]) - set(steps[train])
        if missing:
            problems.append(f"fold {number}: history not in training {sorted(missing)[:3]}")
    return problems


def test_the_validation_windows_are_the_declared_ones() -> None:
    windows = ExpandingStepSplit().validation_windows
    assert windows == (
        StepWindow(201, 240),
        StepWindow(241, 280),
        StepWindow(281, 320),
        StepWindow(321, 360),
        StepWindow(361, 400),
    )
    # The last window ends on the last reference step: the default is derived, not typed.
    assert evaluation.FIRST_VALIDATION_STEP == 201
    assert windows[-1].last == settings.REFERENCE_LAST_STEP


def test_no_step_is_on_both_sides_and_training_is_the_whole_prefix(frame: pd.DataFrame) -> None:
    steps = frame["step"].to_numpy()
    folds = list(ExpandingStepSplit().split(frame))
    assert len(folds) == 5
    assert fold_violations(steps, folds) == []
    for (train, validation), window in zip(
        folds, ExpandingStepSplit().validation_windows, strict=True
    ):
        # Training is exactly steps 1 .. start - 1, ending just before its validation.
        assert set(steps[train]) == set(range(1, window.first))
        assert int(steps[train].max()) == window.first - 1
        assert set(steps[validation]) == set(range(window.first, window.last + 1))


def test_a_row_index_splitter_breaks_the_same_checks(frame: pd.DataFrame) -> None:
    # Positive control: TimeSeriesSplit cuts by row position, so on rows sorted by step with
    # 1 to 3 rows per step some border falls inside a step, and the checks must say so.
    ordered = frame.sort_values("step", kind="stable").reset_index(drop=True)
    steps = ordered["step"].to_numpy()
    folds = list(TimeSeriesSplit(n_splits=5).split(ordered))
    problems = fold_violations(steps, folds)
    assert any("steps in both parts" in problem for problem in problems)


def test_the_validations_cover_201_to_400_exactly_once(frame: pd.DataFrame) -> None:
    steps = frame["step"].to_numpy()
    validations = [validation for _, validation in ExpandingStepSplit().split(frame)]
    rows = np.concatenate(validations)
    # No row in two validations, and the steps they cover are exactly 201..400.
    assert len(rows) == len(set(rows.tolist()))
    assert sorted(set(steps[rows].tolist())) == list(range(201, 401))
    per_fold = [set(steps[validation].tolist()) for validation in validations]
    assert sum(len(fold) for fold in per_fold) == 200


def test_production_rows_are_refused(frame: pd.DataFrame) -> None:
    leaked = frame.copy()
    leaked.loc[0, "step"] = settings.PRODUCTION_FIRST_STEP
    with pytest.raises(ValueError, match="production"):
        list(ExpandingStepSplit().split(leaked))
    # Control: the same frame with every step in reference splits.
    assert len(list(ExpandingStepSplit().split(frame))) == 5


def test_windows_past_the_reference_period_or_without_history_are_refused() -> None:
    with pytest.raises(ValueError, match="past the reference period"):
        ExpandingStepSplit(first_validation_step=202)
    with pytest.raises(ValueError, match="at least one step"):
        ExpandingStepSplit(first_validation_step=1, window=10, n_splits=1)
    assert ExpandingStepSplit(first_validation_step=391, window=10, n_splits=1).n_splits == 1


def test_the_contrast_splitter_is_seeded_from_settings() -> None:
    splitter = evaluation.stratified_kfold()
    assert splitter.n_splits == 5
    assert splitter.shuffle is True
    assert splitter.random_state == settings.RANDOM_STATE
    assert evaluation.SPLITTERS[evaluation.DECIDING_SPLITTER] is ExpandingStepSplit


# --- Recall at a precision floor: the tie policy ----------------------------------------------


def _y(*values: int) -> np.ndarray:
    return np.asarray(values)


def _s(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=float)


def test_tied_scores_are_never_split() -> None:
    # Only a tie-splitting ranking could flag the positive alone at precision 1.
    point = recall_at_precision(_y(1, 0, 0), _s(0.9, 0.9, 0.5), 0.99)
    assert point is None


def test_thresholds_with_the_same_precision_resolve_to_the_larger_recall() -> None:
    # t=0.9: P=1, R=1/3 · t=0.8: P=1, R=2/3 · t=0.7: P=2/3 · t=0.6: P=3/4
    point = recall_at_precision(_y(1, 1, 0, 1), _s(0.9, 0.8, 0.7, 0.6), 0.99)
    assert point == OperatingPoint(threshold=0.8, flagged=2, true_positives=2, positives=3)


def test_thresholds_with_the_same_recall_resolve_to_the_highest() -> None:
    # Recall 1 is reached at t=0.8 (P=1), 0.7 (P=2/3) and 0.6 (P=1/2): report t=0.8.
    point = recall_at_precision(_y(1, 1, 0, 0), _s(0.9, 0.8, 0.7, 0.6), 0.5)
    assert point == OperatingPoint(threshold=0.8, flagged=2, true_positives=2, positives=2)
    assert point.precision == 1.0


def test_a_tie_block_that_mixes_classes_moves_as_one() -> None:
    # Scores 0.9 0.9 0.9 | 0.5: the block of three holds both positives and one negative,
    # so its precision is 2/3. Splitting it would give precision 1 on the first two rows.
    labels, scores = _y(1, 1, 0, 0), _s(0.9, 0.9, 0.9, 0.5)
    assert recall_at_precision(labels, scores, 0.7) is None
    assert recall_at_precision(labels, scores, Fraction(2, 3)) == OperatingPoint(0.9, 3, 2, 2)
    # Both thresholds reach recall 1 at a floor of 1/2: the higher one is reported.
    assert recall_at_precision(labels, scores, 0.5) == OperatingPoint(0.9, 3, 2, 2)


def test_a_precision_equal_to_the_floor_reaches_it() -> None:
    # t=0.6 flags 4 rows with 3 positives: precision exactly 0.75.
    point = recall_at_precision(_y(1, 1, 0, 1, 0), _s(0.9, 0.8, 0.7, 0.6, 0.1), 0.75)
    assert point == OperatingPoint(0.6, 4, 3, 3)
    assert recall_at_precision(_y(1, 1, 0, 1, 0), _s(0.9, 0.8, 0.7, 0.6, 0.1), 0.76) == (
        OperatingPoint(0.8, 2, 2, 3)
    )


def test_precision_is_not_monotone_so_every_threshold_is_examined() -> None:
    # The top row is negative: precision rises as the threshold falls. Stopping at the first
    # threshold below the floor would report nothing.
    point = recall_at_precision(_y(0, 1, 1, 1, 1), _s(0.9, 0.8, 0.7, 0.6, 0.5), 0.8)
    assert point == OperatingPoint(0.5, 5, 4, 4)


def test_recall_at_precision_agrees_with_scikit_learn_on_random_ties() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        labels = (rng.random(300) < 0.2).astype(int)
        scores = np.round(rng.random(300) * 0.6 + 0.4 * labels, 1)  # coarse: many ties
        for floor in (0.5, 0.8, 0.95):
            precision, recall, _ = precision_recall_curve(labels, scores)
            # The last pair (precision 1, recall 0) belongs to no threshold.
            reachable = recall[:-1][precision[:-1] >= floor]
            point = recall_at_precision(labels, scores, floor)
            expected = reachable.max() if len(reachable) else None
            assert (point.recall if point else None) == expected


def test_precision_at_recall_and_its_ties() -> None:
    labels, scores = _y(1, 0, 1, 0, 1, 0), _s(0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
    # Recall >= 2/3 is reached at t=0.7 (P=2/3), 0.6 (P=1/2), 0.5 (P=3/5)...
    assert precision_at_recall(labels, scores, Fraction(2, 3)) == OperatingPoint(0.7, 3, 2, 3)
    # t=0.9 and t=0.8 both have precision 1 (recall 1/4 and 2/4): report the larger recall.
    tied = precision_at_recall(_y(1, 1, 0, 1, 1), _s(0.9, 0.8, 0.7, 0.6, 0.5), 0.25)
    assert tied == OperatingPoint(0.8, 2, 2, 4)
    assert precision_at_recall(_y(1, 1, 0), _s(0.9, 0.9, 0.1), 1.0) == OperatingPoint(0.9, 2, 2, 2)
    with pytest.raises(ValueError, match="cannot reach"):
        precision_at_recall(labels, scores, 1.5)


def test_invalid_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="No positive"):
        recall_at_precision(_y(0, 0), _s(0.1, 0.2), 0.9)
    with pytest.raises(ValueError, match="NaN"):
        recall_at_precision(_y(1, 0), _s(np.nan, 0.2), 0.9)
    with pytest.raises(ValueError, match="equal-length"):
        recall_at_precision(_y(1, 0), _s(0.1), 0.9)


# --- The metrics of one fold -------------------------------------------------------------------


def test_a_constant_score_gives_the_prevalence_as_pr_auc() -> None:
    # The control of the majority-class predictor: every score tied at 0.
    labels = np.r_[np.ones(7, dtype=int), np.zeros(9993, dtype=int)]
    metrics = score_fold(labels, np.zeros(len(labels)), np.zeros(len(labels)))
    assert metrics["pr_auc"] == pytest.approx(0.0007, rel=1e-12)
    assert metrics["prevalence"] == pytest.approx(0.0007, rel=1e-12)
    assert metrics["roc_auc"] == 0.5
    assert metrics["brier"] == pytest.approx(0.0007, rel=1e-12)
    assert metrics["recall_at_p0.99"] == 0.0
    assert metrics["attained_p0.99"] == 0.0
    assert "threshold_at_p0.99" not in metrics


def test_the_baseline_point_is_placed_on_the_model_curve() -> None:
    # 4 positives. The rule flags rows 0, 1 and 3: 2 positives and 1 negative.
    labels = _y(1, 1, 0, 0, 1, 1, 0, 0)
    rule = _y(1, 1, 0, 1, 0, 0, 0, 0)
    scores = _s(0.9, 0.8, 0.2, 0.7, 0.6, 0.1, 0.3, 0.05)
    metrics = score_fold(labels, scores, rule)
    assert metrics["baseline_recall"] == 0.5
    assert metrics["baseline_precision"] == pytest.approx(2 / 3)
    # Recall >= 1/2: t=0.8 already has P=1.
    assert metrics["precision_at_baseline_recall"] == 1.0
    # Precision >= 2/3: t=0.6 flags 4 rows with 3 positives (P=3/4), recall 3/4.
    assert metrics["recall_at_baseline_precision"] == 0.75
    assert metrics["recall_at_p0.99"] == 0.5
    assert metrics["threshold_at_p0.99"] == 0.8
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(labels, scores))


# --- The fold loop -----------------------------------------------------------------------------


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Record the labels and columns it was fitted on; score by amount."""

    fitted_labels: ClassVar[list[np.ndarray]] = []
    fitted_columns: ClassVar[list[list[str]]] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SpyClassifier":  # noqa: N803
        self.classes_ = np.unique(y)
        SpyClassifier.fitted_labels.append(np.asarray(y).copy())
        SpyClassifier.fitted_columns.append(list(X.columns))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        score = np.asarray(X["amount"], dtype=float) / 1000
        return np.column_stack([1 - score, score])


class SpyTransformer(TransformerMixin, BaseEstimator):
    """Record the columns it sees; pass the rows through."""

    seen: ClassVar[list[list[str]]] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SpyTransformer":  # noqa: N803
        SpyTransformer.seen.append(list(X.columns))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        SpyTransformer.seen.append(list(X.columns))
        return X


@pytest.fixture
def spies() -> None:
    SpyClassifier.fitted_labels.clear()
    SpyClassifier.fitted_columns.clear()
    SpyTransformer.seen.clear()


@pytest.mark.usefixtures("spies")
def test_the_factories_are_called_once_per_fold_and_build_new_objects(frame: pd.DataFrame) -> None:
    built: list[FunctionTransformer] = []

    def preprocessor() -> FunctionTransformer:
        built.append(FunctionTransformer())
        return built[-1]

    folds = cross_validate(frame, ExpandingStepSplit(), preprocessor, SpyClassifier)
    assert [fold.fold for fold in folds] == [1, 2, 3, 4, 5]
    assert len(built) == 5
    assert len({id(item) for item in built}) == 5
    assert len(SpyClassifier.fitted_labels) == 5


@pytest.mark.usefixtures("spies")
def test_no_preprocessor_sees_the_label(frame: pd.DataFrame) -> None:
    cross_validate(frame, ExpandingStepSplit(), SpyTransformer, SpyClassifier)
    assert SpyTransformer.seen
    assert all(TARGET_COLUMN not in columns for columns in SpyTransformer.seen)
    assert all(TARGET_COLUMN not in columns for columns in SpyClassifier.fitted_columns)


@pytest.mark.usefixtures("spies")
def test_shuffling_permutes_only_the_training_labels(frame: pd.DataFrame) -> None:
    splitter = ExpandingStepSplit()
    plain = cross_validate(frame, splitter, FunctionTransformer, SpyClassifier)
    true_labels = SpyClassifier.fitted_labels.copy()
    SpyClassifier.fitted_labels.clear()
    shuffled = cross_validate(
        frame, splitter, FunctionTransformer, SpyClassifier, shuffle_training_labels=True
    )
    permuted = SpyClassifier.fitted_labels
    for original, given in zip(true_labels, permuted, strict=True):
        # Same class counts, different order.
        assert np.bincount(given).tolist() == np.bincount(original).tolist()
        assert not np.array_equal(given, original)
    for before, after in zip(plain, shuffled, strict=True):
        assert after.label_shuffled
        assert not before.label_shuffled
        # Validation is untouched: same rows, same true labels, and this spy scores by
        # amount alone, so every validation metric is identical.
        assert after.validation_positives == before.validation_positives
        assert after.metrics == before.metrics
        assert after.train_positives == before.train_positives
        # The training signature rows are the same; their labels are not.
        assert after.train_signature_rows == before.train_signature_rows
        assert before.train_signature_positives == before.train_signature_rows
        assert after.train_signature_positives < before.train_signature_positives


def test_the_permutation_is_seeded_from_settings() -> None:
    labels = np.r_[np.ones(10, dtype=int), np.zeros(90, dtype=int)]
    first, second = evaluation.shuffled_labels(labels), evaluation.shuffled_labels(labels)
    assert np.array_equal(first, second)
    expected = np.random.default_rng(settings.RANDOM_STATE).permutation(labels)
    assert np.array_equal(first, expected)


def test_the_majority_class_scores_the_prevalence_in_every_fold(frame: pd.DataFrame) -> None:
    spec = estimators.MODEL_SPECS["majority_class"]
    for splitter in (ExpandingStepSplit(), evaluation.stratified_kfold()):
        folds = cross_validate(frame, splitter, spec.preprocessor, spec.model)
        for fold in folds:
            prevalence = fold.validation_positives / fold.validation_rows
            assert fold.metrics["pr_auc"] == pytest.approx(prevalence, rel=1e-12)
            assert fold.metrics["roc_auc"] == 0.5


def test_the_signature_rule_through_the_harness(frame: pd.DataFrame) -> None:
    spec = estimators.MODEL_SPECS["signature_rule"]
    for fold in cross_validate(frame, ExpandingStepSplit(), spec.preprocessor, spec.model):
        # The rule flags only frauds here, so its recall at precision 1 is its own recall.
        assert fold.metrics["baseline_precision"] == 1.0
        assert fold.metrics["recall_at_p0.99"] == fold.metrics["baseline_recall"]
        assert fold.metrics["recall_at_baseline_precision"] == fold.metrics["baseline_recall"]


def test_the_events_bracket_each_fit(frame: pd.DataFrame) -> None:
    events: list[FoldEvent] = []
    cross_validate(
        frame, ExpandingStepSplit(), FunctionTransformer, DummyClassifier, on_event=events.append
    )
    kinds = [(event.kind, event.fold) for event in events]
    expected = [
        (kind, fold)
        for fold in range(1, 6)
        for kind in (evaluation.FIT_STARTED, evaluation.FIT_FINISHED, evaluation.FOLD_DONE)
    ]
    assert kinds == expected


def test_one_class_in_training_is_refused(frame: pd.DataFrame) -> None:
    negatives = frame.assign(**{TARGET_COLUMN: frame[TARGET_COLUMN] * 0})
    with pytest.raises(ValueError, match="both classes"):
        cross_validate(negatives, ExpandingStepSplit(), FunctionTransformer, DummyClassifier)


def test_aggregate_is_the_mean_and_the_sample_deviation(frame: pd.DataFrame) -> None:
    folds = cross_validate(frame, ExpandingStepSplit(), FunctionTransformer, DummyClassifier)
    prevalences = [fold.metrics["prevalence"] for fold in folds]
    summary = aggregate(folds)
    assert summary["prevalence"] == Aggregate(
        mean=pytest.approx(sum(prevalences) / 5),
        std=pytest.approx(float(np.sqrt(np.var(prevalences) * 5 / 4))),
        folds=5,
    )
    assert "fit_seconds" in summary
    assert aggregate(()) == {}


def test_the_fold_table_shows_steps_rows_and_positives(frame: pd.DataFrame) -> None:
    folds = cross_validate(frame, ExpandingStepSplit(), FunctionTransformer, DummyClassifier)
    table = evaluation.fold_table(folds, ("prevalence",))
    assert list(table.index) == [1, 2, 3, 4, 5]
    assert table.loc[1, "train_steps"] == "1-200"
    assert table.loc[5, "validation_steps"] == "361-400"
    # Positives sit at every tenth step: 4 per 40-step window, 20 in steps 1-200.
    assert table["validation_positives"].tolist() == [4, 4, 4, 4, 4]
    assert table.loc[1, "train_positives"] == 20
    assert table.loc[1, "prevalence"] == pytest.approx(4 / table.loc[1, "validation_rows"])
