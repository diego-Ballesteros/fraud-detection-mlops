"""Tests of fraud_detection.explain.global_explain, with expected values computed by hand."""

import copy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraud_detection.data.schema import TARGET_COLUMN
from fraud_detection.explain import global_explain
from fraud_detection.explain.global_explain import (
    ABOVE_GRID,
    BELOW_GRID,
    GRID_SIDES,
    INSIDE_GRID,
    LEGITIMATE,
    MONEY_BRANCH,
    OTHER_FRAUD,
    RULE_FRAUD,
    SIGNATURE_FEATURES,
    PermutationImportance,
    compare_rankings,
    decision_metric,
    explanation_sample,
    grid_side,
    ice_curves,
    log_grid,
    mean_abs_shap,
    model_features,
    model_importance,
    outside_grid,
    permutation_importance,
    plot_beeswarm,
    plot_ice,
    raw_grid,
    raw_split_thresholds,
    rule_groups,
    shap_values,
    signature_gate,
    trees_splitting_on,
)
from fraud_detection.models.estimators import full_preprocessor, lightgbm_classifier
from fraud_detection.models.evaluation import compose, derived_seeds
from tests.test_evaluation import synthetic_frame

FLAG = "amount_equals_origin_balance"


def _fit(seed: int = 0, shuffle: bool = False) -> Pipeline:
    frame = synthetic_frame(seed)
    labels = frame[TARGET_COLUMN].to_numpy()
    if shuffle:
        labels = np.random.default_rng(seed).permutation(labels)
    model = partial(lightgbm_classifier, n_estimators=20, min_child_samples=5)
    return compose(full_preprocessor, model).fit(frame.drop(columns=[TARGET_COLUMN]), labels)


@pytest.fixture(scope="module")
def fitted() -> Pipeline:
    return _fit()


def _means(**values: float) -> pd.Series:
    return mean_abs_shap(pd.DataFrame([values]))


# --- The sample -------------------------------------------------------------------------------


def test_the_sample_keeps_every_positive_and_draws_the_negatives_reproducibly() -> None:
    frame = pd.DataFrame({TARGET_COLUMN: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]})
    sample = explanation_sample(frame, negative_rows=4)
    assert len(sample) == 7
    assert sorted(sample.index[sample[TARGET_COLUMN] == 1]) == [0, 3, 7]
    assert (sample[TARGET_COLUMN] == 0).sum() == 4
    assert sample.index.equals(explanation_sample(frame, negative_rows=4).index)
    assert set(sample.index) <= set(frame.index)


# --- SHAP values and their additivity ---------------------------------------------------------


def test_model_features_are_named_by_the_preprocessor(fitted: Pipeline) -> None:
    rows = synthetic_frame().drop(columns=[TARGET_COLUMN]).head(5)
    features = model_features(fitted, rows)
    assert list(features.columns) == list(fitted[:-1].get_feature_names_out())
    assert FLAG in features.columns
    assert features.index.equals(rows.index)


def test_shap_values_add_up_to_the_raw_output(fitted: Pipeline) -> None:
    rows = synthetic_frame().drop(columns=[TARGET_COLUMN])
    result = shap_values(fitted, rows)
    assert result.values.shape == result.features.shape
    assert list(result.values.columns) == list(result.features.columns)
    rebuilt = result.base_value + result.values.sum(axis=1)
    np.testing.assert_allclose(rebuilt, result.raw_output, atol=1e-9)
    assert result.max_additivity_gap <= global_explain.ADDITIVITY_TOLERANCE
    # Not vacuous: the model attributes something to some feature.
    assert result.values.abs().to_numpy().max() > 0


def test_the_additivity_check_fails_when_the_explainer_explains_another_model(
    fitted: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = _fit(seed=1, shuffle=True)
    monkeypatch.setattr(
        global_explain,
        "tree_explainer",
        lambda _: global_explain.shap.TreeExplainer(
            other[-1], feature_perturbation="tree_path_dependent"
        ),
    )
    rows = synthetic_frame().drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="miss the raw output"):
        shap_values(fitted, rows)


# --- Mean |SHAP| ------------------------------------------------------------------------------


def test_mean_abs_shap_by_hand() -> None:
    values = pd.DataFrame({"a": [1.0, -3.0], "b": [-2.0, 0.0], "c": [0.5, 0.5]})
    # a: (1 + 3) / 2 = 2; b: (2 + 0) / 2 = 1; c: 0.5.
    expected = pd.Series({"a": 2.0, "b": 1.0, "c": 0.5})
    pd.testing.assert_series_equal(mean_abs_shap(values), expected)


# --- The gate ---------------------------------------------------------------------------------


def test_the_gate_passes_with_a_majority_and_the_flag_second() -> None:
    # Signature: 3 + 1 + 1 = 5 of 9 = 0.556; the flag is behind hour_sin only.
    gate = signature_gate(
        _means(hour_sin=4.0, amount_equals_origin_balance=3.0, amount=1.0, oldbalanceOrg=1.0)
    )
    assert gate.signature_share == pytest.approx(5 / 9)
    assert (gate.flag_rank, gate.top_feature, gate.passed) == (2, "hour_sin", True)


def test_the_gate_fails_on_exactly_half() -> None:
    # 2 + 0.5 + 0.5 = 3 of 6: half is not a majority.
    gate = signature_gate(
        _means(hour_cos=3.0, amount_equals_origin_balance=2.0, amount=0.5, oldbalanceOrg=0.5)
    )
    assert gate.signature_share == pytest.approx(0.5)
    assert gate.flag_rank == 2
    assert not gate.passed


def test_the_gate_fails_when_the_flag_ranks_fourth_despite_a_majority() -> None:
    # Signature: 3 + 3 + 0.1 = 6.1 of 9.3 = 0.656, but type_TRANSFER pushes the flag to 4th.
    gate = signature_gate(
        _means(type_TRANSFER=3.2, amount=3.0, oldbalanceOrg=3.0, amount_equals_origin_balance=0.1)
    )
    assert gate.signature_share == pytest.approx(6.1 / 9.3)
    assert gate.flag_rank == 4
    assert not gate.passed


def test_the_gate_fails_when_the_hour_dominates() -> None:
    gate = signature_gate(
        _means(hour_sin=10.0, amount_equals_origin_balance=1.0, amount=1.0, oldbalanceOrg=1.0)
    )
    assert gate.signature_share == pytest.approx(3 / 13)
    assert not gate.passed


def test_a_flag_tied_for_third_ranks_third() -> None:
    gate = signature_gate(
        _means(amount=5.0, oldbalanceOrg=4.0, amount_equals_origin_balance=1.0, hour_sin=1.0)
    )
    assert gate.flag_rank == 3
    assert gate.passed


def test_the_gate_refuses_a_table_without_the_signature_features() -> None:
    with pytest.raises(ValueError, match="Missing signature features"):
        signature_gate(_means(amount=1.0, hour_sin=1.0))
    with pytest.raises(ValueError, match="zero"):
        signature_gate(_means(**dict.fromkeys(SIGNATURE_FEATURES, 0.0)))


# --- Permutation importance -------------------------------------------------------------------


class _ScoreIsColumnA:
    """A stub model whose positive-class probability is the column ``a``."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        score = features["a"].to_numpy(dtype=float)
        return np.column_stack([1 - score, score])


def test_decision_metric_by_hand() -> None:
    labels = np.array([1, 1, 0, 0])
    assert decision_metric(labels, np.array([0.9, 0.8, 0.1, 0.2])) == 1.0
    # 0.9 flags one fraud (precision 1, recall 0.5); 0.8 adds a legitimate row (precision 0.5).
    assert decision_metric(labels, np.array([0.1, 0.9, 0.8, 0.2])) == 0.5
    # No threshold reaches 0.99: the metric is 0.
    assert decision_metric(np.array([0, 1]), np.array([1.0, 0.0])) == 0.0


def test_permutation_importance_by_hand() -> None:
    # Two rows: a permutation of "a" either keeps the order (the metric stays 1) or swaps it
    # (the legitimate row ranks first, no threshold reaches 0.99, the metric is 0).
    features = pd.DataFrame({"a": [1.0, 0.0], "b": [5.0, 6.0]})
    before = features.copy()
    seeds = derived_seeds(8)
    result = permutation_importance(_ScoreIsColumnA(), features, np.array([1, 0]), seeds)
    assert result.baseline == 1.0
    swapped = [np.random.default_rng(seed).permutation(2)[0] == 1 for seed in seeds]
    assert result.drops.loc["a"].tolist() == [float(s) for s in swapped]
    assert 0 < sum(swapped) < len(seeds), "the seeds must cover both outcomes"
    assert result.drops.loc["b"].tolist() == [0.0] * len(seeds)
    pd.testing.assert_frame_equal(features, before)


def test_permutation_summary_by_hand() -> None:
    drops = pd.DataFrame({1: [0.0, 0.1], 2: [0.0, 0.3]}, index=["y", "x"])
    summary = PermutationImportance(baseline=1.0, drops=drops).summary()
    assert list(summary.index) == ["x", "y"]
    # x: mean 0.2; std sqrt(((0.1 - 0.2)^2 + (0.3 - 0.2)^2) / 1) = sqrt(0.02).
    assert summary.loc["x"].tolist() == pytest.approx([0.2, np.sqrt(0.02), 0.1, 0.3])
    assert summary.loc["y"].tolist() == [0.0, 0.0, 0.0, 0.0]


# --- The model's importance and the orderings -------------------------------------------------


def test_model_importance_reads_the_booster_by_feature_name(fitted: Pipeline) -> None:
    table = model_importance(fitted)
    booster = fitted[-1].booster_
    assert list(table.index) == list(fitted[:-1].get_feature_names_out())
    assert table["split"].tolist() == booster.feature_importance("split").tolist()
    assert table["gain"].tolist() == booster.feature_importance("gain").tolist()


def test_trees_splitting_on_agrees_with_the_split_count(fitted: Pipeline) -> None:
    importance = model_importance(fitted)
    used = importance.index[importance["split"] > 0]
    assert len(used) > 0
    for feature in used:
        splits = trees_splitting_on(fitted, feature)
        # Two independent readings of the booster: the dump and its own counter.
        assert len(splits) == importance.loc[feature, "split"], feature
        assert (splits["left_count"] + splits["right_count"] == splits["count"]).all()
    unused = importance.index[importance["split"] == 0]
    assert all(trees_splitting_on(fitted, feature).empty for feature in unused)


def _money_branch(pipeline: Pipeline) -> Pipeline:
    columns = pipeline[:-1].named_steps["preprocess"].named_steps["columns"]
    return columns.named_transformers_[MONEY_BRANCH]


def test_raw_split_thresholds_map_back_to_the_booster_thresholds(fitted: Pipeline) -> None:
    raw = raw_split_thresholds(fitted, "amount")
    booster = np.sort(trees_splitting_on(fitted, "amount")["threshold"].to_numpy(dtype=float))
    # Precondition: the synthetic model splits on amount, or there would be nothing to map.
    assert len(raw) == len(booster) > 0
    assert (np.diff(raw) >= 0).all()
    # Round trip: the money branch sends each raw value back to its booster threshold.
    money = _money_branch(fitted)
    frame = pd.DataFrame(0.0, index=range(len(raw)), columns=list(money.feature_names_in_))
    frame["amount"] = raw
    np.testing.assert_allclose(money.transform(frame)["amount"], booster, rtol=0, atol=1e-9)


def test_above_the_largest_raw_threshold_the_output_does_not_depend_on_the_column(
    fitted: Pipeline,
) -> None:
    largest = raw_split_thresholds(fitted, "amount")[-1]
    rows = synthetic_frame().drop(columns=[TARGET_COLUMN]).head(20)
    # The balance stays away from every amount tried, so the derived flag stays off.
    rows = rows.assign(oldbalanceOrg=largest * 100)
    grid = np.array([largest * 1.01, largest * 2, largest * 10])
    curves = ice_curves(fitted, rows, "amount", grid)
    assert (curves.nunique(axis=1) == 1).all()
    # Control: across the largest threshold the output does move for some row.
    across = ice_curves(fitted, rows, "amount", np.array([largest * 0.5, largest * 1.01]))
    assert (across.nunique(axis=1) > 1).any()


def test_raw_split_thresholds_refuse_a_column_outside_the_money_branch(fitted: Pipeline) -> None:
    with pytest.raises(ValueError, match="not a money column"):
        raw_split_thresholds(fitted, "hour_sin")


def test_raw_split_thresholds_refuse_a_money_branch_it_cannot_invert(fitted: Pipeline) -> None:
    changed = copy.deepcopy(fitted)
    _money_branch(changed).named_steps["log1p"].func = np.sqrt
    with pytest.raises(ValueError, match="log1p_non_negative"):
        raw_split_thresholds(changed, "amount")


def test_compare_rankings_by_hand() -> None:
    table = compare_rankings(
        {"m1": pd.Series({"a": 3.0, "b": 1.0, "c": 2.0}), "m2": pd.Series({"a": 1, "b": 5, "c": 1})}
    )
    assert list(table.index) == ["a", "c", "b"]
    assert table["m1_rank"].tolist() == [1, 2, 3]
    # m2: b is first; a and c tie for second and both take rank 2.
    assert table["m2_rank"].tolist() == [2, 2, 1]
    assert table["m2"].tolist() == [1.0, 1.0, 5.0]


# --- ICE and PDP ------------------------------------------------------------------------------


def test_raw_grid_by_hand() -> None:
    assert raw_grid(pd.Series(np.arange(101.0)), points=3, quantiles=(0.0, 1.0)).tolist() == [
        0.0,
        50.0,
        100.0,
    ]
    # Quantiles 0, 0.25, 0.5, 0.75 and 1 of [0, 0, 0, 0, 10] are 0, 0, 0, 0, 10: merged.
    grid = raw_grid(pd.Series([0.0, 0.0, 0.0, 0.0, 10.0]), points=5, quantiles=(0.0, 1.0))
    assert grid.tolist() == [0.0, 10.0]


def test_outside_grid_by_hand() -> None:
    values = pd.Series([-1.0, 0.0, 5.0, 10.0, 11.0])
    assert outside_grid(values, np.array([0.0, 4.0, 10.0])).tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_log_grid_by_hand() -> None:
    # Geometric from the smallest positive to the largest, both exact, and 0 added.
    assert log_grid(pd.Series([0.0, 1.0, 10.0, 100.0]), points=3).tolist() == [
        0.0,
        1.0,
        10.0,
        100.0,
    ]
    # Without a zero in the column, no zero in the grid.
    grid = log_grid(pd.Series([2.0, 8.0, 5.0]), points=3)
    np.testing.assert_allclose(grid, [2.0, 4.0, 8.0])
    assert (grid[0], grid[-1]) == (2.0, 8.0)


def test_log_grid_refuses_what_has_no_log_scale() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        log_grid(pd.Series([-1.0, 5.0]))
    with pytest.raises(ValueError, match="positive"):
        log_grid(pd.Series([0.0, 0.0]))


def test_grid_side_by_hand() -> None:
    values = pd.Series([-1.0, 0.0, 5.0, 10.0, 11.0], index=[3, 4, 5, 6, 7])
    grid = np.array([0.0, 4.0, 10.0])
    sides = grid_side(values, grid)
    assert sides.tolist() == [BELOW_GRID, INSIDE_GRID, INSIDE_GRID, INSIDE_GRID, ABOVE_GRID]
    assert sides.index.equals(values.index)
    assert list(sides.cat.categories) == list(GRID_SIDES)
    # Both sides together are exactly what outside_grid counts.
    assert (sides != INSIDE_GRID).tolist() == outside_grid(values, grid).tolist()
    # A count of rows outside is not a count of rows above: here one is below, one above.
    assert sides.value_counts()[[BELOW_GRID, ABOVE_GRID]].tolist() == [1, 1]


def test_ice_curves_match_the_artifact_on_each_modified_row(fitted: Pipeline) -> None:
    rows = synthetic_frame().drop(columns=[TARGET_COLUMN]).head(30)
    grid = np.array([5.0, 300.0, 2_000.0])
    curves = ice_curves(fitted, rows, "amount", grid)
    assert curves.shape == (30, 3)
    assert curves.index.equals(rows.index)
    for value in grid:
        modified = rows.assign(amount=value)
        probability = fitted.predict_proba(modified)[:, 1]
        np.testing.assert_allclose(curves[value], np.log(probability / (1 - probability)))
    pd.testing.assert_frame_equal(rows, synthetic_frame().drop(columns=[TARGET_COLUMN]).head(30))


def test_ice_recomputes_the_derived_flag(fitted: Pipeline) -> None:
    # A row whose amount differs from its balance: setting the amount to the balance turns the
    # flag on, and that row of the sweep must equal the row rebuilt with the flag on.
    rows = synthetic_frame().drop(columns=[TARGET_COLUMN])
    row = rows.loc[[rows.index[(rows["amount"] != rows["oldbalanceOrg"])][0]]]
    balance = float(row["oldbalanceOrg"].iloc[0])
    swept = ice_curves(fitted, row, "amount", np.array([balance]))
    rebuilt = model_features(fitted, row.assign(amount=balance))
    assert bool(rebuilt[FLAG].iloc[0])
    assert swept.iloc[0, 0] == pytest.approx(fitted[-1].predict(rebuilt, raw_score=True)[0])


def test_rule_groups_by_hand() -> None:
    labels = pd.Series([1, 1, 0, 0], index=[7, 8, 9, 10])
    flags = pd.Series([True, False, True, False], index=[7, 8, 9, 10])
    assert rule_groups(labels, flags).tolist() == [RULE_FRAUD, OTHER_FRAUD, LEGITIMATE, LEGITIMATE]


def test_the_figures_draw(fitted: Pipeline) -> None:
    frame = synthetic_frame()
    rows = frame.drop(columns=[TARGET_COLUMN])
    beeswarm = plot_beeswarm(shap_values(fitted, rows))
    groups = rule_groups(frame[TARGET_COLUMN], pd.Series(False, index=frame.index))
    curves = {name: ice_curves(fitted, rows, name, np.array([1.0, 50.0])) for name in ("amount",)}
    ice = plot_ice(curves, groups, curves_per_group=5)
    assert len(ice.axes) == 1
    assert beeswarm.axes
    plt.close("all")
