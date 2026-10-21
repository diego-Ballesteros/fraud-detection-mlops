"""Global interpretation of the artifact: SHAP, and the gate that validates the explainer.

Everything here explains the loaded artifact, ``models.registry.load_pipeline``. Nothing is
refitted: a model rebuilt for the explanation would explain a different object.

**The explainer** is ``shap.TreeExplainer`` with ``feature_perturbation="tree_path_dependent"``
(ADR 0011). ``amount_equals_origin_balance`` is a deterministic function of ``amount`` and
``oldbalanceOrg``, so the three are perfectly dependent. The interventional mode samples
marginal values and would build rows where the flag claims the equality and the values
contradict it; the model never saw such rows. The path-dependent mode follows the training
data's own paths through each tree, so it stays on the data's manifold. Its values are in the
model's raw output, the log-odds, and they add up exactly to it: ``shap_values`` checks that
on every row it returns.

**The sample** is every positive of the reference period plus a sample of negatives drawn
with ``settings.RANDOM_STATE``, the one of notebook 01's chart 6. A uniform sample at a
prevalence of 0.077% would hold too few positives for SHAP to say anything about them.

**The gate** comes before any figure. The simulator's signature is a known answer (ADR
0007): if SHAP does not find it, the explainer is broken, not the model. Because the three
features that encode the signature are perfectly dependent, the path-dependent mode splits
their credit according to each tree's structure; the gate therefore asks that the three
together hold the majority of the mean |SHAP|, and that the flag be among the first three,
not that the flag come first (ADR 0011).

**Permutation importance** is measured on the decision metric, recall at precision >= 0.99,
over every row of the reference period, at the level of the model features: the artifact's
preprocessing runs once and each of its output columns is permuted in turn. The artifact was
fitted on those same rows, so the measurement is in-sample. Permuting one column alone is
interventional: a permuted flag, or a permuted ``amount`` or ``oldbalanceOrg``, builds rows
where the flag and the values disagree, and a permuted one-hot column of ``type`` builds rows
with two types or none. That is the objection that excluded interventional SHAP, and it is
declared, not avoided (ADR 0011).

**PDP and ICE** vary raw columns and pass every modified row through the whole artifact, so
the derivation recomputes the flag at each grid point and no row contradicts it (ADR 0011).
The flag has no curve of its own: it is not an input. Values are the model's raw output, the
log-odds, the scale of the SHAP values.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from fraud_detection.config import settings
from fraud_detection.data.profiling import sample_with_all_positives
from fraud_detection.features.builder import SIGNATURE_FEATURE
from fraud_detection.models.evaluation import DECISION_PRECISION, recall_at_precision

EXPLANATION_NEGATIVE_ROWS: Final[int] = 20_000
"""Negatives in the explanation sample: the same 20,000 of notebook 01's chart 6."""

SIGNATURE_FEATURES: Final[tuple[str, ...]] = (SIGNATURE_FEATURE, "amount", "oldbalanceOrg")
"""The three model features that encode the signature: the flag and the two columns it is
computed from."""

GATE_MAJORITY: Final[float] = 0.5
"""The signature features must hold strictly more than this share of the total mean |SHAP|."""

GATE_FLAG_MAX_RANK: Final[int] = 3
"""The flag must rank at least this high by mean |SHAP|."""

ADDITIVITY_TOLERANCE: Final[float] = 1e-6
"""Largest accepted gap, in log-odds, between a row's SHAP sum and the model's raw output."""


def explanation_sample(
    reference: pd.DataFrame,
    negative_rows: int = EXPLANATION_NEGATIVE_ROWS,
    random_state: int = settings.RANDOM_STATE,
) -> pd.DataFrame:
    """Every positive row of the reference period and a uniform sample of its negatives.

    Args:
        reference: Reference rows, with the label; their index identifies each row.
        negative_rows: How many negatives to draw, without replacement.
        random_state: Seed of the draw; by default the project's.

    Returns:
        The positives, then the sampled negatives, with their original index.
    """
    return sample_with_all_positives(reference, negative_rows, random_state=random_state)


def model_features(pipeline: Pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    """What the artifact's model receives: its own preprocessing step applied to raw rows.

    Args:
        pipeline: The fitted artifact.
        rows: Raw rows with at least the columns the artifact reads.

    Returns:
        One column per name of ``get_feature_names_out``, on the index of ``rows``.

    Raises:
        ValueError: If the transformed columns are not the preprocessor's declared names.
    """
    preprocess = pipeline[:-1]
    features = preprocess.transform(rows)
    names = list(preprocess.get_feature_names_out())
    if list(features.columns) != names:
        raise ValueError(f"Transformed columns {list(features.columns)} differ from {names}")
    return features


@dataclass(frozen=True)
class ShapResult:
    """SHAP values of some rows, in the model's raw output (log-odds).

    Attributes:
        values: One row per explained row and one column per model feature.
        base_value: The explainer's expected value; each row's values add up to its raw
            output minus this.
        features: The model features of the explained rows, as the model received them.
        raw_output: The model's raw output (log-odds) for each row.
        max_additivity_gap: Largest gap between ``base_value + values.sum(axis=1)`` and
            ``raw_output`` over the rows.
    """

    values: pd.DataFrame
    base_value: float
    features: pd.DataFrame
    raw_output: pd.Series
    max_additivity_gap: float


def tree_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """The path-dependent tree explainer of the artifact's model (ADR 0011).

    Args:
        pipeline: The fitted artifact.

    Returns:
        A ``shap.TreeExplainer`` with no background data.
    """
    return shap.TreeExplainer(pipeline[-1], feature_perturbation="tree_path_dependent")


def shap_values(pipeline: Pipeline, rows: pd.DataFrame) -> ShapResult:
    """Explain raw rows with the path-dependent tree explainer, and check additivity.

    Args:
        pipeline: The fitted artifact.
        rows: Raw rows to explain.

    Returns:
        The SHAP values with the base value, the model features and the raw output.

    Raises:
        ValueError: If some row's SHAP values do not add up to its raw output within
            ``ADDITIVITY_TOLERANCE``: the explainer is then not explaining this model.
    """
    features = model_features(pipeline, rows)
    explainer = tree_explainer(pipeline)
    values = np.asarray(explainer.shap_values(features, check_additivity=False), dtype=float)
    if values.shape != features.shape:
        raise ValueError(f"SHAP values {values.shape} do not match features {features.shape}")
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    raw = np.asarray(pipeline[-1].predict(features, raw_score=True), dtype=float)
    gap = float(np.max(np.abs(base_value + values.sum(axis=1) - raw))) if len(raw) else 0.0
    if gap > ADDITIVITY_TOLERANCE:
        raise ValueError(f"SHAP values miss the raw output by up to {gap:.3g} log-odds")
    return ShapResult(
        values=pd.DataFrame(values, index=features.index, columns=features.columns),
        base_value=base_value,
        features=features,
        raw_output=pd.Series(raw, index=features.index, name="raw_output"),
        max_additivity_gap=gap,
    )


def mean_abs_shap(values: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, largest first.

    Args:
        values: SHAP values, one column per feature.

    Returns:
        The mean absolute value of each column, sorted in descending order; ties keep the
        column order.
    """
    return values.abs().mean().sort_values(ascending=False, kind="stable")


@dataclass(frozen=True)
class GateResult:
    """Whether the explainer found the simulator's signature.

    Attributes:
        signature_share: Share of the total mean |SHAP| held by ``SIGNATURE_FEATURES``.
        flag_rank: Rank of the flag by mean |SHAP|, 1 for the largest; tied values share
            the best rank.
        top_feature: The feature with the largest mean |SHAP|.
        passed: Whether the share is a strict majority and the flag ranks in the first
            ``GATE_FLAG_MAX_RANK``.
    """

    signature_share: float
    flag_rank: int
    top_feature: str
    passed: bool


def signature_gate(
    mean_abs: pd.Series,
    signature: tuple[str, ...] = SIGNATURE_FEATURES,
    flag: str = SIGNATURE_FEATURE,
) -> GateResult:
    """Apply the gate of ADR 0011 to a table of mean |SHAP|.

    Args:
        mean_abs: Mean |SHAP| per feature, as ``mean_abs_shap`` returns it.
        signature: The features that encode the signature.
        flag: The signature flag, which must be one of ``signature``.

    Returns:
        The share, the flag's rank and the verdict.

    Raises:
        ValueError: If a signature feature is missing, the flag is not among them, or every
            mean |SHAP| is zero.
    """
    missing = [name for name in signature if name not in mean_abs.index]
    if missing or flag not in signature:
        raise ValueError(f"Missing signature features {missing} or flag {flag!r} not in them")
    total = float(mean_abs.sum())
    if total <= 0:
        raise ValueError("Every mean |SHAP| is zero: the explainer attributes nothing")
    share = float(mean_abs[list(signature)].sum()) / total
    rank = int(mean_abs.rank(ascending=False, method="min")[flag])
    return GateResult(
        signature_share=share,
        flag_rank=rank,
        top_feature=str(mean_abs.idxmax()),
        passed=share > GATE_MAJORITY and rank <= GATE_FLAG_MAX_RANK,
    )


# --- Permutation importance on the decision metric --------------------------------------------

PERMUTATION_REPEATS: Final[int] = 10
"""Permutations per feature; the seeds are ``evaluation.derived_seeds(PERMUTATION_REPEATS)``."""


def decision_metric(labels: np.ndarray, scores: np.ndarray) -> float:
    """Recall at precision >= 0.99, 0 when no threshold reaches the floor (ADR 0009).

    Args:
        labels: 0/1 labels.
        scores: Scores, higher meaning more likely positive.

    Returns:
        The recall of ``evaluation.recall_at_precision`` at ``DECISION_PRECISION``.
    """
    point = recall_at_precision(labels, scores, DECISION_PRECISION)
    return point.recall if point is not None else 0.0


@dataclass(frozen=True)
class PermutationImportance:
    """How much the decision metric drops when one model feature is permuted.

    Attributes:
        baseline: The metric with no column permuted.
        drops: One row per feature and one column per seed: ``baseline`` minus the metric
            with that feature permuted with that seed.
    """

    baseline: float
    drops: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Mean, sample standard deviation (ddof = 1), minimum and maximum drop per feature.

        Returns:
            One row per feature, largest mean drop first; ties keep the feature order.
        """
        table = pd.DataFrame(
            {
                "mean": self.drops.mean(axis=1),
                "std": self.drops.std(axis=1, ddof=1),
                "min": self.drops.min(axis=1),
                "max": self.drops.max(axis=1),
            }
        )
        return table.sort_values("mean", ascending=False, kind="stable")


def permutation_importance(
    model: Any,
    features: pd.DataFrame,
    labels: np.ndarray,
    seeds: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float] = decision_metric,
) -> PermutationImportance:
    """Permute each model feature in turn and measure the drop of ``metric``.

    The permutation of seed ``s`` is ``np.random.default_rng(s).permutation(len(features))``,
    the same row order for every feature. ``features`` is not modified.

    Args:
        model: A fitted classifier with ``predict_proba``: the artifact's last step.
        features: Its input, as ``model_features`` builds it.
        labels: 0/1 labels of the rows.
        seeds: One permutation per seed.
        metric: Score to measure, from labels and the positive-class probability.

    Returns:
        The unpermuted metric and the drop of every feature under every seed.
    """
    labels = np.asarray(labels)
    baseline = metric(labels, model.predict_proba(features)[:, 1])
    work = features.copy()
    drops = pd.DataFrame(index=features.columns, columns=list(seeds), dtype=float)
    for column in features.columns:
        original = features[column].to_numpy()
        for seed in seeds:
            order = np.random.default_rng(seed).permutation(len(features))
            work[column] = original[order]
            drops.loc[column, seed] = baseline - metric(labels, model.predict_proba(work)[:, 1])
        work[column] = original
    return PermutationImportance(baseline=baseline, drops=drops)


# --- The model's own importance, and the orderings side by side -------------------------------


def model_importance(pipeline: Pipeline) -> pd.DataFrame:
    """LightGBM's own importance of each model feature: split count and total gain.

    Args:
        pipeline: The fitted artifact; its last step is a LightGBM classifier.

    Returns:
        One row per model feature, in the preprocessor's order, with ``split`` and ``gain``.

    Raises:
        ValueError: If the booster's feature names are not the preprocessor's.
    """
    booster = pipeline[-1].booster_
    names = list(pipeline[:-1].get_feature_names_out())
    if list(booster.feature_name()) != names:
        raise ValueError(f"Booster features {booster.feature_name()} differ from {names}")
    return pd.DataFrame(
        {
            "split": booster.feature_importance(importance_type="split").astype(float),
            "gain": booster.feature_importance(importance_type="gain"),
        },
        index=names,
    )


def trees_splitting_on(pipeline: Pipeline, feature: str) -> pd.DataFrame:
    """Every split node of the artifact's trees on one model feature, with its two children.

    Args:
        pipeline: The fitted artifact; its last step is a LightGBM classifier.
        feature: A model feature name.

    Returns:
        One row per split node: ``tree_index``, ``node_depth``, ``split_gain``,
        ``threshold``, the rows that reached it in training (``count``), and each child's
        output (``left_value``, ``right_value``) and training rows (``left_count``,
        ``right_count``). LightGBM sends a row left when its value is at most ``threshold``.
    """
    nodes = pipeline[-1].booster_.trees_to_dataframe()
    by_id = nodes.set_index("node_index")
    splits = nodes[nodes["split_feature"] == feature]
    return pd.DataFrame(
        {
            "tree_index": splits["tree_index"].to_numpy(),
            "node_depth": splits["node_depth"].to_numpy(),
            "split_gain": splits["split_gain"].to_numpy(),
            "threshold": splits["threshold"].to_numpy(),
            "count": splits["count"].to_numpy(),
            "left_value": by_id.loc[splits["left_child"], "value"].to_numpy(),
            "left_count": by_id.loc[splits["left_child"], "count"].to_numpy(),
            "right_value": by_id.loc[splits["right_child"], "value"].to_numpy(),
            "right_count": by_id.loc[splits["right_child"], "count"].to_numpy(),
        }
    )


def compare_rankings(orderings: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Put several importance measures side by side, each with its value and its rank.

    Ranks are descending, 1 for the largest; tied values share the best rank.

    Args:
        orderings: Measure name to one importance value per feature.

    Returns:
        Columns ``<measure>`` and ``<measure>_rank`` for each measure, rows ordered by the
        first measure's rank.
    """
    columns: dict[str, pd.Series] = {}
    for name, values in orderings.items():
        columns[name] = values.astype(float)
        columns[f"{name}_rank"] = values.rank(ascending=False, method="min")
    table = pd.DataFrame(columns)
    first = next(iter(orderings))
    return table.sort_values(f"{first}_rank", kind="stable")


# --- ICE and PDP on raw columns, through the whole artifact -----------------------------------

ICE_GRID_POINTS: Final[int] = 50
"""Quantiles of the reference distribution used as grid points; equal ones are merged."""

ICE_GRID_QUANTILES: Final[tuple[float, float]] = (0.01, 0.99)
"""Lowest and highest quantile of the grid: the extremes have too few rows to lean on."""


def raw_grid(
    values: pd.Series,
    points: int = ICE_GRID_POINTS,
    quantiles: tuple[float, float] = ICE_GRID_QUANTILES,
) -> np.ndarray:
    """Grid points for one raw column: evenly spaced quantiles of its distribution.

    Args:
        values: The column over the rows that define its distribution.
        points: How many quantiles to take.
        quantiles: Lowest and highest quantile.

    Returns:
        The distinct quantile values, ascending.
    """
    return np.unique(np.quantile(values.to_numpy(dtype=float), np.linspace(*quantiles, points)))


def outside_grid(values: pd.Series, grid: np.ndarray) -> pd.Series:
    """Whether each row's own value lies outside the grid's range.

    An ICE curve shows the model at the grid's values only: a row whose own value is
    outside the range is never evaluated at, or near, what it actually holds.

    Args:
        values: The rows' own values of the varied column.
        grid: The grid, ascending.

    Returns:
        True where the value is below the first grid point or above the last.
    """
    return (values < grid[0]) | (values > grid[-1])


def ice_curves(
    pipeline: Pipeline, rows: pd.DataFrame, column: str, grid: np.ndarray
) -> pd.DataFrame:
    """Each row's raw model output as one raw column sweeps the grid, all else fixed.

    Every modified row passes through the whole artifact, preprocessing included, so any
    feature derived from ``column`` is recomputed at each grid point.

    Args:
        pipeline: The fitted artifact.
        rows: Raw rows with at least the columns the artifact reads.
        column: The raw column to vary.
        grid: The values it takes.

    Returns:
        One row per input row and one column per grid value: the log-odds.
    """
    model = pipeline[-1]
    curves = np.empty((len(rows), len(grid)))
    modified = rows.copy()
    for position, value in enumerate(grid):
        modified[column] = np.full(len(rows), value, dtype=rows[column].dtype)
        features = model_features(pipeline, modified)
        curves[:, position] = model.predict(features, raw_score=True)
    return pd.DataFrame(curves, index=rows.index, columns=pd.Index(grid, name=column))


# --- Figures ----------------------------------------------------------------------------------

RULE_FRAUD: Final[str] = "Fraude cubierto por la regla"
OTHER_FRAUD: Final[str] = "Fraude fuera de la regla"
LEGITIMATE: Final[str] = "Legítima"

GROUP_COLORS: Final[Mapping[str, str]] = {
    RULE_FRAUD: "#c44e52",
    OTHER_FRAUD: "#d4a017",
    LEGITIMATE: "#4c72b0",
}
"""Fraud and legitimate keep notebook 01's colors; the third passed the palette validator."""


def rule_groups(labels: pd.Series, rule_flags: pd.Series) -> pd.Series:
    """Name each row's group: fraud the signature rule covers, fraud it misses, legitimate.

    Args:
        labels: 0/1 labels.
        rule_flags: Whether ``SignatureRule`` flags the row, on the same index.

    Returns:
        One group name per row.
    """
    covered = np.where(np.asarray(rule_flags, dtype=bool), RULE_FRAUD, OTHER_FRAUD)
    groups = np.where(np.asarray(labels) == 1, covered, LEGITIMATE)
    return pd.Series(groups, index=labels.index, name="group")


def plot_beeswarm(result: ShapResult) -> Figure:
    """SHAP summary of the explanation sample: one dot per row and feature.

    Args:
        result: SHAP values of the sample.

    Returns:
        The figure.
    """
    explanation = shap.Explanation(
        values=result.values.to_numpy(),
        base_values=np.full(len(result.values), result.base_value),
        data=result.features.to_numpy(dtype=float),
        feature_names=list(result.values.columns),
    )
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=len(result.values.columns), show=False)
    figure = plt.gcf()
    figure.axes[0].set_xlabel("Valor SHAP (log-odds; positivo empuja hacia fraude)")
    figure.suptitle("Resumen SHAP sobre la muestra declarada")
    return figure


def plot_ice(
    curves: Mapping[str, pd.DataFrame],
    groups: pd.Series,
    curves_per_group: int,
    random_state: int = settings.RANDOM_STATE,
) -> Figure:
    """ICE curves by group with each group's PDP, and the PDP of the whole sample.

    Args:
        curves: Raw column to its ICE curves, as ``ice_curves`` returns them.
        groups: Group name of each row, as ``rule_groups`` returns it.
        curves_per_group: How many individual curves of each group to draw.
        random_state: Seed of the choice of drawn curves.

    Returns:
        The figure, one panel per column with a shared vertical axis.
    """
    figure, axes = plt.subplots(
        1, len(curves), figsize=(5.2 * len(curves), 4.6), sharey=True, squeeze=False
    )
    for ax, (column, ice) in zip(axes[0], curves.items(), strict=True):
        grid = ice.columns.to_numpy(dtype=float)
        for group, color in GROUP_COLORS.items():
            members = ice.index[groups.reindex(ice.index) == group]
            if not len(members):
                continue
            drawn = pd.Series(members).sample(
                n=min(curves_per_group, len(members)), random_state=random_state
            )
            ax.plot(grid, ice.loc[drawn].to_numpy().T, color=color, lw=0.6, alpha=0.18)
            size = f"{len(members):,}".replace(",", ".")
            ax.plot(
                grid,
                ice.loc[members].mean(),
                color=color,
                lw=2.2,
                label=f"PDP · {group} (n = {size})",
            )
        ax.plot(grid, ice.mean(), color="black", lw=2, ls="--", label="PDP · toda la muestra")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel(f"{column} (escala logarítmica simétrica)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("Salida del modelo (log-odds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("ICE y PDP por columna cruda, a través del artefacto completo")
    figure.tight_layout(rect=(0, 0.12, 1, 0.97))
    return figure
