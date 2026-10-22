"""Feature drift between the reference period and production: PSI decides, KS goes beside.

**What is compared** are the 11 model features, as the artifact's own preprocessing outputs
them (``explain.global_explain.model_features``): what the model receives, not a
reimplementation. ``log1p`` and ``RobustScaler`` are strictly increasing on each money
column, so the reference-quantile bins and the KS distance give the same value on the
transformed column as on the raw one (a test proves it); the transformed scale costs nothing.

**PSI is the measure.** Its cuts, ``PSI_MODERATE`` (0.1) and ``PSI_SIGNIFICANT`` (0.25), are
a **convention**, declared before measuring and not derived from these data. They mean the
same at any sample size only when the sample is large: drawn from the reference itself, 80
rows already reach a PSI of 0.68 by sampling noise alone, and 4,474 rows stay at 0.0062
(``null_psi``). And they have a measured blind spot: a rare category is nearly invisible to
PSI, which weighs absolute differences of share. In production the signature flag goes from
0.076% to 0.631% of the rows, 8.3 times as many, and its PSI is 0.012, "stable"; a prevalence
that matters is measured directly, not through PSI. **KS
is reported as a distance only**, the largest gap between the two empirical distribution
functions. No p-value is computed anywhere in this module: with hundreds of thousands of
rows on each side any difference is significant, and that says nothing about its size.

**Bins come from the reference period only.** ``FeatureProfile.from_reference`` receives the
reference values and nothing else; every production sample, whole or by window, is then
binned on those edges. No production row can move a bin. Evidently's numerical PSI does not
work this way: for a column with more than 20 distinct values it uses equal-width bins over
reference and current together, so its edges depend on production (read in evidently 0.7.23,
``legacy/calculations/stattests/utils.py``).

Two kinds of bins, chosen by the artifact's own partition of its columns:

- **Continuous**, the money branch: the deciles of the reference (``method="inverted_cdf"``,
  so every edge is a reference value), duplicate edges merged. Bins are right-closed and open
  at both ends, so a production value outside the reference range lands in an end bin.
- **Discrete**, every other feature (the flag, the five one-hot columns of ``type``,
  ``hour_sin`` and ``hour_cos``): one bin per distinct reference value, plus one for any value
  the reference never had.

A bin empty on both sides is dropped: it carries no information. A share of zero on one side
is floored by Evidently's rule (``_floor_zero_shares``), so that the two instruments agree
wherever their bins agree.

**What drift means per feature.** On the money features, the amounts and balances moved. On
the one-hot columns, the mix of operations moved. The flag takes the whole balance, and in
reference only frauds do that (``docs/DATA_DICTIONARY.md``, section 4.a), so its drift tracks
the prevalence and the fraud agent's behavior, not a covariate. ``hour_sin`` and ``hour_cos``
have the same support in both periods by construction: every step falls on one of the 24
hours. Drift there can never be a value the model did not see; it can only be a change in how
the volume of transactions spreads over the hours of the day.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from fraud_detection.features.builder import HOURS_PER_DAY

PSI_MODERATE: Final[float] = 0.1
"""PSI at or above this is moderate drift. A convention, not derived from the data."""

PSI_SIGNIFICANT: Final[float] = 0.25
"""PSI at or above this is significant drift. A convention, not derived from the data."""

STABLE: Final[str] = "stable"
MODERATE: Final[str] = "moderate"
SIGNIFICANT: Final[str] = "significant"

DECILES: Final[tuple[float, ...]] = tuple(step / 10 for step in range(1, 10))
"""Inner quantiles whose reference values are the edges of a continuous feature's bins."""

_EVIDENTLY_FLOOR: Final[float] = 1e-4
_EVIDENTLY_FLOOR_DIVISOR: Final[float] = 1e6


def psi_level(psi: float) -> str:
    """Name the drift a PSI value describes.

    Args:
        psi: A PSI value.

    Returns:
        ``STABLE``, ``MODERATE`` or ``SIGNIFICANT``.
    """
    if psi >= PSI_SIGNIFICANT:
        return SIGNIFICANT
    if psi >= PSI_MODERATE:
        return MODERATE
    return STABLE


def _floor_zero_shares(shares: np.ndarray) -> np.ndarray:
    """Replace zero shares as Evidently does.

    The floor is 1e-4, or the smallest nonzero share divided by 1e6 when that share is not
    above 1e-4.
    """
    shares = np.asarray(shares, dtype=float).copy()
    smallest = shares[shares > 0].min()
    floor = (
        smallest / _EVIDENTLY_FLOOR_DIVISOR if smallest <= _EVIDENTLY_FLOOR else _EVIDENTLY_FLOOR
    )
    shares[shares == 0] = floor
    return shares


def _histogram(edges: np.ndarray, x: np.ndarray, discrete: bool) -> np.ndarray:
    """Rows per bin.

    Continuous: right-closed bins between the edges, open at both ends. Discrete: one bin per
    edge value, then one for the values that match no edge.
    """
    position = np.searchsorted(edges, x, side="left")
    if discrete:
        inside = position < len(edges)
        matched = np.zeros(len(x), dtype=bool)
        matched[inside] = edges[position[inside]] == x[inside]
        position = np.where(matched, position, len(edges))
    return np.bincount(position, minlength=len(edges) + 1)


def psi_from_counts(reference_counts: np.ndarray, production_counts: np.ndarray) -> float:
    """PSI between two histograms on the same bins.

    ``sum((p - r) * ln(p / r))`` over the bins, with ``r`` and ``p`` the reference and
    production shares, after dropping the bins empty on both sides and flooring zero shares.

    Args:
        reference_counts: Rows per bin in the reference.
        production_counts: Rows per bin in production, on the same bins.

    Returns:
        The PSI, 0 for identical shares.

    Raises:
        ValueError: If the histograms differ in length or either side is empty.
    """
    r = np.asarray(reference_counts, dtype=float)
    p = np.asarray(production_counts, dtype=float)
    if r.shape != p.shape or r.sum() == 0 or p.sum() == 0:
        raise ValueError("Both histograms need rows, on the same bins")
    kept = (r > 0) | (p > 0)
    r_shares = _floor_zero_shares(r[kept] / r.sum())
    p_shares = _floor_zero_shares(p[kept] / p.sum())
    return float(np.sum((p_shares - r_shares) * np.log(p_shares / r_shares)))


@dataclass(frozen=True)
class FeatureProfile:
    """What the reference period says about one feature: its bins and its distribution.

    Built by ``from_reference`` from reference values only.

    Attributes:
        name: The feature.
        discrete: Whether bins are the distinct reference values (else, reference deciles).
        edges: Continuous: the right-closed inner edges. Discrete: the distinct values.
        reference_counts: Reference rows per bin; the last bin of a discrete feature holds the
            values the reference never had, so it is 0.
        values: Distinct reference values, ascending, for the KS distance.
        cumulative: Share of reference rows at or below each of ``values``.
    """

    name: str
    discrete: bool
    edges: np.ndarray
    reference_counts: np.ndarray
    values: np.ndarray
    cumulative: np.ndarray

    @classmethod
    def from_reference(cls, name: str, reference: np.ndarray, discrete: bool) -> "FeatureProfile":
        """Profile a feature from its reference values, and from nothing else.

        Args:
            name: The feature.
            reference: Its values over the reference period.
            discrete: Whether to bin by distinct value instead of by decile.

        Returns:
            The profile.

        Raises:
            ValueError: If the reference is empty or holds a NaN.
        """
        x = np.asarray(reference, dtype=float)
        if not len(x) or np.isnan(x).any():
            raise ValueError(f"{name}: the reference needs values, and no NaN")
        values, counts = np.unique(x, return_counts=True)
        edges = values if discrete else np.unique(np.quantile(x, DECILES, method="inverted_cdf"))
        return cls(
            name=name,
            discrete=discrete,
            edges=edges,
            reference_counts=_histogram(edges, x, discrete),
            values=values,
            cumulative=np.cumsum(counts) / len(x),
        )

    @property
    def bins(self) -> int:
        """Number of bins."""
        return len(self.edges) + 1

    def counts(self, sample: np.ndarray) -> np.ndarray:
        """Rows of a sample per bin of this profile.

        Args:
            sample: Values of the feature.

        Returns:
            One count per bin.
        """
        return _histogram(self.edges, np.asarray(sample, dtype=float), self.discrete)

    def psi(self, sample: np.ndarray) -> float:
        """PSI of a sample against the reference, on the reference's bins."""
        return psi_from_counts(self.reference_counts, self.counts(sample))

    def ks(self, sample: np.ndarray) -> float:
        """KS distance: the largest gap between the reference and the sample's empirical CDFs.

        Both CDFs are right-continuous step functions that only jump at observed values, so
        the largest gap is reached at one of them; it is checked at every distinct value of
        either side.

        Args:
            sample: Values of the feature.

        Returns:
            The distance, between 0 and 1.

        Raises:
            ValueError: If the sample is empty.
        """
        x = np.sort(np.asarray(sample, dtype=float))
        if not len(x):
            raise ValueError(f"{self.name}: the KS distance needs a non-empty sample")
        gaps = []
        for points in (self.values, np.unique(x)):
            below = np.searchsorted(self.values, points, side="right")
            reference_cdf = np.where(below > 0, self.cumulative[np.maximum(below - 1, 0)], 0.0)
            sample_cdf = np.searchsorted(x, points, side="right") / len(x)
            gaps.append(float(np.max(np.abs(reference_cdf - sample_cdf))))
        return max(gaps)


def reference_profiles(
    reference: pd.DataFrame, discrete: Sequence[str]
) -> dict[str, FeatureProfile]:
    """Profile every column of the reference features.

    Args:
        reference: The model features over the reference period.
        discrete: The columns binned by distinct value; the others are binned by decile.

    Returns:
        Feature name to profile, in column order.
    """
    return {
        name: FeatureProfile.from_reference(name, reference[name].to_numpy(), name in discrete)
        for name in reference.columns
    }


def drift_table(profiles: Mapping[str, FeatureProfile], sample: pd.DataFrame) -> pd.DataFrame:
    """PSI, its level and the KS distance of each feature, sample against reference.

    Args:
        profiles: From ``reference_profiles``.
        sample: The same features over the sample compared.

    Returns:
        One row per feature: ``kind``, ``bins``, ``psi``, ``level`` and ``ks``.
    """
    rows = []
    for name, profile in profiles.items():
        values = sample[name].to_numpy()
        psi = profile.psi(values)
        rows.append(
            {
                "feature": name,
                "kind": "discrete" if profile.discrete else "continuous",
                "bins": profile.bins,
                "psi": psi,
                "level": psi_level(psi),
                "ks": profile.ks(values),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def window_drift(
    profiles: Mapping[str, FeatureProfile], sample: pd.DataFrame, windows: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PSI and KS of each window against the whole reference, feature by feature.

    Args:
        profiles: From ``reference_profiles``.
        sample: The features over the production period.
        windows: The window number of each row of ``sample``.

    Returns:
        PSI and KS, each with one row per window and one column per feature.
    """
    numbers = np.asarray(windows)
    psi, ks = {}, {}
    for number in np.unique(numbers):
        rows = sample.loc[numbers == number]
        psi[int(number)] = {name: p.psi(rows[name].to_numpy()) for name, p in profiles.items()}
        ks[int(number)] = {name: p.ks(rows[name].to_numpy()) for name, p in profiles.items()}
    as_frame = {
        "psi": pd.DataFrame.from_dict(psi, orient="index"),
        "ks": pd.DataFrame.from_dict(ks, orient="index"),
    }
    for frame in as_frame.values():
        frame.index.name = "window"
    return as_frame["psi"], as_frame["ks"]


def null_psi(
    profile: FeatureProfile, reference: np.ndarray, size: int, seeds: Sequence[int]
) -> np.ndarray:
    """PSI of samples of ``size`` rows drawn, with replacement, from the reference itself.

    A sample drawn from the reference has no drift against it, so these values are what PSI
    reads as pure sampling noise at that size: its null. PSI grows with that noise, about
    (bins - 1) / size for a small sample, so its cuts are independent of sample size only
    when the sample is large enough for the noise to sit far below them.

    Args:
        profile: The feature's profile, built from ``reference``.
        reference: The feature's reference values.
        size: Rows per sample, the size of the window being judged.
        seeds: One sample per seed.

    Returns:
        One PSI per seed.
    """
    x = np.asarray(reference, dtype=float)
    draws = (x[np.random.default_rng(seed).integers(0, len(x), size)] for seed in seeds)
    return np.asarray([profile.psi(sample) for sample in draws])


def null_psi_quantiles(
    profiles: Mapping[str, FeatureProfile],
    reference: pd.DataFrame,
    sizes: Mapping[str, int],
    seeds: Sequence[int],
    quantile: float,
) -> pd.DataFrame:
    """A quantile of ``null_psi`` for every feature at every sample size.

    Args:
        profiles: From ``reference_profiles``.
        reference: The features over the reference period, the profiles' source.
        sizes: A label for each sample size, and the size.
        seeds: One sample per seed, at every size.
        quantile: Which quantile of the draws to report, between 0 and 1.

    Returns:
        One row per feature and one column per label of ``sizes``.
    """
    return pd.DataFrame(
        {
            label: {
                name: float(
                    np.quantile(
                        null_psi(profile, reference[name].to_numpy(), size, seeds), quantile
                    )
                )
                for name, profile in profiles.items()
            }
            for label, size in sizes.items()
        }
    )


def bin_shares(profile: FeatureProfile, sample: np.ndarray) -> pd.DataFrame:
    """The shares PSI compares: each bin's share of the reference rows and of a sample.

    The shares are the raw ones, before PSI drops the bins empty on both sides and floors the
    zeros.

    Args:
        profile: The feature's profile.
        sample: Values of the feature.

    Returns:
        One row per bin, with the columns ``reference`` and ``sample``. A discrete feature's
        bins are labelled by their value, and the bin for unseen values by NaN; a continuous
        feature's, by their upper edge, and the open last bin by infinity.
    """
    counts = profile.counts(sample)
    last = np.nan if profile.discrete else np.inf
    shares = pd.DataFrame(
        {
            "reference": profile.reference_counts / profile.reference_counts.sum(),
            "sample": counts / counts.sum(),
        },
        index=pd.Index([*profile.edges, last], name="bin"),
    )
    return shares


def zero_shares(frame: pd.DataFrame, columns: Sequence[str], windows: np.ndarray) -> pd.DataFrame:
    """Share of rows equal to zero in each column, window by window.

    Args:
        frame: Raw rows.
        columns: Columns to read.
        windows: The window number of each row.

    Returns:
        One row per window and one column per name of ``columns``.
    """
    shares = frame[list(columns)].eq(0).groupby(np.asarray(windows)).mean()
    shares.index.name = "window"
    return shares


def volume_by_window(steps: np.ndarray, labels: np.ndarray, windows: np.ndarray) -> pd.DataFrame:
    """How many rows of each class arrive per step, window by window.

    Args:
        steps: The step of each row.
        labels: 0/1 labels.
        windows: The window number of each row; one constant value gives the whole sample.

    Returns:
        One row per window: ``steps`` (distinct steps with rows), ``legitimate_per_step`` and
        ``frauds_per_step``.
    """
    frame = pd.DataFrame({"step": steps, "fraud": np.asarray(labels) == 1, "window": windows})
    grouped = frame.groupby("window")
    distinct = grouped["step"].nunique()
    frauds = grouped["fraud"].sum()
    return pd.DataFrame(
        {
            "steps": distinct,
            "legitimate_per_step": (grouped.size() - frauds) / distinct,
            "frauds_per_step": frauds / distinct,
        }
    )


def hour_profile(steps: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Share of each class's rows that fall in each hour of the day, ``step % 24``.

    Args:
        steps: The step of each row.
        labels: 0/1 labels.

    Returns:
        One row per hour, 0 to 23, and the columns ``legitimate`` and ``fraud``, each adding
        up to 1.
    """
    hour = np.asarray(steps) % HOURS_PER_DAY
    fraud = np.asarray(labels) == 1
    shares = {
        name: pd.Series(hour[rows]).value_counts(normalize=True)
        for name, rows in (("legitimate", ~fraud), ("fraud", fraud))
    }
    profile = pd.DataFrame(shares).reindex(range(HOURS_PER_DAY), fill_value=0.0)
    profile.index.name = "hour"
    return profile
