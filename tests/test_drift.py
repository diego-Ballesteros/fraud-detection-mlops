"""Tests of fraud_detection.monitoring.drift: hand-computed values and positive controls.

PSI near 0 on real data is a negative control, not evidence that nothing drifted: a mute
detector and a broken one read the same. Every detector here is therefore shown to fire on a
synthetic shift of known magnitude before its silence means anything.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp, norm

from fraud_detection.explain.global_explain import model_features
from fraud_detection.features.pipeline import CONFIGURATIONS
from fraud_detection.models.registry import ARTIFACT_PATH, MODEL_INPUT_COLUMNS, load_pipeline
from fraud_detection.monitoring.drift import (
    MODERATE,
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    SIGNIFICANT,
    STABLE,
    FeatureProfile,
    bin_shares,
    drift_table,
    hour_profile,
    null_psi,
    null_psi_quantiles,
    psi_from_counts,
    psi_level,
    reference_profiles,
    volume_by_window,
    window_drift,
    zero_shares,
)

MONEY = CONFIGURATIONS["full"].money


def test_psi_by_hand() -> None:
    # Shares 0.5 / 0.5 against 0.25 / 0.75.
    expected = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(0.75 / 0.5)
    assert psi_from_counts(np.array([50, 50]), np.array([25, 75])) == pytest.approx(expected)
    assert expected == pytest.approx(0.274653, abs=1e-6)


def test_identical_shares_give_zero_whatever_the_sizes() -> None:
    assert psi_from_counts(np.array([10, 20, 30]), np.array([1, 2, 3])) == 0.0


def test_a_bin_empty_on_both_sides_is_dropped() -> None:
    assert psi_from_counts(np.array([5, 0, 5]), np.array([5, 0, 5])) == 0.0
    # With equal floors on both sides a kept empty bin adds nothing, so the case above cannot
    # tell. Here the floors differ (reference 1e-5 / 1e6, production 1e-4): kept, the third
    # bin would add about 0.0016.
    r = np.array([0.99999, 0.00001])
    p = np.array([0.5, 0.5])
    expected = float(np.sum((p - r) * np.log(p / r)))
    assert psi_from_counts(np.array([99_999, 1, 0]), np.array([50, 50, 0])) == pytest.approx(
        expected, rel=1e-12
    )


def test_zero_shares_are_floored_by_evidentlys_rule() -> None:
    # Reference shares 0.5, 0, 0.5: the smallest nonzero is above 1e-4, so the floor is 1e-4.
    r = np.array([0.5, 1e-4, 0.5])
    p = np.array([0.25, 0.25, 0.5])
    expected = float(np.sum((p - r) * np.log(p / r)))
    assert psi_from_counts(np.array([100, 0, 100]), np.array([50, 50, 100])) == pytest.approx(
        expected
    )
    # Reference shares 0.99999, 0.00001, 0: the smallest nonzero is not above 1e-4, so the
    # reference floor is 1e-5 / 1e6. Production shares 0.5, 0, 0.5: its floor is 1e-4.
    r = np.array([0.99999, 1e-5, 1e-11])
    p = np.array([0.5, 1e-4, 0.5])
    expected = float(np.sum((p - r) * np.log(p / r)))
    assert psi_from_counts(np.array([99_999, 1, 0]), np.array([1, 0, 1])) == pytest.approx(expected)


def test_psi_needs_rows_on_the_same_bins() -> None:
    with pytest.raises(ValueError, match="same bins"):
        psi_from_counts(np.array([1, 1]), np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="same bins"):
        psi_from_counts(np.array([0, 0]), np.array([1, 1]))


def test_levels() -> None:
    assert psi_level(0.0) == STABLE
    assert psi_level(PSI_MODERATE) == MODERATE
    assert psi_level(PSI_SIGNIFICANT) == SIGNIFICANT


def test_continuous_bins_are_reference_deciles_right_closed_and_open_ended() -> None:
    profile = FeatureProfile.from_reference("x", np.arange(1.0, 11.0), discrete=False)
    assert profile.edges.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert profile.reference_counts.tolist() == [1] * 10
    # 1 -> (-inf, 1]; 1.5 -> (1, 2]; 9 -> (8, 9]; 9.5 and 10 -> (9, inf); -100 and 100 at the ends.
    counts = profile.counts(np.array([1.0, 1.5, 9.0, 9.5, 10.0, -100.0, 100.0]))
    assert counts.tolist() == [2, 1, 0, 0, 0, 0, 0, 0, 1, 3]


def test_duplicate_deciles_merge() -> None:
    reference = np.r_[np.zeros(60), np.arange(1.0, 41.0)]  # 60% zeros
    profile = FeatureProfile.from_reference("x", reference, discrete=False)
    assert profile.edges[0] == 0
    assert len(profile.edges) == len(set(profile.edges)) < 9
    assert profile.reference_counts[0] == 60


def test_discrete_bins_are_the_reference_values_plus_one_for_unseen() -> None:
    profile = FeatureProfile.from_reference("b", np.array([0, 0, 1, 1, 1]), discrete=True)
    assert profile.edges.tolist() == [0, 1]
    assert profile.reference_counts.tolist() == [2, 3, 0]
    assert profile.counts(np.array([0, 1, 2, 0.5])).tolist() == [1, 1, 2]


def test_the_profile_is_never_moved_by_what_it_compares() -> None:
    reference = np.random.default_rng(0).normal(0, 1, 1_000)
    profile = FeatureProfile.from_reference("x", reference, discrete=False)
    edges, counts = profile.edges.copy(), profile.reference_counts.copy()
    profile.psi(np.random.default_rng(1).normal(5, 1, 1_000))
    profile.ks(np.random.default_rng(1).normal(5, 1, 1_000))
    assert np.array_equal(profile.edges, edges)
    assert np.array_equal(profile.reference_counts, counts)


def test_a_nan_in_the_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="no NaN"):
        FeatureProfile.from_reference("x", np.array([1.0, np.nan]), discrete=False)


def test_ks_by_hand() -> None:
    profile = FeatureProfile.from_reference("x", np.array([1.0, 2.0, 3.0, 4.0]), discrete=False)
    assert profile.ks(np.array([3.0, 4.0, 5.0, 6.0])) == 0.5
    assert profile.ks(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    # Ties: F_ref(0) = 0.75 and F_sample(0) = 0.25.
    tied = FeatureProfile.from_reference("b", np.array([0.0, 0.0, 0.0, 1.0]), discrete=True)
    assert tied.ks(np.array([0.0, 1.0, 1.0, 1.0])) == 0.5
    with pytest.raises(ValueError, match="non-empty"):
        profile.ks(np.array([]))


@pytest.mark.parametrize("discrete", [False, True])
def test_ks_matches_scipy(discrete: bool) -> None:
    rng = np.random.default_rng(3)
    if discrete:
        reference, sample = rng.integers(0, 6, 5_000), rng.integers(1, 7, 3_000)
    else:
        reference, sample = rng.normal(0, 1, 5_000), rng.normal(0.2, 1.3, 3_000)
    profile = FeatureProfile.from_reference("x", reference, discrete=discrete)
    assert profile.ks(sample) == pytest.approx(ks_2samp(reference, sample).statistic, abs=1e-12)


def test_positive_control_a_known_shift_fires_in_proportion_to_its_size() -> None:
    reference = np.random.default_rng(0).normal(0, 1, 200_000)
    profile = FeatureProfile.from_reference("x", reference, discrete=False)
    shifts = [0.0, 0.1, 0.2, 0.4, 0.8, 1.0]
    psi = [profile.psi(np.random.default_rng(1).normal(d, 1, 100_000)) for d in shifts]
    ks = [profile.ks(np.random.default_rng(1).normal(d, 1, 100_000)) for d in shifts]
    assert psi == sorted(psi)
    assert ks == sorted(ks)
    # Negative control: a new sample of the same distribution.
    assert psi[0] < 0.001
    assert psi_level(psi[0]) == STABLE
    # One shift in each band.
    assert [psi_level(psi[shifts.index(d)]) for d in (0.2, 0.4, 0.8)] == [
        STABLE,
        MODERATE,
        SIGNIFICANT,
    ]
    for d, value, distance in zip(shifts, psi, ks, strict=True):
        if d >= 0.2:
            # For two unit normals PSI tends to the symmetric KL divergence, d squared; decile
            # bins lose a little of it.
            assert 0.85 * d**2 < value < d**2, d
        # The KS distance between N(0, 1) and N(d, 1) is 2 Phi(d / 2) - 1.
        assert distance == pytest.approx(2 * norm.cdf(d / 2) - 1, abs=0.01), d


def test_bins_and_ks_ignore_a_strictly_increasing_transform() -> None:
    rng = np.random.default_rng(5)
    reference, sample = rng.lognormal(10, 2, 20_000), rng.lognormal(10.5, 2, 10_000)
    raw = FeatureProfile.from_reference("x", reference, discrete=False)
    logged = FeatureProfile.from_reference("x", np.log1p(reference), discrete=False)
    assert logged.psi(np.log1p(sample)) == raw.psi(sample)
    assert logged.ks(np.log1p(sample)) == raw.ks(sample)


def _synthetic_rows(rng: np.random.Generator, hours: np.ndarray, size: int) -> pd.DataFrame:
    """Raw rows that never take the whole balance, with the given hours of the day."""
    amount = rng.lognormal(9, 1.5, size).round(2)
    return pd.DataFrame(
        {
            "step": rng.choice(hours, size) + 24 * rng.integers(1, 15, size),
            "type": pd.Categorical(
                rng.choice(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], size),
                categories=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            ),
            "amount": amount,
            "oldbalanceOrg": amount + rng.uniform(1, 1_000, size).round(2),
            "oldbalanceDest": rng.lognormal(8, 2, size).round(2),
        }
    )[list(MODEL_INPUT_COLUMNS)]


@pytest.mark.skipif(
    not ARTIFACT_PATH.is_file(),
    reason="The artifact is written by scripts/register_model.py and is not in this checkout.",
)
def test_positive_control_through_the_artifact() -> None:
    pipeline = load_pipeline(ARTIFACT_PATH)
    rng = np.random.default_rng(7)
    daytime = np.arange(8, 20)  # volume only during the day: a non-uniform hour profile
    rows = _synthetic_rows(rng, daytime, 20_000)
    features = model_features(pipeline, rows)
    discrete = [name for name in features.columns if name not in MONEY]
    profiles = reference_profiles(features, discrete)

    same = drift_table(profiles, features)
    assert (same["psi"] == 0).all()
    assert (same["ks"] == 0).all()

    # Every amount tripled, balances moved with it so the flag stays false: only the money
    # features move, and only money drift is reported.
    tripled = rows.assign(amount=rows["amount"] * 3, oldbalanceOrg=rows["oldbalanceOrg"] * 3)
    table = drift_table(profiles, model_features(pipeline, tripled))
    assert table.loc["amount", "level"] == SIGNIFICANT
    assert table.loc["oldbalanceOrg", "level"] == SIGNIFICANT
    assert (table.loc[discrete, "psi"] == 0).all()
    assert table.loc["oldbalanceDest", "psi"] == 0

    # The same day shifted by 12 hours: the hour features fire, nothing else moves.
    night = rows.assign(step=rows["step"] + 12)
    table = drift_table(profiles, model_features(pipeline, night))
    assert table.loc["hour_sin", "level"] == SIGNIFICANT
    assert table.loc["hour_cos", "level"] == SIGNIFICANT
    assert (table.loc[[*MONEY, "type_CASH_IN", "amount_equals_origin_balance"], "psi"] == 0).all()

    # Uniform hours shifted by 12 are the same distribution: identical support, no drift. A
    # shift of the clock is invisible unless the volume is uneven over the day.
    uniform = pd.concat([rows.assign(step=24 + hour) for hour in range(24)], ignore_index=True)
    uniform_profiles = reference_profiles(model_features(pipeline, uniform), discrete)
    rotated = uniform.assign(step=uniform["step"] + 12)
    table = drift_table(uniform_profiles, model_features(pipeline, rotated))
    assert table.loc["hour_sin", "psi"] == 0
    assert table.loc["hour_cos", "psi"] == 0
    # Control: the same detector fires when the whole day collapses on one hour.
    one_hour = drift_table(uniform_profiles, model_features(pipeline, uniform.assign(step=36)))
    assert one_hour.loc["hour_sin", "level"] == SIGNIFICANT


def test_window_drift_finds_the_window_that_moved() -> None:
    rng = np.random.default_rng(11)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 50_000), "b": rng.integers(0, 2, 50_000)})
    profiles = reference_profiles(reference, discrete=["b"])
    production = pd.DataFrame({"x": rng.normal(0, 1, 30_000), "b": rng.integers(0, 2, 30_000)})
    windows = np.repeat([1, 2, 3], 10_000)
    production.loc[windows == 2, "x"] += 1.0
    psi, ks = window_drift(profiles, production, windows)
    assert psi.index.tolist() == [1, 2, 3]
    assert psi.columns.tolist() == ["x", "b"]
    assert psi.loc[2, "x"] > PSI_SIGNIFICANT
    assert psi.loc[[1, 3], "x"].max() < PSI_MODERATE
    assert psi["b"].max() < PSI_MODERATE
    assert ks.loc[2, "x"] > 3 * ks.loc[[1, 3], "x"].max()


def test_null_psi_has_the_known_size_of_sampling_noise() -> None:
    # Ten equal bins: drawn from the reference itself, PSI behaves like a chi-square with
    # 9 degrees of freedom over the sample size, so its mean is about 9 / size.
    reference = np.repeat(np.arange(10.0), 1_000)
    profile = FeatureProfile.from_reference("x", reference, discrete=True)
    seeds = list(range(300))
    small = null_psi(profile, reference, 200, seeds)
    large = null_psi(profile, reference, 20_000, seeds)
    assert small.mean() == pytest.approx(9 / 200, rel=0.15)
    assert large.mean() == pytest.approx(9 / 20_000, rel=0.15)
    # At 200 rows the noise alone reaches the moderate cut in some draws; at 20,000 it never
    # stays an order of magnitude below it.
    assert small.max() > PSI_MODERATE > 10 * large.max()
    assert np.array_equal(null_psi(profile, reference, 200, seeds[:5]), small[:5])


def test_zero_shares_by_hand() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 0.0, 0.0], "b": [5.0, 0.0, 2.0, 3.0]})
    shares = zero_shares(frame, ["a", "b"], np.array([1, 1, 2, 2]))
    assert shares.index.tolist() == [1, 2]
    assert shares.loc[1].tolist() == [0.5, 0.5]
    assert shares.loc[2].tolist() == [1.0, 0.0]


def test_volume_by_window_by_hand() -> None:
    # Window 1: steps 1 and 2, three legitimate rows and one fraud. Window 2: step 3, one fraud.
    steps = np.array([1, 1, 2, 2, 3])
    labels = np.array([0, 0, 0, 1, 1])
    volume = volume_by_window(steps, labels, np.array([1, 1, 1, 1, 2]))
    assert volume.loc[1].tolist() == [2, 1.5, 0.5]
    assert volume.loc[2].tolist() == [1, 0.0, 1.0]
    whole = volume_by_window(steps, labels, np.zeros(5))
    assert whole.loc[0].tolist() == [3, 1.0, 2 / 3]


def test_hour_profile_by_hand() -> None:
    # Steps 1 and 25 are both hour 1; step 26 is hour 2.
    profile = hour_profile(np.array([1, 25, 26, 2]), np.array([0, 0, 0, 1]))
    assert profile.index.tolist() == list(range(24))
    assert profile.loc[1, "legitimate"] == pytest.approx(2 / 3)
    assert profile.loc[2, "legitimate"] == pytest.approx(1 / 3)
    assert profile.loc[2, "fraud"] == 1.0
    assert profile.sum().tolist() == [1.0, 1.0]


def test_null_psi_quantiles_is_the_quantile_of_the_draws() -> None:
    rng = np.random.default_rng(13)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 5_000), "b": rng.integers(0, 2, 5_000)})
    profiles = reference_profiles(reference, discrete=["b"])
    seeds = [1, 2, 3, 4, 5]
    table = null_psi_quantiles(profiles, reference, {"small": 50, "large": 2_000}, seeds, 0.95)
    assert table.index.tolist() == ["x", "b"]
    assert table.columns.tolist() == ["small", "large"]
    for name in ("x", "b"):
        for label, size in (("small", 50), ("large", 2_000)):
            draws = null_psi(profiles[name], reference[name].to_numpy(), size, seeds)
            assert table.loc[name, label] == np.quantile(draws, 0.95)
    assert (table["small"] > table["large"]).all()


def test_bin_shares_by_hand() -> None:
    discrete = FeatureProfile.from_reference("b", np.array([0, 0, 0, 1]), discrete=True)
    shares = bin_shares(discrete, np.array([0, 1, 1, 2]))
    assert shares["reference"].tolist() == [0.75, 0.25, 0.0]
    assert shares["sample"].tolist() == [0.25, 0.5, 0.25]
    assert shares.index[:2].tolist() == [0, 1]
    assert np.isnan(shares.index[2])
    continuous = FeatureProfile.from_reference("x", np.arange(1.0, 11.0), discrete=False)
    shares = bin_shares(continuous, np.array([0.5, 9.5]))
    assert shares.index[-1] == np.inf
    assert shares["reference"].tolist() == [0.1] * 10
    assert shares["sample"].tolist() == [0.5] + [0.0] * 8 + [0.5]
