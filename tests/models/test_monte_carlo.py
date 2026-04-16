"""Unit tests for Monte Carlo game simulation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nba_predictor.models.monte_carlo import (
    home_win_probability,
    simulate_series,
    simulate_series_from_features,
)


def test_probabilities_sum_to_one():
    """P(higher seed wins) + P(lower seed wins) = 1."""
    result = simulate_series(0.6, 0.4, n_simulations=10000, random_seed=42)
    total = result["p_higher_seed_wins"] + result["p_lower_seed_wins"]
    assert abs(total - 1.0) < 1e-10


def test_series_length_probabilities_sum_to_one():
    """Series length probabilities should sum to 1."""
    result = simulate_series(0.6, 0.4, n_simulations=10000, random_seed=42)
    length_sum = (
        result["p_length_4"] + result["p_length_5"] + result["p_length_6"] + result["p_length_7"]
    )
    assert abs(length_sum - 1.0) < 0.01, f"Length probs sum to {length_sum}"


def test_dominant_team_wins_more():
    """A team with p=0.9 home, 0.7 away should win the series most of the time."""
    result = simulate_series(0.9, 0.7, n_simulations=10000, random_seed=42)
    assert (
        result["p_higher_seed_wins"] > 0.90
    ), f"Dominant team should win >90% of series, got {result['p_higher_seed_wins']:.3f}"


def test_equal_teams_50pct():
    """Teams with p=0.5 everywhere should win ~50% of series."""
    result = simulate_series(0.5, 0.5, n_simulations=20000, random_seed=42)
    assert (
        0.45 < result["p_higher_seed_wins"] < 0.55
    ), f"Equal teams should win ~50%, got {result['p_higher_seed_wins']:.3f}"


def test_min_series_length_is_4():
    """No series should last fewer than 4 games."""
    result = simulate_series(0.9, 0.9, n_simulations=5000, random_seed=42)
    # These keys should all exist
    assert "p_length_4" in result
    assert "p_length_5" in result
    assert "p_length_6" in result
    assert "p_length_7" in result
    # No other lengths
    for length in [1, 2, 3, 8]:
        assert f"p_length_{length}" not in result


def test_expected_length_in_range():
    """Expected series length should be between 4 and 7."""
    result = simulate_series(0.6, 0.5, n_simulations=10000, random_seed=42)
    assert 4.0 <= result["expected_length"] <= 7.0


def test_home_win_probability_increases_with_delta():
    """Higher NRtg delta should give higher home win probability."""
    p_low = home_win_probability(-5.0)
    p_med = home_win_probability(0.0)
    p_high = home_win_probability(5.0)
    assert p_low < p_med < p_high


def test_home_win_probability_range():
    """Win probability should always be in (0, 1)."""
    for delta in [-20, -10, -5, 0, 5, 10, 20]:
        p = home_win_probability(delta)
        assert 0 < p < 1, f"delta={delta}: probability {p} out of range"


def test_reproducibility():
    """Same seed should give identical results."""
    r1 = simulate_series(0.65, 0.55, n_simulations=5000, random_seed=99)
    r2 = simulate_series(0.65, 0.55, n_simulations=5000, random_seed=99)
    assert r1["p_higher_seed_wins"] == r2["p_higher_seed_wins"]


def test_simulate_from_features():
    """simulate_series_from_features with positive delta should favor higher seed."""
    result = simulate_series_from_features({"delta_NRtg": 8.0}, n_simulations=5000, random_seed=42)
    assert (
        result["p_higher_seed_wins"] > 0.6
    ), "Positive NRtg delta should give >60% win probability"


def test_simulate_from_features_neutral():
    """Zero delta should give ~50% win probability."""
    result = simulate_series_from_features({"delta_NRtg": 0.0}, n_simulations=20000, random_seed=42)
    # Home court advantage means higher seed should be slightly favored even at delta=0
    assert 0.5 < result["p_higher_seed_wins"] < 0.70
