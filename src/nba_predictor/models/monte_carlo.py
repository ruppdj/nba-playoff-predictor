"""Monte Carlo game-level series simulation.

Simulates NBA playoff series by:
  1. Computing per-game win probability for the higher-seeded team
     using the logistic relationship between NRtg delta and home/away location.
  2. Simulating 10,000 series using the actual 2-2-1-1-1 home court format.
  3. Returning win probability and series length distribution.

This module is used by bracket_simulator.py to propagate uncertainty
through all four rounds of the playoffs.
"""

from __future__ import annotations

import random

import numpy as np

# NBA playoff home court format (from higher seed's perspective):
# Games 1-2: home, Games 3-4: away, Game 5: home, Game 6: away, Game 7: home
GAME_LOCATIONS = ["H", "H", "A", "A", "H", "A", "H"]


def home_win_probability(
    nrtg_delta: float,
    home_court_boost: float = 2.5,
) -> float:
    """Convert a net rating differential to a per-game home win probability.

    Uses logistic function: p = 1 / (1 + exp(-k * nrtg_delta))
    with an empirically calibrated scale factor k and home court boost.

    Args:
        nrtg_delta: Higher seed NRtg minus lower seed NRtg (positive = higher seed better).
        home_court_boost: Additional NRtg points for the home team (historical ~3 points).

    Returns:
        Probability that the home team (higher seed) wins a home game.
    """
    # Scale factor: in NBA, a 10-point NRtg advantage → ~73% win probability
    k = 0.104
    effective_delta = nrtg_delta + home_court_boost
    return float(1.0 / (1.0 + np.exp(-k * effective_delta)))


def away_win_probability(nrtg_delta: float, home_court_boost: float = 2.5) -> float:
    """Win probability for the higher seed in an away game."""
    k = 0.104
    effective_delta = nrtg_delta - home_court_boost
    return float(1.0 / (1.0 + np.exp(-k * effective_delta)))


def simulate_series(
    p_home: float,
    p_away: float,
    n_simulations: int = 10000,
    random_seed: int | None = None,
) -> dict[str, float]:
    """Simulate a best-of-7 series many times and return outcome distribution.

    Args:
        p_home: Probability that higher seed wins a HOME game.
        p_away: Probability that higher seed wins an AWAY game.
        n_simulations: Number of Monte Carlo simulations.
        random_seed: Optional seed for reproducibility.

    Returns:
        Dictionary with:
          - p_higher_seed_wins: overall series win probability
          - p_length_4, p_length_5, p_length_6, p_length_7: series length probs
          - expected_length: expected number of games
    """
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    higher_seed_wins = 0
    length_counts = {4: 0, 5: 0, 6: 0, 7: 0}

    for _ in range(n_simulations):
        wins_higher = 0
        wins_lower = 0

        for _game_num, location in enumerate(GAME_LOCATIONS):
            p = p_home if location == "H" else p_away
            if rng.random() < p:
                wins_higher += 1
            else:
                wins_lower += 1

            if wins_higher == 4 or wins_lower == 4:
                total_games = wins_higher + wins_lower
                length_counts[total_games] = length_counts.get(total_games, 0) + 1
                if wins_higher == 4:
                    higher_seed_wins += 1
                break

    p_higher = higher_seed_wins / n_simulations
    p_lengths = {k: v / n_simulations for k, v in length_counts.items()}
    expected = sum(length * prob for length, prob in p_lengths.items())

    return {
        "p_higher_seed_wins": p_higher,
        "p_lower_seed_wins": 1.0 - p_higher,
        "p_length_4": p_lengths.get(4, 0.0),
        "p_length_5": p_lengths.get(5, 0.0),
        "p_length_6": p_lengths.get(6, 0.0),
        "p_length_7": p_lengths.get(7, 0.0),
        "expected_length": expected,
    }


def simulate_series_from_features(
    features: dict[str, float],
    n_simulations: int = 10000,
    home_court_boost: float = 2.5,
    random_seed: int | None = None,
) -> dict[str, float]:
    """Simulate a series using matchup features.

    Args:
        features: Matchup feature dict. Must contain 'delta_NRtg'.
        n_simulations: Number of Monte Carlo draws.
        home_court_boost: Home court advantage in NRtg points.
        random_seed: Optional seed.

    Returns:
        Simulation result dict (see simulate_series).
    """
    nrtg_delta = features.get("delta_NRtg", 0.0)
    p_home = home_win_probability(float(nrtg_delta), home_court_boost)
    p_away = away_win_probability(float(nrtg_delta), home_court_boost)
    return simulate_series(p_home, p_away, n_simulations, random_seed)
