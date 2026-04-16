"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_team_stats() -> pd.DataFrame:
    """Minimal team stats DataFrame for testing features."""
    rng = np.random.default_rng(42)
    seasons = list(range(1984, 1995))
    teams = ["BOS", "LAL", "CHI", "DET", "PHO"]
    rows = []
    for season in seasons:
        for team in teams:
            rows.append(
                {
                    "season": season,
                    "Team": team,
                    "Team_abbrev": team,
                    "W": rng.integers(20, 60),
                    "L": rng.integers(20, 60),
                    "ORtg": 100 + rng.normal(0, 5),
                    "DRtg": 100 + rng.normal(0, 5),
                    "NRtg": rng.normal(0, 6),
                    "Pace": 90 + rng.normal(0, 5),
                    "eFG%": 0.48 + rng.normal(0, 0.03),
                    "TOV%": 14 + rng.normal(0, 2),
                    "ORB%": 25 + rng.normal(0, 4),
                    "DRB%": 75 + rng.normal(0, 4),
                    "FT/FGA": 0.22 + rng.normal(0, 0.03),
                    "opp_eFG%": 0.48 + rng.normal(0, 0.03),
                    "opp_TOV%": 14 + rng.normal(0, 2),
                    "SRS": rng.normal(0, 4),
                    "MOV": rng.normal(0, 6),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_playoff_series() -> pd.DataFrame:
    """Synthetic playoff series data for 5 seasons."""
    rows = []
    for season in range(1990, 1995):
        # First round: 4 series per conference
        pairs = [("BOS", "NYK"), ("CHI", "MIL"), ("LAL", "POR"), ("PHO", "UTA")]
        for team_a, team_b in pairs:
            wins_a = 4
            wins_b = np.random.default_rng(season).integers(0, 4)
            rows.append(
                {
                    "season": season,
                    "team_a": team_a,
                    "team_b": team_b,
                    "team_a_wins": wins_a,
                    "team_b_wins": wins_b,
                    "series_winner": team_a,
                    "series_length": wins_a + wins_b,
                    "seed_a": 1,
                    "seed_b": 8,
                    "conference": "East" if team_a in {"BOS", "CHI"} else "West",
                    "round": "first_round",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_player_advanced() -> pd.DataFrame:
    """Minimal player advanced stats DataFrame."""
    rng = np.random.default_rng(42)
    seasons = [1990, 1991, 1992]
    players_per_team = [
        ("BOS", ["Larry Bird", "Robert Parish", "Kevin McHale"]),
        ("LAL", ["Magic Johnson", "James Worthy", "Byron Scott"]),
        ("CHI", ["Michael Jordan", "Scottie Pippen", "Horace Grant"]),
    ]
    rows = []
    for season in seasons:
        for team, players in players_per_team:
            for i, player in enumerate(players):
                rows.append(
                    {
                        "season": season,
                        "Player": player,
                        "Team": team,
                        "Team_abbrev": team,
                        "Pos": ["G", "F", "C"][i % 3],
                        "G": rng.integers(60, 82),
                        "GS": rng.integers(50, 82),
                        "MP": rng.integers(25, 40),
                        "PTS": rng.uniform(10, 30),
                        "TRB": rng.uniform(3, 12),
                        "AST": rng.uniform(1, 10),
                        "BPM": rng.normal(3, 3),
                        "VORP": rng.uniform(0, 6),
                        "WS/48": rng.uniform(0.05, 0.2),
                        "PER": rng.uniform(12, 28),
                        "TS%": rng.uniform(0.50, 0.62),
                        "USG%": rng.uniform(15, 30),
                        "DBPM": rng.normal(0, 2),
                        "DRB": rng.uniform(2, 8),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_series_dataset(
    sample_team_stats, sample_playoff_series, sample_player_advanced
) -> pd.DataFrame:
    """Minimal series dataset with matchup features for modeling tests."""
    rows = []
    for _, series in sample_playoff_series.iterrows():
        rows.append(
            {
                "season": series["season"],
                "team_a": series["team_a"],
                "team_b": series["team_b"],
                "higher_seed": series["team_a"],
                "lower_seed": series["team_b"],
                "series_winner": series["series_winner"],
                "series_length": series["series_length"],
                "higher_seed_wins": 1,
                "seed_diff": 7,
                "home_court_advantage": 1,
                "delta_NRtg": np.random.normal(2, 3),
                "delta_BPM": np.random.normal(1, 2),
                "delta_VORP": np.random.normal(2, 3),
                "delta_adj_VORP": np.random.normal(1.5, 2.5),
                "delta_Experience": np.random.normal(0, 2),
                "higher_Star_injured": 0,
                "lower_Star_injured": 0,
                "higher_Roster_VORP_available_pct": 1.0,
                "lower_Roster_VORP_available_pct": 1.0,
                "era_analytics": 0,
                "era_defensive": 0,
                "era_showtime": 1,
                "era_transition": 0,
                "season_flag": 0,
            }
        )
    return pd.DataFrame(rows)
