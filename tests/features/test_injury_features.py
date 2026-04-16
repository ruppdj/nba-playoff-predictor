"""Unit tests for injury feature computation.

Key assertions:
  - A fully healthy roster has Roster_VORP_available_pct = 1.0
  - A team with all players OUT has Roster_VORP_available_pct ≈ 0.0
  - Role-specific flags (Lost_top_scorer, etc.) are set correctly
  - status_to_availability maps correctly to [0, 1]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nba_predictor.features.injury_features import (
    availability_from_gp_ratio,
    compute_injury_features,
    status_to_availability,
)

# ── Status mapping tests ──────────────────────────────────────────────────────


def test_status_out():
    assert status_to_availability("out") == 0.0
    assert status_to_availability("Out") == 0.0
    assert status_to_availability("OUT (DTD)") == 0.0


def test_status_questionable():
    assert status_to_availability("questionable") == 0.5


def test_status_available():
    assert status_to_availability("available") == 1.0
    assert status_to_availability("Active") == 1.0


def test_status_unknown_defaults_to_available():
    """Unknown status should default to fully available (conservative)."""
    assert status_to_availability("unknown status xyz") == 1.0


def test_gp_ratio_full_season():
    assert availability_from_gp_ratio(82, 82) == 1.0


def test_gp_ratio_half_season():
    assert abs(availability_from_gp_ratio(41, 82) - 0.5) < 1e-10


def test_gp_ratio_clamped():
    # Can't play more games than available
    assert availability_from_gp_ratio(90, 82) == 1.0
    assert availability_from_gp_ratio(0, 82) == 0.0


# ── compute_injury_features tests ────────────────────────────────────────────


def _make_roster(season: int, team: str, players: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [season] * len(players),
            "Team_abbrev": [team] * len(players),
            "PLAYER_NAME": players,
        }
    )


def _make_player_advanced(
    season: int,
    team: str,
    players: list[str],
    vorp: list[float],
    pts: list[float],
    trb: list[float],
    ast: list[float],
    dbpm: list[float],
    gp: list[int],
    mp: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [season] * len(players),
            "Team_abbrev": [team] * len(players),
            "Player": players,
            "Tm": [team] * len(players),
            "VORP": vorp,
            "BPM": [v / 2 for v in vorp],
            "PTS": pts,
            "TRB": trb,
            "AST": ast,
            "DBPM": dbpm,
            "G": gp,
            "MP": mp,
            "DRB": [r * 0.7 for r in trb],
        }
    )


def test_fully_healthy_roster_pct():
    """A team with all players available should have Roster_VORP_available_pct = 1.0."""
    season, team = 2020, "BOS"
    players = ["Player A", "Player B", "Player C"]
    roster = _make_roster(season, team, players)
    pa = _make_player_advanced(
        season,
        team,
        players,
        vorp=[5.0, 3.0, 1.0],
        pts=[25.0, 18.0, 10.0],
        trb=[5.0, 8.0, 3.0],
        ast=[6.0, 2.0, 1.0],
        dbpm=[1.0, 0.5, -0.5],
        gp=[82, 82, 82],
        mp=[36.0, 32.0, 24.0],
    )
    result = compute_injury_features(roster, None, pa)
    row = result[result["Team_abbrev"] == team].iloc[0]
    assert (
        abs(row["Roster_VORP_available_pct"] - 1.0) < 1e-6
    ), f"Healthy roster should have pct=1.0, got {row['Roster_VORP_available_pct']}"
    assert row["Star_injured"] == 0
    assert row["Lost_top_scorer"] == 0
    assert row["Lost_top_rebounder"] == 0
    assert row["Injured_player_count"] == 0


def test_star_player_out():
    """When the #1 VORP player has GP=0 (proxy for out), Star_injured should be 1."""
    season, team = 2021, "LAL"
    players = ["Star Player", "Role Player"]
    roster = _make_roster(season, team, players)
    pa = _make_player_advanced(
        season,
        team,
        players,
        vorp=[8.0, 2.0],
        pts=[30.0, 15.0],
        trb=[5.0, 6.0],
        ast=[7.0, 2.0],
        dbpm=[2.0, 0.0],
        gp=[0, 82],  # star played 0 games → availability ≈ 0
        mp=[36.0, 28.0],
    )
    result = compute_injury_features(roster, None, pa)
    row = result[result["Team_abbrev"] == team].iloc[0]
    assert row["Star_injured"] == 1, "Star with GP=0 should be marked injured"
    assert row["Lost_top_scorer"] == 1, "Top scorer with GP=0 should be Lost_top_scorer=1"


def test_adj_vorp_sum_less_than_full():
    """Injury-adjusted VORP should be <= raw VORP sum when players are injured."""
    season, team = 2022, "GSW"
    players = ["P1", "P2", "P3"]
    roster = _make_roster(season, team, players)
    pa = _make_player_advanced(
        season,
        team,
        players,
        vorp=[4.0, 3.0, 2.0],
        pts=[20.0, 16.0, 12.0],
        trb=[5.0, 7.0, 4.0],
        ast=[5.0, 3.0, 2.0],
        dbpm=[1.0, 0.5, 0.0],
        gp=[20, 82, 82],  # P1 played only 20 games → injured
        mp=[36.0, 30.0, 25.0],
    )
    result = compute_injury_features(roster, None, pa)
    row = result[result["Team_abbrev"] == team].iloc[0]
    raw_vorp = 4.0 + 3.0 + 2.0
    assert (
        row["adj_VORP_sum"] < raw_vorp
    ), "adj_VORP_sum should be less than raw sum when a player is injured"


def test_lost_top_rebounder():
    """If the leading rebounder has GP=0, Lost_top_rebounder should be 1."""
    season, team = 2023, "DEN"
    players = ["Big Man", "Guard"]
    roster = _make_roster(season, team, players)
    pa = _make_player_advanced(
        season,
        team,
        players,
        vorp=[2.0, 5.0],  # Guard is the star by VORP
        pts=[12.0, 25.0],
        trb=[14.0, 3.0],  # Big Man is top rebounder
        ast=[1.0, 8.0],
        dbpm=[1.0, 0.5],
        gp=[0, 82],  # Big Man is out
        mp=[28.0, 36.0],
    )
    result = compute_injury_features(roster, None, pa)
    row = result[result["Team_abbrev"] == team].iloc[0]
    assert row["Lost_top_rebounder"] == 1


def test_has_injury_data_flag_with_no_report():
    """When no injury report is provided, has_injury_data should be 0."""
    season, team = 2019, "MIL"
    players = ["Giannis"]
    roster = _make_roster(season, team, players)
    pa = _make_player_advanced(
        season,
        team,
        players,
        vorp=[9.0],
        pts=[27.0],
        trb=[13.0],
        ast=[6.0],
        dbpm=[3.0],
        gp=[72],
        mp=[32.0],
    )
    result = compute_injury_features(roster, None, pa)
    row = result[result["Team_abbrev"] == team].iloc[0]
    assert row["has_injury_data"] == 0
