"""Unit tests for player feature aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nba_predictor.features.player_features import aggregate_player_to_team


def test_vorp_sum_is_additive(sample_player_advanced):
    """team_VORP_sum should equal the sum of all players' VORP."""
    result = aggregate_player_to_team(sample_player_advanced)
    # Check for one team-season
    row = result[(result["season"] == 1990) & (result["Team_abbrev"] == "BOS")].iloc[0]
    pa = sample_player_advanced[
        (sample_player_advanced["season"] == 1990)
        & (sample_player_advanced["Team_abbrev"] == "BOS")
    ]
    expected_vorp = pa["VORP"].sum()
    assert abs(row["team_VORP_sum"] - expected_vorp) < 1e-6


def test_has_allnba_threshold(sample_player_advanced):
    """Has_AllNBA_player should be 1 if any player has BPM > 5.0."""
    result = aggregate_player_to_team(sample_player_advanced)
    for _, row in result.iterrows():
        team = row["Team_abbrev"]
        season = row["season"]
        pa = sample_player_advanced[
            (sample_player_advanced["season"] == season)
            & (sample_player_advanced["Team_abbrev"] == team)
        ]
        has_allnba = int((pa["BPM"] > 5.0).any())
        assert row["Has_AllNBA_player"] == has_allnba


def test_output_has_all_expected_columns(sample_player_advanced):
    """aggregate_player_to_team should return all key feature columns."""
    result = aggregate_player_to_team(sample_player_advanced)
    expected_cols = [
        "team_VORP_sum", "team_BPM_weighted_avg", "team_WS48_weighted_avg",
        "Top3_VORP_sum", "Star_player_BPM", "Has_AllNBA_player", "Top8_WS48_avg",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_one_row_per_team_season(sample_player_advanced):
    """Output should have exactly one row per unique team-season."""
    result = aggregate_player_to_team(sample_player_advanced)
    expected = sample_player_advanced.groupby(["season", "Team_abbrev"]).ngroups
    assert len(result) == expected
