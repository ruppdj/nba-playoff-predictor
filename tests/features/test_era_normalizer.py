"""Unit tests for era normalization.

Critical correctness requirement: z-scoring must be applied PER SEASON,
not across the full dataset. Tests enforce this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nba_predictor.features.era_normalizer import (
    z_score_per_season,
    add_era_flags,
    add_season_flag,
    league_averages_by_season,
)


def test_z_score_per_season_zero_mean():
    """Within each season, z-scored values should have mean ~0."""
    df = pd.DataFrame({
        "season": [2020] * 10 + [2021] * 10,
        "ORtg": np.random.default_rng(0).uniform(95, 115, 20),
    })
    result = z_score_per_season(df, ["ORtg"])
    for season, group in result.groupby("season"):
        assert abs(group["ORtg_norm"].mean()) < 1e-10, (
            f"Season {season}: z-scored mean should be 0, got {group['ORtg_norm'].mean()}"
        )


def test_z_score_per_season_unit_std():
    """Within each season, z-scored values should have std ~1."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "season": [2020] * 15 + [2021] * 15,
        "ORtg": rng.uniform(95, 115, 30),
    })
    result = z_score_per_season(df, ["ORtg"])
    for season, group in result.groupby("season"):
        std = group["ORtg_norm"].std(ddof=1)
        assert abs(std - 1.0) < 0.01, (
            f"Season {season}: z-scored std should be ~1, got {std}"
        )


def test_z_score_does_not_use_global_mean():
    """Normalization must NOT use the global mean/std across all seasons.

    This is a key correctness requirement. If a team has ORtg=110 in a
    season where the league average is 105, their normalized score should
    reflect that season's average, not the average of all seasons combined.
    """
    # Season 1: league avg = 100, Season 2: league avg = 115
    df = pd.DataFrame({
        "season": [1, 1, 1, 2, 2, 2],
        "ORtg":   [100, 100, 100, 115, 115, 115],  # exactly at each season's mean
    })
    result = z_score_per_season(df, ["ORtg"])
    # All teams are exactly at their season average → all norms should be ~0
    assert all(abs(result["ORtg_norm"]) < 1e-10), (
        "Teams at season average should have norm=0 regardless of other seasons"
    )


def test_z_score_preserves_original_col():
    """Original columns should not be modified."""
    df = pd.DataFrame({
        "season": [2020, 2020, 2021, 2021],
        "ORtg": [100.0, 110.0, 105.0, 115.0],
    })
    result = z_score_per_season(df, ["ORtg"])
    pd.testing.assert_series_equal(result["ORtg"], df["ORtg"])


def test_add_era_flags_coverage():
    """Every season from 1984-2025 should be assigned exactly one era."""
    df = pd.DataFrame({"season": list(range(1984, 2026))})
    result = add_era_flags(df)
    era_cols = [c for c in result.columns if c.startswith("era_")]
    assert len(era_cols) == 4, f"Expected 4 era columns, got {len(era_cols)}"
    # Each row should have exactly one era = 1
    era_sum = result[era_cols].sum(axis=1)
    assert (era_sum == 1).all(), "Each season should belong to exactly one era"


def test_add_era_flags_correct_assignment():
    """Specific seasons should map to correct eras."""
    df = pd.DataFrame({"season": [1990, 2000, 2010, 2020]})
    result = add_era_flags(df)
    assert result.loc[result["season"] == 1990, "era_showtime"].iloc[0] == 1
    assert result.loc[result["season"] == 2000, "era_defensive"].iloc[0] == 1
    assert result.loc[result["season"] == 2010, "era_transition"].iloc[0] == 1
    assert result.loc[result["season"] == 2020, "era_analytics"].iloc[0] == 1


def test_add_season_flag_aberrant_seasons():
    """Lockout (2012) and bubble (2020) seasons should be flagged."""
    df = pd.DataFrame({"season": [2010, 2012, 2015, 2020, 2021]})
    result = add_season_flag(df)
    assert result.loc[result["season"] == 2012, "season_flag"].iloc[0] == 1
    assert result.loc[result["season"] == 2020, "season_flag"].iloc[0] == 1
    assert result.loc[result["season"] == 2010, "season_flag"].iloc[0] == 0
    assert result.loc[result["season"] == 2021, "season_flag"].iloc[0] == 0


def test_z_score_missing_columns_handled():
    """Columns not present in the DataFrame should be skipped silently."""
    df = pd.DataFrame({
        "season": [2020, 2021],
        "ORtg": [100.0, 105.0],
    })
    # Request normalization of a column that doesn't exist
    result = z_score_per_season(df, ["ORtg", "NONEXISTENT_COL"])
    assert "ORtg_norm" in result.columns
    assert "NONEXISTENT_COL_norm" not in result.columns


def test_league_averages_by_season(sample_team_stats):
    """League averages should be computed per season."""
    result = league_averages_by_season(sample_team_stats, ["ORtg", "DRtg"])
    assert "season" in result.columns
    assert "ORtg" in result.columns
    assert len(result) == sample_team_stats["season"].nunique()
