"""Era normalization for cross-season comparability.

All rate stats are z-scored relative to the league average for that season.
This handles the dramatic rule changes, pace changes, and 3-point revolution
that make raw stats incomparable across eras (e.g. 0.36 3P% was elite in 2005
but mediocre in 2025).

CRITICAL: normalization is per-season, NOT across the full dataset.
A common bug is accidentally using the full-dataset mean/std, which
conflates era effects with team quality.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

# Columns to normalize (z-score per season)
NORMALIZE_COLS = [
    "ORtg",
    "DRtg",
    "NRtg",
    "Pace",
    "eFG%",
    "TOV%",
    "ORB%",
    "DRB%",
    "FT/FGA",
    "opp_eFG%",
    "opp_TOV%",
    "SRS",
    "MOV",
]


def add_era_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot era columns based on season year."""
    era_map = {era_key: (cfg["start"], cfg["end"]) for era_key, cfg in cfg.eras.items()}

    for era_key, (start, end) in era_map.items():
        col = f"era_{era_key}"
        df[col] = ((df["season"] >= start) & (df["season"] <= end)).astype(int)

    return df


def add_season_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag aberrant seasons (lockout, bubble) with season_flag = 1."""
    aberrant = set(cfg.seasons.get("aberrant", []))
    df["season_flag"] = df["season"].isin(aberrant).astype(int)
    return df


def z_score_per_season(
    df: pd.DataFrame,
    cols: Sequence[str],
    season_col: str = "season",
    suffix: str = "_norm",
) -> pd.DataFrame:
    """Z-score each column within each season group.

    Creates new columns named {col}{suffix} (e.g. ORtg_norm).
    Original columns are preserved.

    Args:
        df: DataFrame with a season column and stat columns to normalize.
        cols: Columns to normalize.
        season_col: Name of the season column.
        suffix: Suffix appended to each normalized column name.

    Returns:
        DataFrame with additional _norm columns.
    """
    df = df.copy()
    available_cols = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available_cols)
    if missing:
        logger.debug("Normalization: skipping missing columns: %s", missing)

    for col in available_cols:
        norm_col = f"{col}{suffix}"
        # Coerce to numeric — bball-ref scraper returns some columns as object dtype
        df[col] = pd.to_numeric(df[col], errors="coerce")
        season_means = df.groupby(season_col)[col].transform("mean")
        season_stds = df.groupby(season_col)[col].transform("std")
        # Avoid division by zero in seasons with only 1 team entry
        df[norm_col] = (df[col] - season_means) / season_stds.replace(0.0, 1.0)

    logger.debug(
        "Normalized %d columns across %d seasons", len(available_cols), df[season_col].nunique()
    )
    return df


def normalize_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full era normalization pipeline to a team stats DataFrame.

    Expects df to have a 'season' column plus the stat columns listed in
    NORMALIZE_COLS (or a subset thereof).

    Returns df with added _norm columns, era flags, and season_flag.
    """
    df = z_score_per_season(df, NORMALIZE_COLS)
    df = add_era_flags(df)
    df = add_season_flag(df)
    logger.info(
        "Era normalization complete: %d team-seasons, %d era columns added",
        len(df),
        sum(1 for c in df.columns if c.startswith("era_")),
    )
    return df


def league_averages_by_season(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Compute league average (mean) per season for a set of columns.

    Useful for notebook analysis of era drift.
    """
    available = [c for c in cols if c in df.columns]
    return df.groupby("season")[available].mean().reset_index()
