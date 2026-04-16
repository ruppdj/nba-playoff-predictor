"""Walk-forward cross-validation by season.

CRITICAL: NBA playoff data is temporal. Using standard k-fold would leak
future seasons into the training set, producing optimistic and invalid
performance estimates.

This module implements a walk-forward (expanding window) CV where:
  - Training set: all seasons strictly before the test season
  - Test set: a single season's worth of playoff series
  - Minimum training window enforced to avoid tiny early folds

Example splits (min_train_seasons=10, start=1984):
  Fold 1: Train 1984–1993, Test 1994
  Fold 2: Train 1984–1994, Test 1995
  ...
  Fold N: Train 1984–2024, Test 2025
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pandas as pd

from nba_predictor.config import cfg

MIN_TRAIN_SEASONS = cfg.modeling["min_train_seasons"]


def playoff_season_cv_splits(
    df: pd.DataFrame,
    season_col: str = "season",
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (train_idx, test_idx) arrays for walk-forward CV by season.

    Args:
        df: DataFrame containing a season column.
        season_col: Name of the season column.
        min_train_seasons: Minimum number of training seasons before first test.

    Yields:
        (train_indices, test_indices) as numpy integer arrays.
    """
    seasons = sorted(df[season_col].unique())
    n = len(seasons)

    if n <= min_train_seasons:
        raise ValueError(
            f"Dataset has only {n} seasons but min_train_seasons={min_train_seasons}. "
            "Need more seasons to perform walk-forward CV."
        )

    for i in range(min_train_seasons, n):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_idx = df.index[df[season_col].isin(train_seasons)].to_numpy()
        test_idx = df.index[df[season_col] == test_season].to_numpy()

        if len(test_idx) == 0:
            continue

        yield train_idx, test_idx


def n_cv_folds(
    df: pd.DataFrame,
    season_col: str = "season",
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> int:
    """Return the number of CV folds that will be generated."""
    seasons = sorted(df[season_col].unique())
    return max(0, len(seasons) - min_train_seasons)


def get_cv_fold_summary(
    df: pd.DataFrame,
    season_col: str = "season",
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> pd.DataFrame:
    """Return a summary DataFrame describing each CV fold.

    Useful for notebook inspection of the CV structure.
    """
    rows = []
    for train_idx, test_idx in playoff_season_cv_splits(df, season_col, min_train_seasons):
        train_seasons = sorted(df.loc[train_idx, season_col].unique())
        test_season = df.loc[test_idx, season_col].iloc[0]
        rows.append(
            {
                "train_start": train_seasons[0],
                "train_end": train_seasons[-1],
                "n_train_seasons": len(train_seasons),
                "n_train_series": len(train_idx),
                "test_season": test_season,
                "n_test_series": len(test_idx),
            }
        )
    return pd.DataFrame(rows)
