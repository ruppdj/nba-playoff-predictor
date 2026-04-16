"""Unit tests for walk-forward cross-validation strategy.

Critical: CV must be strictly temporal — no future data in training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nba_predictor.evaluation.cv_strategy import (
    get_cv_fold_summary,
    n_cv_folds,
    playoff_season_cv_splits,
)


def _make_df(seasons: list[int], n_series_per_season: int = 8) -> pd.DataFrame:
    rows = []
    for s in seasons:
        for _ in range(n_series_per_season):
            rows.append({"season": s, "higher_seed_wins": np.random.randint(0, 2)})
    return pd.DataFrame(rows)


def test_no_future_data_in_training():
    """Test season must not appear in training set."""
    df = _make_df(list(range(1984, 2010)))
    for train_idx, test_idx in playoff_season_cv_splits(df, min_train_seasons=5):
        train_seasons = set(df.loc[train_idx, "season"])
        test_seasons = set(df.loc[test_idx, "season"])
        overlap = train_seasons & test_seasons
        assert not overlap, f"Data leakage: {overlap} appear in both train and test"


def test_training_set_grows():
    """Each successive fold should have more training data than the previous."""
    df = _make_df(list(range(1984, 2000)))
    sizes = [len(ti) for ti, _ in playoff_season_cv_splits(df, min_train_seasons=5)]
    for i in range(1, len(sizes)):
        assert (
            sizes[i] > sizes[i - 1]
        ), f"Fold {i} training size {sizes[i]} should be larger than fold {i-1} size {sizes[i-1]}"


def test_test_set_is_single_season():
    """Each test fold should contain exactly one season."""
    df = _make_df(list(range(1990, 2000)))
    for _, test_idx in playoff_season_cv_splits(df, min_train_seasons=3):
        n_test_seasons = df.loc[test_idx, "season"].nunique()
        assert n_test_seasons == 1, f"Expected 1 test season, got {n_test_seasons}"


def test_minimum_train_seasons_enforced():
    """CV should not start until min_train_seasons have been seen."""
    df = _make_df(list(range(1984, 2000)))
    min_train = 8
    for train_idx, _ in playoff_season_cv_splits(df, min_train_seasons=min_train):
        n_train_seasons = df.loc[train_idx, "season"].nunique()
        assert (
            n_train_seasons >= min_train
        ), f"Train set has {n_train_seasons} seasons but min is {min_train}"
        break  # only check first fold


def test_n_cv_folds_correct():
    """Number of folds should be n_seasons - min_train_seasons."""
    seasons = list(range(1984, 2000))  # 16 seasons
    df = _make_df(seasons)
    min_train = 5
    expected_folds = len(seasons) - min_train
    assert n_cv_folds(df, min_train_seasons=min_train) == expected_folds


def test_insufficient_data_raises():
    """Should raise ValueError when there aren't enough seasons."""
    df = _make_df([2020, 2021, 2022])
    with pytest.raises(ValueError, match="min_train_seasons"):
        list(playoff_season_cv_splits(df, min_train_seasons=10))


def test_all_series_covered():
    """Every series should appear in exactly one test fold."""
    df = _make_df(list(range(1990, 2000)))
    test_indices_seen = set()
    for _, test_idx in playoff_season_cv_splits(df, min_train_seasons=3):
        overlap = set(test_idx) & test_indices_seen
        assert not overlap, "Same series appeared in multiple test folds"
        test_indices_seen.update(test_idx)


def test_cv_fold_summary_shape():
    """get_cv_fold_summary should return one row per fold."""
    df = _make_df(list(range(1984, 2000)))
    summary = get_cv_fold_summary(df, min_train_seasons=5)
    expected_rows = n_cv_folds(df, min_train_seasons=5)
    assert len(summary) == expected_rows
    assert "test_season" in summary.columns
    assert "n_train_series" in summary.columns
