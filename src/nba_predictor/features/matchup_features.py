"""Matchup feature construction — one row per playoff series.

For each playoff series (team_a vs team_b), computes:
  - Delta features: team_a_stat - team_b_stat for all key metrics
  - Seeding features: seed_diff, home_court_advantage
  - Head-to-head regular season record
  - Target variables: higher_seed_wins (binary), series_length (4-class)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)


def build_matchup_dataset(
    playoff_series: pd.DataFrame,
    team_features: pd.DataFrame,
    player_features: pd.DataFrame,
    injury_features: pd.DataFrame,
    game_logs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full series-level matchup dataset (one row per series).

    Args:
        playoff_series: Rows with season, team_a, team_b, series_winner,
                        series_length, and optionally round, seed_a, seed_b.
        team_features: Team-season stats (one row per team-season).
        player_features: Team-level player aggregates (one row per team-season).
        injury_features: Team-season injury features.
        game_logs: Optional game logs for H2H computation.

    Returns:
        DataFrame with ~50-60 features and two targets:
            higher_seed_wins (int: 0/1)
            series_length (int: 4/5/6/7)
    """
    rows = []

    for _, series in playoff_series.iterrows():
        season = int(series["season"])
        team_a = str(series["team_a"])
        team_b = str(series["team_b"])
        winner = str(series["series_winner"])
        length = int(series["series_length"])

        # Identify higher and lower seed (use team_a as default higher seed)
        # Seeding info may or may not be present in playoff_series
        seed_a = float(series.get("seed_a", 1))
        seed_b = float(series.get("seed_b", 8))
        higher_seed = team_a if seed_a <= seed_b else team_b
        lower_seed = team_b if seed_a <= seed_b else team_a
        seed_diff = abs(seed_b - seed_a)

        # Fetch team features for each side
        feat_a = _get_team_row(team_features, season, team_a)
        feat_b = _get_team_row(team_features, season, team_b)
        pfeat_a = _get_team_row(player_features, season, team_a)
        pfeat_b = _get_team_row(player_features, season, team_b)
        ifeat_a = _get_team_row(injury_features, season, team_a)
        ifeat_b = _get_team_row(injury_features, season, team_b)

        row: dict = {
            "season": season,
            "team_a": team_a,
            "team_b": team_b,
            "higher_seed": higher_seed,
            "lower_seed": lower_seed,
            "series_round": series.get("round", np.nan),
            "series_winner": winner,
            "series_length": length,
            # Target: did the higher seed win?
            "higher_seed_wins": int(winner == higher_seed),
        }

        # ── Seeding features ─────────────────────────────────────────────────
        row["seed_diff"] = seed_diff
        row["home_court_advantage"] = 1  # higher seed always has HCA
        row["conference_East"] = int(series.get("conference", "") == "East")
        row["conference_West"] = int(series.get("conference", "") == "West")
        row["era_showtime"] = int(_get_era(season) == "showtime")
        row["era_defensive"] = int(_get_era(season) == "defensive")
        row["era_transition"] = int(_get_era(season) == "transition")
        row["era_analytics"] = int(_get_era(season) == "analytics")
        row["season_flag"] = int(season in set(cfg.seasons.get("aberrant", [])))

        # ── Team stat deltas (higher_seed - lower_seed) ───────────────────────
        feat_higher = _get_team_row(team_features, season, higher_seed)
        feat_lower = _get_team_row(team_features, season, lower_seed)
        pfeat_higher = _get_team_row(player_features, season, higher_seed)
        pfeat_lower = _get_team_row(player_features, season, lower_seed)
        ifeat_higher = _get_team_row(injury_features, season, higher_seed)
        ifeat_lower = _get_team_row(injury_features, season, lower_seed)

        delta_cols = [
            ("NRtg_norm", "delta_NRtg"),
            ("ORtg_norm", "delta_ORtg"),
            ("DRtg_norm", "delta_DRtg"),
            ("Pace_norm", "delta_Pace"),
            ("Win_pct", "delta_Win_pct"),
            ("Playoff_experience_years", "delta_Experience"),
            ("L10_NRtg", "delta_L10_NRtg"),
            ("L10_NRtg_delta", "delta_L10_NRtg_trend"),
            ("current_win_streak", "delta_streak"),
        ]
        for src, dst in delta_cols:
            h_val = _scalar(feat_higher, src)
            l_val = _scalar(feat_lower, src)
            row[dst] = h_val - l_val if not np.isnan(h_val) and not np.isnan(l_val) else np.nan

        player_delta_cols = [
            ("team_VORP_sum", "delta_VORP"),
            ("team_BPM_weighted_avg", "delta_BPM"),
            ("Top3_VORP_sum", "delta_Top3_VORP"),
            ("Star_player_BPM", "delta_Star_BPM"),
        ]
        for src, dst in player_delta_cols:
            h_val = _scalar(pfeat_higher, src)
            l_val = _scalar(pfeat_lower, src)
            row[dst] = h_val - l_val if not np.isnan(h_val) and not np.isnan(l_val) else np.nan

        injury_delta_cols = [
            ("adj_VORP_sum", "delta_adj_VORP"),
            ("Roster_VORP_available_pct", "delta_roster_health"),
        ]
        for src, dst in injury_delta_cols:
            h_val = _scalar(ifeat_higher, src)
            l_val = _scalar(ifeat_lower, src)
            row[dst] = h_val - l_val if not np.isnan(h_val) and not np.isnan(l_val) else np.nan

        # ── Include absolute injury flags for higher/lower seed ───────────────
        for col in [
            "Star_injured", "Second_star_injured",
            "Lost_top_scorer", "Lost_top_rebounder", "Lost_top_playmaker",
            "Roster_VORP_available_pct", "Injured_player_count",
            "has_injury_data",
        ]:
            row[f"higher_{col}"] = _scalar(ifeat_higher, col)
            row[f"lower_{col}"] = _scalar(ifeat_lower, col)

        # ── Head-to-head regular season record ───────────────────────────────
        if game_logs is not None and not game_logs.empty:
            h2h = _compute_h2h(game_logs, season, higher_seed, lower_seed)
            row.update(h2h)
        else:
            row["H2H_win_pct"] = np.nan
            row["H2H_NRtg_avg"] = np.nan
            row["H2H_games_played"] = 0

        # ── Series length features ────────────────────────────────────────────
        # These depend on win probability — computed post-modeling.
        # Leave placeholder here; filled by bracket_simulator during prediction.
        row["Competitive_balance_index"] = np.nan  # filled post-hoc

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Matchup dataset built: %d series, %d features", len(df), len(df.columns)
    )
    return df


def _get_team_row(df: pd.DataFrame, season: int, team: str) -> pd.Series:
    """Return the row for a team-season from a features DataFrame."""
    mask = (df["season"] == season) & (df["Team_abbrev"] == team)
    rows = df[mask]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.iloc[0]


def _scalar(row: pd.Series, col: str) -> float:
    """Safely get a scalar float value from a Series."""
    if row.empty or col not in row.index:
        return np.nan
    val = row[col]
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _get_era(season: int) -> str:
    return cfg.get_era(season)


def _compute_h2h(
    game_logs: pd.DataFrame, season: int, higher_seed: str, lower_seed: str
) -> dict:
    """Compute head-to-head regular season stats between two teams in a season."""
    logs = game_logs[game_logs["season"] == season].copy()
    if logs.empty or "Team_abbrev" not in logs.columns:
        return {"H2H_win_pct": np.nan, "H2H_NRtg_avg": np.nan, "H2H_games_played": 0}

    # Games where higher_seed hosted lower_seed or vice versa
    h2h_games = logs[
        (
            (logs["Team_abbrev"] == higher_seed)
            & logs.get("OPP_abbrev", pd.Series(dtype=str)).eq(lower_seed)
        )
    ]

    n_games = len(h2h_games)
    if n_games == 0:
        return {"H2H_win_pct": np.nan, "H2H_NRtg_avg": np.nan, "H2H_games_played": 0}

    wins = (h2h_games.get("WL", pd.Series(dtype=str)) == "W").sum()
    nrtg_avg = h2h_games.get("PLUS_MINUS", pd.Series(dtype=float)).mean()

    return {
        "H2H_win_pct": wins / n_games,
        "H2H_NRtg_avg": nrtg_avg,
        "H2H_games_played": n_games,
    }
