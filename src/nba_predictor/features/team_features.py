"""Team-level feature computation.

Computes season-level team features including:
  - Offensive / defensive / combined metrics (normalized)
  - Win percentage and momentum (last 10/20 games)
  - Playoff experience and coaching experience
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nba_predictor.config import cfg
from nba_predictor.features.era_normalizer import normalize_team_stats

logger = logging.getLogger(__name__)

# Momentum window sizes (games before playoffs to look back)
MOMENTUM_WINDOWS = cfg.modeling["momentum"]["last_n_games"]  # [10, 20]


def _safe_normalize_team(name: str) -> str:
    try:
        return cfg.normalize_team(name)
    except KeyError:
        return name


def compute_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Add Win_pct column from W and L columns."""
    df = df.copy()
    if "W" in df.columns and "L" in df.columns:
        df["W"] = pd.to_numeric(df["W"], errors="coerce")
        df["L"] = pd.to_numeric(df["L"], errors="coerce")
        df["Win_pct"] = df["W"] / (df["W"] + df["L"]).replace(0, np.nan)
    return df


def _round_depth(round_name: str) -> int:
    """Convert a round name string to a numeric depth value (1–4).

    Actual values in data: "Eastern/Western Conference First Round",
    "Eastern/Western Conference Semifinals", "Eastern/Western Conference Finals",
    "Finals" (the championship round — no conference prefix).

    1 = First Round, 2 = Conference Semifinals, 3 = Conference Finals,
    4 = NBA Finals appearance (loser). Champion (Finals winner) gets 5 separately.
    """
    r = round_name.lower()
    if "first round" in r:
        return 1
    if "semifinal" in r:
        return 2
    if "conference finals" in r:
        return 3
    # Plain "Finals" (no "conference" prefix) is the championship round
    if "finals" in r:
        return 4
    return 0


def compute_playoff_experience(
    team_stats: pd.DataFrame, playoff_series: pd.DataFrame
) -> pd.DataFrame:
    """Add playoff experience and history features to team stats.

    Features added:
      Playoff_experience_years   — appearances in prior 5 seasons
      Prior_playoff_win_pct      — series W% over prior 3 seasons
      Prior_deepest_round        — max round reached across last 2 appearances
                                   (0=none, 1=1st rd, 2=semis, 3=conf finals,
                                    4=finals loss, 5=champion)
      Prior_champion_3yr         — 1 if won championship in any of prior 3 seasons
      Prior_playoff_appearances_2yr — appearances in prior 2 seasons (0, 1, or 2)
    """
    team_stats = team_stats.copy()

    # Build set of playoff teams by season
    playoff_teams: dict[int, set[str]] = {}
    for _, row in playoff_series.iterrows():
        season = int(row["season"])
        if season not in playoff_teams:
            playoff_teams[season] = set()
        playoff_teams[season].add(row["team_a"])
        playoff_teams[season].add(row["team_b"])

    # Series win counts per team per season (for Prior_playoff_win_pct)
    playoff_wins: dict[tuple[int, str], int] = {}
    playoff_games: dict[tuple[int, str], int] = {}

    # Max round depth per team per season (for Prior_deepest_round / Prior_champion_3yr)
    # Value 5 = champion (won NBA Finals)
    round_depth: dict[tuple[int, str], int] = {}

    for _, row in playoff_series.iterrows():
        season = int(row["season"])
        winner = str(row["series_winner"])
        loser = str(row["team_b"]) if winner == str(row["team_a"]) else str(row["team_a"])
        round_name = str(row.get("round", ""))

        # Series win tracking
        playoff_wins[(season, winner)] = playoff_wins.get((season, winner), 0) + 1
        playoff_games[(season, winner)] = playoff_games.get((season, winner), 0) + 1
        playoff_games[(season, loser)] = playoff_games.get((season, loser), 0) + 1

        # Round depth tracking — champion gets 5, Finals loser gets 4, others equal
        depth = _round_depth(round_name)
        winner_depth = 5 if depth == 4 else depth  # NBA Finals winner = champion
        loser_depth = depth

        key_w = (season, winner)
        key_l = (season, loser)
        round_depth[key_w] = max(round_depth.get(key_w, 0), winner_depth)
        round_depth[key_l] = max(round_depth.get(key_l, 0), loser_depth)

    experience_years = []
    prior_win_pct = []
    prior_deepest = []
    prior_champion = []
    apps_2yr = []

    for _, row in team_stats.iterrows():
        season = int(row["season"])
        team = str(row.get("Team_abbrev", ""))

        # Experience: appearances in prior 5 seasons
        exp = sum(1 for s in range(season - 5, season) if team in playoff_teams.get(s, set()))
        experience_years.append(exp)

        # Prior win pct: series W% over prior 3 seasons
        wins = sum(playoff_wins.get((s, team), 0) for s in range(season - 3, season))
        games = sum(playoff_games.get((s, team), 0) for s in range(season - 3, season))
        prior_win_pct.append(wins / games if games > 0 else np.nan)

        # Deepest round across last 2 playoff appearances
        depth_vals = [
            round_depth[(s, team)] for s in range(season - 2, season) if (s, team) in round_depth
        ]
        prior_deepest.append(max(depth_vals) if depth_vals else 0)

        # Champion in any of prior 3 seasons
        champ = int(any(round_depth.get((s, team), 0) == 5 for s in range(season - 3, season)))
        prior_champion.append(champ)

        # Appearances in prior 2 seasons
        apps = sum(1 for s in range(season - 2, season) if team in playoff_teams.get(s, set()))
        apps_2yr.append(apps)

    team_stats["Playoff_experience_years"] = experience_years
    team_stats["Prior_playoff_win_pct"] = prior_win_pct
    team_stats["Prior_deepest_round"] = prior_deepest
    team_stats["Prior_champion_3yr"] = prior_champion
    team_stats["Prior_playoff_appearances_2yr"] = apps_2yr
    return team_stats


def compute_momentum_features(team_stats: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Add last-N-game momentum features to team stats.

    For each team-season, computes:
      - L10_NRtg, L10_Win_pct, L10_NRtg_delta (vs season average)
      - L20_NRtg, L20_Win_pct, L20_NRtg_delta
      - current_win_streak (+ = winning, - = losing)
      - L10_home_win_pct, L10_away_win_pct

    Requires a game_logs DataFrame with columns:
      season, Team_abbrev, GAME_DATE, WL (W/L), PLUS_MINUS (net points)
    """
    if game_logs.empty:
        logger.warning("Game logs empty — skipping momentum features.")
        for window in MOMENTUM_WINDOWS:
            n = window
            for col in [f"L{n}_NRtg", f"L{n}_Win_pct", f"L{n}_NRtg_delta"]:
                team_stats[col] = np.nan
        team_stats["current_win_streak"] = np.nan
        team_stats["L10_home_win_pct"] = np.nan
        team_stats["L10_away_win_pct"] = np.nan
        return team_stats

    team_stats = team_stats.copy()
    game_logs = game_logs.copy()
    game_logs["GAME_DATE"] = pd.to_datetime(game_logs["GAME_DATE"], errors="coerce")

    # Derive Team_abbrev from MATCHUP if not already present ("ATL @ MIA" → "ATL")
    if "Team_abbrev" not in game_logs.columns and "MATCHUP" in game_logs.columns:
        game_logs["Team_abbrev"] = (
            game_logs["MATCHUP"]
            .str.extract(r"^([A-Z]{2,3})", expand=False)
            .map(lambda t: _safe_normalize_team(str(t)) if pd.notna(t) else t)
        )

    if "Team_abbrev" not in game_logs.columns:
        logger.warning("Game logs missing Team_abbrev — skipping momentum features.")
        for window in MOMENTUM_WINDOWS:
            n = window
            for col in [f"L{n}_NRtg", f"L{n}_Win_pct", f"L{n}_NRtg_delta"]:
                team_stats[col] = np.nan
        team_stats["current_win_streak"] = np.nan
        team_stats["L10_home_win_pct"] = np.nan
        team_stats["L10_away_win_pct"] = np.nan
        return team_stats

    game_logs = game_logs.sort_values(["season", "Team_abbrev", "GAME_DATE"])

    # Compute per-game net rating proxy from PLUS_MINUS if available.
    # If absent, derive it as PTS - opponent_PTS using Game_ID: each game
    # has exactly two rows in the combined log (one per team), so we can
    # join each row with the other team's row for the same Game_ID.
    if "PLUS_MINUS" not in game_logs.columns:
        if "Game_ID" in game_logs.columns and "PTS" in game_logs.columns:
            opp_pts = game_logs[["Game_ID", "Team_abbrev", "PTS"]].rename(
                columns={"Team_abbrev": "opp_abbrev", "PTS": "OPP_PTS"}
            )
            game_logs = game_logs.merge(opp_pts, on="Game_ID", how="left")
            # Drop self-join rows (same team)
            same = game_logs["Team_abbrev"] == game_logs["opp_abbrev"]
            game_logs = game_logs[~same].copy()
            game_logs["PLUS_MINUS"] = game_logs["PTS"] - game_logs["OPP_PTS"]
            game_logs = game_logs.drop(columns=["opp_abbrev", "OPP_PTS"])
        else:
            game_logs["PLUS_MINUS"] = np.nan

    def _last_n_stats(group: pd.DataFrame, n: int) -> dict:
        last = group.tail(n)
        wins = (last["WL"] == "W").sum()
        wl_total = len(last)
        win_pct = wins / wl_total if wl_total > 0 else np.nan
        nrtg = last["PLUS_MINUS"].mean() if "PLUS_MINUS" in last else np.nan

        home_games = last[last.get("MATCHUP", pd.Series(dtype=str)).str.contains(r"vs\.", na=False)]
        away_games = last[last.get("MATCHUP", pd.Series(dtype=str)).str.contains(r"@", na=False)]
        home_wpct = (home_games["WL"] == "W").mean() if len(home_games) > 0 else np.nan
        away_wpct = (away_games["WL"] == "W").mean() if len(away_games) > 0 else np.nan

        # Win streak (+ = winning streak, - = losing streak)
        streak = 0
        last_wl = last["WL"].tolist()
        if last_wl:
            current = last_wl[-1]
            for result in reversed(last_wl):
                if result == current:
                    streak += 1 if current == "W" else -1
                else:
                    break

        return {
            "win_pct": win_pct,
            "nrtg": nrtg,
            "home_wpct": home_wpct,
            "away_wpct": away_wpct,
            "streak": streak,
        }

    momentum_rows = []
    for (season, team), group in game_logs.groupby(["season", "Team_abbrev"]):
        row: dict = {"season": season, "Team_abbrev": team}
        for n in MOMENTUM_WINDOWS:
            stats = _last_n_stats(group, n)
            row[f"L{n}_Win_pct"] = stats["win_pct"]
            row[f"L{n}_NRtg"] = stats["nrtg"]
            if n == 10:
                row["L10_home_win_pct"] = stats["home_wpct"]
                row["L10_away_win_pct"] = stats["away_wpct"]
                row["current_win_streak"] = stats["streak"]
        momentum_rows.append(row)

    momentum_df = pd.DataFrame(momentum_rows)

    # Merge momentum back to team_stats
    team_stats = team_stats.merge(momentum_df, on=["season", "Team_abbrev"], how="left")

    # Compute deltas vs. season-average NRtg (from era-normalized data)
    if "NRtg_norm" in team_stats.columns:
        for n in MOMENTUM_WINDOWS:
            season_nrtg = team_stats.groupby("season")["NRtg"].transform("mean")
            if f"L{n}_NRtg" in team_stats.columns:
                team_stats[f"L{n}_NRtg_delta"] = team_stats[f"L{n}_NRtg"] - season_nrtg
    else:
        for n in MOMENTUM_WINDOWS:
            team_stats[f"L{n}_NRtg_delta"] = np.nan

    logger.info("Momentum features added for %d team-seasons", len(team_stats))
    return team_stats


def build_team_season_features(
    bball_ref_team: pd.DataFrame,
    game_logs: pd.DataFrame,
    playoff_series: pd.DataFrame,
) -> pd.DataFrame:
    """Full team feature pipeline.

    Steps:
      1. Compute win percentage
      2. Apply era normalization (z-score per season)
      3. Add playoff and coaching experience
      4. Add momentum features from game logs

    Returns one row per team-season.
    """
    df = compute_win_pct(bball_ref_team)
    df = normalize_team_stats(df)
    df = compute_playoff_experience(df, playoff_series)
    df = compute_momentum_features(df, game_logs)
    logger.info("Team season features built: %d rows, %d columns", len(df), len(df.columns))
    return df
