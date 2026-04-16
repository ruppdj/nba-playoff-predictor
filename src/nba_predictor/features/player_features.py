"""Player-level feature aggregation.

Aggregates player advanced stats to the team level (minutes-weighted).
Also computes player momentum from last-10-game logs.

Key aggregated features:
  - team_VORP_sum: total team VORP (additive — designed to be summed)
  - team_BPM_weighted_avg: minutes-weighted average BPM
  - team_WS48_weighted_avg: minutes-weighted WS/48
  - Top3_VORP_sum: top-3 players by BPM (star power)
  - Star_player_BPM: single best player's BPM
  - Has_AllNBA_player: any player with BPM > threshold
  - Top8_WS48_avg: rotation depth (top 8 by minutes)
  - Guard_VORP, Forward_VORP, Center_VORP: positional balance
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

ALLNBA_BPM_THRESHOLD = cfg.modeling["player_tiers"]["allnba_bpm_threshold"]
ROTATION_TOP_N = cfg.modeling["player_tiers"]["rotation_top_n"]


def _minutes_weight(player_df: pd.DataFrame) -> pd.Series:
    """Compute fractional minutes weight for each player on their team."""
    total_mp = player_df.groupby(["season", "Team_abbrev"])["MP"].transform("sum")
    return player_df["MP"] / total_mp.replace(0, np.nan)


def aggregate_player_to_team(
    player_advanced: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate player advanced stats to team-season level.

    Expects player_advanced to have columns:
      season, Team_abbrev, Player, MP, BPM, VORP, WS/48, PER, Pos
    """
    pa = player_advanced.copy()

    # Convert key columns to numeric
    for col in ["MP", "BPM", "VORP", "WS/48", "PER", "G"]:
        if col in pa.columns:
            pa[col] = pd.to_numeric(pa[col], errors="coerce")

    # Handle players who were traded mid-season (multiple rows, "TOT" row)
    # Keep the TOT row when present (represents the full season)
    pa = pa[pa["Team"] != "TOT"].copy()  # drop individual stints
    # Re-add TOT rows separately if they exist
    tot = player_advanced[player_advanced["Team"] == "TOT"].copy()
    pa = pd.concat([pa, tot], ignore_index=True)

    pa["MP_weight"] = _minutes_weight(pa)

    def _agg_team(group: pd.DataFrame) -> pd.Series:
        # Sort by MP descending for top-N slicing
        g = group.sort_values("MP", ascending=False)

        top8 = g.head(ROTATION_TOP_N)
        top3 = g.head(3)
        star = g.head(1)

        # Minutes-weighted averages
        bpm_wavg = (g["BPM"] * g["MP_weight"]).sum() if "BPM" in g.columns else np.nan

        ws48_wavg = (g["WS/48"] * g["MP_weight"]).sum() if "WS/48" in g.columns else np.nan
        vorp_sum = g["VORP"].sum() if "VORP" in g.columns else np.nan
        top3_vorp = top3["VORP"].sum() if "VORP" in top3.columns else np.nan
        star_bpm = star["BPM"].iloc[0] if not star.empty and "BPM" in star.columns else np.nan
        has_allnba = int((g["BPM"] > ALLNBA_BPM_THRESHOLD).any()) if "BPM" in g.columns else 0
        top8_ws48 = top8["WS/48"].mean() if "WS/48" in top8.columns else np.nan

        # Positional VORP (Guard, Forward, Center)
        def _pos_vorp(pos_filter: str) -> float:
            subset = (
                g[g["Pos"].str.contains(pos_filter, na=False)]
                if "Pos" in g.columns
                else pd.DataFrame()
            )
            return subset["VORP"].sum() if not subset.empty and "VORP" in subset.columns else np.nan

        return pd.Series(
            {
                "team_VORP_sum": vorp_sum,
                "team_BPM_weighted_avg": bpm_wavg,
                "team_WS48_weighted_avg": ws48_wavg,
                "Top3_VORP_sum": top3_vorp,
                "Star_player_BPM": star_bpm,
                "Has_AllNBA_player": has_allnba,
                "Top8_WS48_avg": top8_ws48,
                "Guard_VORP": _pos_vorp("G"),
                "Forward_VORP": _pos_vorp("F"),
                "Center_VORP": _pos_vorp("C"),
            }
        )

    team_player = (
        pa.groupby(["season", "Team_abbrev"]).apply(_agg_team, include_groups=False).reset_index()
    )

    logger.info("Player aggregation complete: %d team-seasons", len(team_player))
    return team_player


def compute_player_momentum(
    player_game_logs: pd.DataFrame,
    player_advanced: pd.DataFrame,
    top_n_players: int = 5,
    window: int = 10,
) -> pd.DataFrame:
    """Compute last-N-game momentum features for the top players on each team.

    Returns one row per team-season with:
      Star_PTS_L10_delta, Star_TS_pct_L10_delta,
      Top5_PTS_L10_delta_avg, Top5_MP_L10_avg,
      Star_GP_L10, Top5_GP_L10_avg
    """
    if player_game_logs.empty:
        logger.warning("Player game logs empty — returning empty momentum DataFrame.")
        return pd.DataFrame(
            columns=[
                "season",
                "Team_abbrev",
                "Star_PTS_L10_delta",
                "Star_TS_pct_L10_delta",
                "Top5_PTS_L10_delta_avg",
                "Top5_MP_L10_avg",
                "Star_GP_L10",
                "Top5_GP_L10_avg",
            ]
        )

    logs = player_game_logs.copy()
    if "GAME_DATE" in logs.columns:
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
        logs = logs.sort_values(["PLAYER_ID", "season", "GAME_DATE"])

    # Convert numeric columns
    for col in ["PTS", "MIN", "FGA", "FTA", "FT_PCT"]:
        if col in logs.columns:
            logs[col] = pd.to_numeric(logs[col], errors="coerce")

    # Identify top-N players per team by season minutes
    pa = player_advanced.copy()
    for col in ["MP", "BPM", "VORP"]:
        if col in pa.columns:
            pa[col] = pd.to_numeric(pa[col], errors="coerce")

    top_players = (
        pa[pa["Team"] != "TOT"]
        .sort_values("MP", ascending=False)
        .groupby(["season", "Team_abbrev"])
        .head(top_n_players)[["season", "Team_abbrev", "Player", "MP", "BPM"]]
    )

    rows = []
    for (season, team), players in top_players.groupby(["season", "Team_abbrev"]):
        star_pid = (
            players.sort_values("BPM", ascending=False)["Player"].iloc[0] if len(players) else None
        )

        team_metrics: dict = {"season": season, "Team_abbrev": team}
        star_pts_delta = np.nan
        star_ts_delta = np.nan
        star_gp = np.nan
        top5_pts_deltas = []
        top5_mp_last = []
        top5_gp = []

        for _, prow in players.iterrows():
            player_name = prow["Player"]
            # Match game log by player ID — requires player name match or ID join
            player_logs = logs[
                (logs["season"] == season)
                & (logs.get("PLAYER_NAME", logs.get("Player", pd.Series(dtype=str))) == player_name)
            ]
            if player_logs.empty:
                continue

            last_n = player_logs.tail(window)
            season_pts_avg = player_logs["PTS"].mean() if "PTS" in player_logs.columns else np.nan
            last_n_pts_avg = last_n["PTS"].mean() if "PTS" in last_n.columns else np.nan
            pts_delta = last_n_pts_avg - season_pts_avg

            # TS% proxy: PTS / (2 * (FGA + 0.44 * FTA))
            def _ts_pct(df: pd.DataFrame) -> float:
                pts = df["PTS"].sum() if "PTS" in df.columns else 0
                fga = df["FGA"].sum() if "FGA" in df.columns else 0
                fta = df["FTA"].sum() if "FTA" in df.columns else 0
                denom = 2 * (fga + 0.44 * fta)
                return pts / denom if denom > 0 else np.nan

            season_ts = _ts_pct(player_logs)
            last_n_ts = _ts_pct(last_n)
            ts_delta = last_n_ts - season_ts if not np.isnan(season_ts) else np.nan

            gp_last = len(last_n)
            mp_last = last_n["MIN"].mean() if "MIN" in last_n.columns else np.nan

            if player_name == star_pid:
                star_pts_delta = pts_delta
                star_ts_delta = ts_delta
                star_gp = gp_last

            top5_pts_deltas.append(pts_delta)
            top5_mp_last.append(mp_last)
            top5_gp.append(gp_last)

        team_metrics["Star_PTS_L10_delta"] = star_pts_delta
        team_metrics["Star_TS_pct_L10_delta"] = star_ts_delta
        team_metrics["Star_GP_L10"] = star_gp
        team_metrics["Top5_PTS_L10_delta_avg"] = (
            np.nanmean(top5_pts_deltas) if top5_pts_deltas else np.nan
        )
        team_metrics["Top5_MP_L10_avg"] = np.nanmean(top5_mp_last) if top5_mp_last else np.nan
        team_metrics["Top5_GP_L10_avg"] = np.nanmean(top5_gp) if top5_gp else np.nan

        rows.append(team_metrics)

    result = pd.DataFrame(rows)
    logger.info("Player momentum features computed: %d team-seasons", len(result))
    return result
