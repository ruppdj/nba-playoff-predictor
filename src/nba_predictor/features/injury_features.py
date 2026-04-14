"""Injury-adjusted availability features.

Computes both simple availability weights and role-specific injury impact:

Simple weights:
  - adj_VORP_sum: VORP weighted by player availability probability
  - Star_injured, Second_star_injured: binary flags

Role-specific (the key insight — losing your scorer ≠ losing your rebounder):
  - Lost_top_scorer, Lost_top2_scorers, Scoring_VORP_available_pct
  - Lost_top_rebounder, Rebounding_capacity_pct
  - Lost_top_playmaker
  - Top_Defender_available
  - Roster_VORP_available_pct, Injured_player_count

Availability status → probability mapping (from config):
  Out: 0.0, Doubtful: 0.15, Questionable: 0.50, Probable: 0.85, Available: 1.0

For seasons without structured injury data (pre-2010):
  - Availability is estimated from GP/82 ratio (player games played / 82)
  - has_injury_data = 0 flag marks these rows
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

# Availability probability weights from config
_AW = cfg.modeling["injury_availability_thresholds"]
AVAILABILITY_MAP: dict[str, float] = {
    "out": _AW["out"],
    "out (dtd)": _AW["out"],
    "doubtful": _AW["doubtful"],
    "questionable": _AW["questionable"],
    "probable": _AW["probable"],
    "available": _AW["available"],
    "active": _AW["available"],
}

INJURED_THRESHOLD = _AW["injured_threshold"]  # availability < this = "injured" for binary flags
NBA_GAMES_PER_SEASON = 82


def status_to_availability(status: str) -> float:
    """Convert injury report status string to availability probability."""
    return AVAILABILITY_MAP.get(status.lower().strip(), _AW["available"])


def availability_from_gp_ratio(gp: float, total_games: float = NBA_GAMES_PER_SEASON) -> float:
    """Estimate availability from games played ratio (proxy for pre-2010 eras)."""
    if total_games <= 0:
        return 1.0
    return min(1.0, max(0.0, gp / total_games))


def compute_injury_features(
    roster_df: pd.DataFrame,
    injury_report: pd.DataFrame | None,
    player_advanced: pd.DataFrame,
) -> pd.DataFrame:
    """Compute full injury feature set for each team in a playoff series.

    Args:
        roster_df: DataFrame with columns [season, Team_abbrev, PLAYER_NAME,
                   PLAYER_ID] — one row per player per team.
        injury_report: DataFrame with columns [season, TEAM_ABBREV, PLAYER_NAME,
                       STATUS] or None. If None, falls back to GP-based estimates.
        player_advanced: DataFrame with player season stats including
                         VORP, BPM, PTS (PPG), TRB (RPG), AST (APG), DBPM.

    Returns:
        DataFrame with one row per team-season with all injury features.
    """
    rows = []

    for (season, team), players in roster_df.groupby(["season", "Team_abbrev"]):
        has_injury_data = injury_report is not None and not injury_report.empty

        # Get player-level stats for this team
        pa = player_advanced[
            (player_advanced["season"] == season)
            & (player_advanced.get("Team_abbrev", pd.Series(dtype=str)) == team)
        ].copy()

        for col in ["VORP", "BPM", "PTS", "TRB", "AST", "DBPM", "G", "MP"]:
            if col in pa.columns:
                pa[col] = pd.to_numeric(pa[col], errors="coerce")

        if pa.empty:
            rows.append(_empty_injury_row(season, team))
            continue

        # Build availability weights per player
        availabilities: dict[str, float] = {}
        for _, prow in pa.iterrows():
            player_name = prow.get("Player", "")
            if has_injury_data:
                # Look up player in injury report
                match = injury_report[
                    (injury_report["season"] == season)
                    & (injury_report["TEAM_ABBREV"] == team)
                    & (injury_report["PLAYER_NAME"].str.lower() == str(player_name).lower())
                ]
                if not match.empty:
                    status = match.iloc[0]["STATUS"]
                    availabilities[player_name] = status_to_availability(status)
                else:
                    availabilities[player_name] = _AW["available"]
            else:
                # Fallback: estimate from games played ratio
                gp = prow.get("G", NBA_GAMES_PER_SEASON)
                availabilities[player_name] = availability_from_gp_ratio(float(gp))

        pa["availability"] = pa["Player"].map(availabilities).fillna(_AW["available"])

        # ── Basic availability features ──────────────────────────────────────
        vorp_avail = pa.get("VORP", pd.Series(dtype=float))
        adj_vorp_sum = (pa["VORP"] * pa["availability"]).sum() if "VORP" in pa.columns else np.nan

        # Sort by VORP for star identification
        pa_sorted_vorp = pa.sort_values("VORP", ascending=False)
        star_avail = pa_sorted_vorp.iloc[0]["availability"] if len(pa_sorted_vorp) > 0 else 1.0
        second_star_avail = pa_sorted_vorp.iloc[1]["availability"] if len(pa_sorted_vorp) > 1 else 1.0

        star_injured = int(star_avail < INJURED_THRESHOLD)
        second_star_injured = int(second_star_avail < INJURED_THRESHOLD)

        # ── Role-specific injury features ────────────────────────────────────

        # Scoring: ranked by PPG
        if "PTS" in pa.columns:
            pa_pts = pa.sort_values("PTS", ascending=False)
            top1_scorer_avail = pa_pts.iloc[0]["availability"] if len(pa_pts) > 0 else 1.0
            top2_scorer_avail = (
                pa_pts.iloc[1]["availability"] if len(pa_pts) > 1 else 1.0
            )
            top2_scorer_avg_avail = np.mean([top1_scorer_avail, top2_scorer_avail])
            lost_top_scorer = int(top1_scorer_avail < INJURED_THRESHOLD)
            lost_top2_scorers = int(
                top1_scorer_avail < INJURED_THRESHOLD and top2_scorer_avail < INJURED_THRESHOLD
            )
            # Scoring VORP available pct
            scoring_vorp_total = pa["VORP"].sum() if "VORP" in pa.columns else np.nan
            scoring_vorp_avail = (pa["VORP"] * pa["availability"]).sum() if "VORP" in pa.columns else np.nan
            scoring_vorp_pct = (
                scoring_vorp_avail / scoring_vorp_total
                if scoring_vorp_total and scoring_vorp_total > 0
                else np.nan
            )
        else:
            top2_scorer_avg_avail = np.nan
            lost_top_scorer = 0
            lost_top2_scorers = 0
            scoring_vorp_pct = np.nan

        # Rebounding: ranked by RPG
        if "TRB" in pa.columns:
            pa_reb = pa.sort_values("TRB", ascending=False)
            top_rebounder_avail = pa_reb.iloc[0]["availability"] if len(pa_reb) > 0 else 1.0
            top_rebounder_available = int(top_rebounder_avail >= INJURED_THRESHOLD)
            lost_top_rebounder = int(top_rebounder_avail < INJURED_THRESHOLD)

            # Rebounding capacity pct: weighted DRB availability proxy
            if "DRB" in pa.columns:
                pa["DRB"] = pd.to_numeric(pa["DRB"], errors="coerce")
                drb_total = pa["DRB"].sum()
                drb_avail = (pa["DRB"] * pa["availability"]).sum()
                rebounding_capacity_pct = drb_avail / drb_total if drb_total > 0 else np.nan
            else:
                rebounding_capacity_pct = np.nan
        else:
            top_rebounder_available = 1
            lost_top_rebounder = 0
            rebounding_capacity_pct = np.nan

        # Playmaking: ranked by APG
        if "AST" in pa.columns:
            pa_ast = pa.sort_values("AST", ascending=False)
            top_playmaker_avail = pa_ast.iloc[0]["availability"] if len(pa_ast) > 0 else 1.0
            top_playmaker_available = int(top_playmaker_avail >= INJURED_THRESHOLD)
            lost_top_playmaker = int(top_playmaker_avail < INJURED_THRESHOLD)
        else:
            top_playmaker_available = 1
            lost_top_playmaker = 0

        # Defense: ranked by DBPM
        if "DBPM" in pa.columns:
            pa_def = pa.sort_values("DBPM", ascending=False)
            top_defender_avail = pa_def.iloc[0]["availability"] if len(pa_def) > 0 else 1.0
            top_defender_available = int(top_defender_avail >= INJURED_THRESHOLD)
        else:
            top_defender_available = 1

        # Aggregate roster availability
        if "VORP" in pa.columns:
            vorp_total = pa["VORP"].sum()
            roster_vorp_avail_pct = (
                (pa["VORP"] * pa["availability"]).sum() / vorp_total
                if vorp_total > 0
                else np.nan
            )
        else:
            roster_vorp_avail_pct = np.nan

        injured_count = int((pa["availability"] < INJURED_THRESHOLD).sum())

        rows.append(
            {
                "season": season,
                "Team_abbrev": team,
                "has_injury_data": int(has_injury_data),
                "adj_VORP_sum": adj_vorp_sum,
                "Star_injured": star_injured,
                "Second_star_injured": second_star_injured,
                "Top2_Scorer_availability_avg": top2_scorer_avg_avail,
                "Lost_top_scorer": lost_top_scorer,
                "Lost_top2_scorers": lost_top2_scorers,
                "Scoring_VORP_available_pct": scoring_vorp_pct,
                "Top_Rebounder_available": top_rebounder_available,
                "Lost_top_rebounder": lost_top_rebounder,
                "Rebounding_capacity_pct": rebounding_capacity_pct,
                "Top_Playmaker_available": top_playmaker_available,
                "Lost_top_playmaker": lost_top_playmaker,
                "Top_Defender_available": top_defender_available,
                "Roster_VORP_available_pct": roster_vorp_avail_pct,
                "Injured_player_count": injured_count,
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Injury features computed for %d team-seasons", len(result))
    return result


def _empty_injury_row(season: int, team: str) -> dict:
    """Return a row of NaN injury features for teams with no data."""
    return {
        "season": season,
        "Team_abbrev": team,
        "has_injury_data": 0,
        "adj_VORP_sum": np.nan,
        "Star_injured": np.nan,
        "Second_star_injured": np.nan,
        "Top2_Scorer_availability_avg": np.nan,
        "Lost_top_scorer": np.nan,
        "Lost_top2_scorers": np.nan,
        "Scoring_VORP_available_pct": np.nan,
        "Top_Rebounder_available": np.nan,
        "Lost_top_rebounder": np.nan,
        "Rebounding_capacity_pct": np.nan,
        "Top_Playmaker_available": np.nan,
        "Lost_top_playmaker": np.nan,
        "Top_Defender_available": np.nan,
        "Roster_VORP_available_pct": np.nan,
        "Injured_player_count": np.nan,
    }
