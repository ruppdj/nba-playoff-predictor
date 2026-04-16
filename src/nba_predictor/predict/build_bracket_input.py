"""Build bracket input files for the current season's playoff prediction.

Produces two files:
  data/predictions/{season}/bracket_teams.csv  — per-team feature store (16 rows)
  data/predictions/{season}/bracket_input.csv  — first-round matchup features (8 rows)

Run once before 'make predict':
    python -m nba_predictor.predict.build_bracket_input --season 2026

Edit BRACKET_2026 below with the actual first-round seedings once the bracket is set.
Seeds are numbered 1 (best) through 8 (worst) within each conference.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2026 NBA Playoff first-round matchups
# (higher_seed_abbrev, lower_seed_abbrev, higher_seed_number, lower_seed_number, conference)
#
# Confirmed seeds via CBS Sports / NBA.com (April 15, 2026).
# East 7/8 and West 8 from play-in (Apr 15-17); most-likely outcomes used:
#   East 7 → ORL (Magic play PHI tonight for 7 seed)
#   East 8 → PHI (76ers expected to beat CHO in 8-seed game Apr 17)
#   West 8 → LAC (Clippers play GSW tonight; winner faces PHO Apr 17 for 8 seed)
# ---------------------------------------------------------------------------
BRACKET_2026 = {
    "East": [
        ("DET", "PHI", 1, 8, "East"),  # E1 Pistons vs E8 76ers
        ("BOS", "ORL", 2, 7, "East"),  # E2 Celtics vs E7 Magic
        ("NYK", "ATL", 3, 6, "East"),  # E3 Knicks  vs E6 Hawks
        ("CLE", "TOR", 4, 5, "East"),  # E4 Cavs    vs E5 Raptors
    ],
    "West": [
        ("OKC", "LAC", 1, 8, "West"),  # W1 Thunder vs W8 Clippers (play-in TBD)
        ("SAS", "POR", 2, 7, "West"),  # W2 Spurs   vs W7 Trail Blazers
        ("DEN", "MIN", 3, 6, "West"),  # W3 Nuggets vs W6 Timberwolves
        ("HOU", "LAL", 4, 5, "West"),  # W4 Rockets vs W5 Lakers
    ],
}

ROUND1_IDS = {
    "East": ["E1v8", "E2v7", "E3v6", "E4v5"],
    "West": ["W1v8", "W2v7", "W3v6", "W4v5"],
}

# Team-level feature columns pulled from each processed parquet
_TEAM_COLS = [
    "Win_pct",
    "NRtg_norm",
    "ORtg_norm",
    "DRtg_norm",
    "Pace_norm",
    "Playoff_experience_years",
    "Prior_playoff_win_pct",
    "Prior_deepest_round",
    "Prior_champion_3yr",
    "Prior_playoff_appearances_2yr",
    "L10_NRtg",
    "L10_NRtg_delta",
    "current_win_streak",
]
_PLAYER_COLS = [
    "team_VORP_sum",
    "team_BPM_weighted_avg",
    "Top3_VORP_sum",
    "Star_player_BPM",
]
_INJURY_COLS = [
    "adj_VORP_sum",
    "Star_injured",
    "Second_star_injured",
    "Lost_top_scorer",
    "Lost_top_rebounder",
    "Lost_top_playmaker",
    "Roster_VORP_available_pct",
    "Injured_player_count",
    "has_injury_data",
]


def _scalar(row: pd.Series, col: str) -> float:
    if row.empty or col not in row.index:
        return np.nan
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return np.nan


def _get_team_row(df: pd.DataFrame, season: int, team: str) -> pd.Series:
    mask = (df["season"] == season) & (df["Team_abbrev"] == team)
    rows = df[mask]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=float)


def build_team_store(season: int, features_season: int) -> pd.DataFrame:
    """Build a per-team feature store for all 16 playoff teams.

    Returns a DataFrame with one row per team and columns:
        team, seed, conference, + all raw feature columns.
    """
    processed = cfg.project_root / "data" / "processed"
    team_features = pd.read_parquet(processed / "team_season_features.parquet")
    player_features = pd.read_parquet(processed / "player_season_features.parquet")
    injury_features = pd.read_parquet(processed / "injury_adjusted.parquet")

    rows = []
    for _conf, matchups in BRACKET_2026.items():
        for higher, lower, hseed, lseed, conference in matchups:
            for team, seed in [(higher, hseed), (lower, lseed)]:
                tf = _get_team_row(team_features, features_season, team)
                pf = _get_team_row(player_features, features_season, team)
                if_ = _get_team_row(injury_features, features_season, team)

                if tf.empty:
                    logger.warning("No team features for %s in season %d", team, features_season)

                row: dict = {"team": team, "seed": seed, "conference": conference}
                for col in _TEAM_COLS:
                    row[col] = _scalar(tf, col)
                for col in _PLAYER_COLS:
                    row[col] = _scalar(pf, col)
                for col in _INJURY_COLS:
                    row[col] = _scalar(if_, col)
                rows.append(row)

    return pd.DataFrame(rows)


def _build_matchup_row(
    higher: str,
    lower: str,
    seed_diff: float,
    conference: str,
    matchup_id: str,
    round_name: str,
    season: int,
    team_store: pd.DataFrame,
) -> dict:
    """Build one matchup feature row from the team store."""

    def get(team: str, col: str) -> float:
        rows = team_store[team_store["team"] == team]
        if rows.empty or col not in rows.columns:
            return np.nan
        return float(rows.iloc[0][col])

    row: dict = {
        "matchup_id": matchup_id,
        "season": season,
        "round": round_name,
        "conference": conference,
        "higher_seed": higher,
        "lower_seed": lower,
        "seed_diff": seed_diff,
        "home_court_advantage": 1,
        "conference_East": int(conference == "East"),
        "conference_West": int(conference == "West"),
        "era_showtime": 0,
        "era_defensive": 0,
        "era_transition": 0,
        "era_analytics": 1,
        "season_flag": int(season in set(cfg.seasons.get("aberrant", []))),
    }

    delta_cols = [
        ("NRtg_norm", "delta_NRtg"),
        ("ORtg_norm", "delta_ORtg"),
        ("DRtg_norm", "delta_DRtg"),
        ("Pace_norm", "delta_Pace"),
        ("Win_pct", "delta_Win_pct"),
        ("Playoff_experience_years", "delta_Experience"),
        ("Prior_playoff_win_pct", "delta_Prior_playoff_win_pct"),
        ("Prior_deepest_round", "delta_Prior_deepest_round"),
        ("Prior_playoff_appearances_2yr", "delta_Recent_appearances"),
        ("L10_NRtg", "delta_L10_NRtg"),
        ("L10_NRtg_delta", "delta_L10_NRtg_trend"),
        ("current_win_streak", "delta_streak"),
        ("team_VORP_sum", "delta_VORP"),
        ("team_BPM_weighted_avg", "delta_BPM"),
        ("Top3_VORP_sum", "delta_Top3_VORP"),
        ("Star_player_BPM", "delta_Star_BPM"),
        ("adj_VORP_sum", "delta_adj_VORP"),
        ("Roster_VORP_available_pct", "delta_roster_health"),
    ]
    for src, dst in delta_cols:
        h_val = get(higher, src)
        l_val = get(lower, src)
        row[dst] = h_val - l_val if not (np.isnan(h_val) or np.isnan(l_val)) else np.nan

    for col in ["Prior_champion_3yr"]:
        row[f"higher_seed_{col}"] = get(higher, col)
        row[f"lower_seed_{col}"] = get(lower, col)

    for col in _INJURY_COLS:
        row[f"higher_{col}"] = get(higher, col)
        row[f"lower_{col}"] = get(lower, col)

    row["H2H_win_pct"] = np.nan
    row["H2H_NRtg_avg"] = np.nan
    row["H2H_games_played"] = 0
    row["Competitive_balance_index"] = np.nan

    return row


def build_bracket_input(season: int, features_season: int) -> pd.DataFrame:
    """Build first-round bracket_input DataFrame."""
    team_store = build_team_store(season, features_season)

    rows = []
    for conf, matchups in BRACKET_2026.items():
        labels = ROUND1_IDS[conf]
        for (higher, lower, hseed, lseed, conference), matchup_id in zip(
            matchups, labels, strict=False
        ):
            seed_diff = float(lseed - hseed)
            row = _build_matchup_row(
                higher,
                lower,
                seed_diff,
                conference,
                matchup_id,
                "first_round",
                season,
                team_store,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Build bracket input files for playoff prediction")
    parser.add_argument("--season", type=int, default=cfg.seasons["current"])
    parser.add_argument(
        "--features-season",
        type=int,
        default=None,
        help="Season features to use (default: most recent available)",
    )
    args = parser.parse_args()

    processed = cfg.project_root / "data" / "processed"
    tf = pd.read_parquet(processed / "team_season_features.parquet")
    available = sorted(tf["season"].unique())
    features_season = args.features_season
    if features_season is None:
        features_season = args.season if args.season in available else max(available)
    logger.info("Using %d season features for %d playoff predictions", features_season, args.season)

    out_dir = cfg.project_root / "data" / "predictions" / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)

    team_store = build_team_store(args.season, features_season)
    teams_path = out_dir / "bracket_teams.csv"
    team_store.to_csv(teams_path, index=False)
    logger.info("Saved team store: %s (%d teams)", teams_path, len(team_store))

    bracket_df = build_bracket_input(args.season, features_season)
    bracket_path = out_dir / "bracket_input.csv"
    bracket_df.to_csv(bracket_path, index=False)
    logger.info("Saved bracket input: %s (%d matchups)", bracket_path, len(bracket_df))

    print("\nTeam strengths (delta_NRtg is higher_seed advantage):")
    print(
        bracket_df[
            ["matchup_id", "higher_seed", "lower_seed", "seed_diff", "delta_NRtg", "delta_VORP"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
