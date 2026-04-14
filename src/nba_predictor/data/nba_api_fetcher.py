"""NBA Stats API fetcher — secondary data source (1996–2025).

Fetches:
  - Team season advanced stats (OFF_RATING, DEF_RATING, NET_RATING, PACE)
  - Player season advanced stats
  - Team game logs (for momentum features)
  - Player game logs (for player momentum features)
  - Common team rosters (player list per team/season)

Rate limiting: 0.65 seconds between requests with exponential backoff on failure.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashplayerstats,
    leaguedashteamstats,
    playergamelog,
    teamgamelog,
)
from nba_api.stats.library.parameters import SeasonTypeAllStar

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

RATE_LIMIT = cfg.scraping["nba_api"]["rate_limit_seconds"]
MAX_RETRIES = cfg.scraping["nba_api"]["max_retries"]
HEADERS = cfg.scraping["nba_api"]["headers"]

# nba_api uses 1996 as the first available season
NBA_API_FIRST_SEASON = 1996


def _season_str(year: int) -> str:
    """Convert season end year (e.g. 2025) to nba_api format '2024-25'."""
    return f"{year - 1}-{str(year)[-2:]}"


def _call_with_retry(endpoint_cls, **kwargs) -> pd.DataFrame:
    """Call an nba_api endpoint with rate limiting and retries."""
    for attempt in range(MAX_RETRIES):
        try:
            result = endpoint_cls(**kwargs, timeout=30)
            time.sleep(RATE_LIMIT)
            return result.get_data_frames()[0]
        except Exception as exc:
            wait = RATE_LIMIT * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "nba_api call failed (%s). Retrying in %.1fs (attempt %d/%d).",
                    exc, wait, attempt + 1, MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                logger.error("All retries exhausted for %s: %s", endpoint_cls.__name__, exc)
                return pd.DataFrame()
    return pd.DataFrame()


# =============================================================================
# Team advanced stats
# =============================================================================

def fetch_team_advanced_stats(season: int) -> pd.DataFrame:
    """Fetch team advanced stats (OFF_RATING, DEF_RATING, NET_RATING, PACE, PIE)."""
    season_str = _season_str(season)
    logger.info("Fetching NBA API team advanced stats: %s", season_str)
    df = _call_with_retry(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season_str,
        measure_type_detailed_defense="Advanced",
        per_mode_simple="PerGame",
        season_type_all_star=SeasonTypeAllStar.regular,
    )
    if not df.empty:
        df["season"] = season
        df["Team_abbrev"] = df["TEAM_ABBREVIATION"].map(
            lambda a: _safe_normalize(a)
        )
    return df


def _safe_normalize(name: str) -> str:
    try:
        return cfg.normalize_team(name)
    except KeyError:
        return name


def fetch_all_team_advanced(start: int, end: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for season in range(max(start, NBA_API_FIRST_SEASON), end + 1):
        df = fetch_team_advanced_stats(season)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_dir / "nba_api_team_advanced_all.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(combined), out_path)


# =============================================================================
# Player advanced stats
# =============================================================================

def fetch_player_advanced_stats(season: int) -> pd.DataFrame:
    """Fetch player advanced stats (PIE, USG_PCT, TS_PCT, OFF/DEF_RATING)."""
    season_str = _season_str(season)
    logger.info("Fetching NBA API player advanced stats: %s", season_str)
    df = _call_with_retry(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season_str,
        measure_type_detailed_defense="Advanced",
        per_mode_simple="PerGame",
        season_type_all_star=SeasonTypeAllStar.regular,
    )
    if not df.empty:
        df["season"] = season
    return df


def fetch_all_player_advanced(start: int, end: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for season in range(max(start, NBA_API_FIRST_SEASON), end + 1):
        df = fetch_player_advanced_stats(season)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_dir / "nba_api_player_advanced_all.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(combined), out_path)


# =============================================================================
# Team game logs (for momentum features — last N games)
# =============================================================================

def fetch_team_game_log(team_id: int, season: int,
                        season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch all games for a team in a season."""
    season_str = _season_str(season)
    df = _call_with_retry(
        teamgamelog.TeamGameLog,
        team_id=team_id,
        season=season_str,
        season_type_all_star=season_type,
    )
    if not df.empty:
        df["season"] = season
        df["TEAM_ID"] = team_id
    return df


def fetch_all_team_game_logs(
    team_ids: list[int], start: int, end: int, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for season in range(max(start, NBA_API_FIRST_SEASON), end + 1):
        for team_id in team_ids:
            df = fetch_team_game_log(team_id, season)
            if not df.empty:
                all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_dir / "team_game_logs_all.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("Saved %d game log rows to %s", len(combined), out_path)


# =============================================================================
# Player game logs (for player momentum features)
# =============================================================================

def fetch_player_game_log(player_id: int, season: int) -> pd.DataFrame:
    """Fetch all games for a player in a regular season."""
    season_str = _season_str(season)
    df = _call_with_retry(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=season_str,
        season_type_all_star="RegularSeason",
    )
    if not df.empty:
        df["season"] = season
        df["PLAYER_ID"] = player_id
    return df


# =============================================================================
# Team rosters
# =============================================================================

def fetch_team_roster(team_id: int, season: int) -> pd.DataFrame:
    """Fetch the roster for a team in a given season."""
    season_str = _season_str(season)
    df = _call_with_retry(
        commonteamroster.CommonTeamRoster,
        team_id=team_id,
        season=season_str,
    )
    if not df.empty:
        df["season"] = season
        df["TEAM_ID"] = team_id
    return df


def fetch_all_rosters(team_ids: list[int], start: int, end: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for season in range(max(start, NBA_API_FIRST_SEASON), end + 1):
        for team_id in team_ids:
            df = fetch_team_roster(team_id, season)
            if not df.empty:
                all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_dir / "rosters_all.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("Saved %d roster rows to %s", len(combined), out_path)


# =============================================================================
# Active NBA team IDs helper
# =============================================================================

def get_nba_team_ids() -> list[int]:
    """Return list of all active NBA team IDs."""
    from nba_api.stats.static import teams as nba_teams
    return [t["id"] for t in nba_teams.get_teams()]


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NBA Stats API data")
    parser.add_argument("--start-season", type=int, default=NBA_API_FIRST_SEASON)
    parser.add_argument("--end-season", type=int, default=cfg.seasons["end"])
    parser.add_argument(
        "--only",
        choices=["teams", "players", "gamelogs", "rosters"],
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    start, end = args.start_season, args.end_season
    logger.info("NBA API fetch: seasons %d–%d", start, end)

    team_out = cfg.project_root / cfg.paths["raw"]["nba_api"]["team_game_logs"]
    player_out = cfg.project_root / cfg.paths["raw"]["nba_api"]["player_game_logs"]

    team_ids = get_nba_team_ids()

    if args.only in (None, "teams"):
        fetch_all_team_advanced(start, end, team_out.parent)
    if args.only in (None, "players"):
        fetch_all_player_advanced(start, end, player_out.parent)
    if args.only in (None, "gamelogs"):
        fetch_all_team_game_logs(team_ids, start, end, team_out)
    if args.only in (None, "rosters"):
        fetch_all_rosters(team_ids, start, end, team_out.parent)

    logger.info("NBA API fetch complete.")


if __name__ == "__main__":
    main()
