"""Basketball Reference scraper — primary data source (1984–2025).

Fetches:
  - Team advanced stats per season
  - Player per-game and advanced stats per season
  - Playoff series results (bracket outcomes)
  - Team game logs (last N games for momentum features)
  - Player game logs (last N games for momentum features)

Rate limiting: 5 seconds between requests (bball-ref blocks faster).
Raw HTML is cached to SQLite via requests_cache so scraping only happens once.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import requests_cache
from bs4 import BeautifulSoup

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

BASE_URL = cfg.scraping["bball_ref"]["base_url"]
RATE_LIMIT = cfg.scraping["bball_ref"]["rate_limit_seconds"]
MAX_RETRIES = cfg.scraping["bball_ref"]["max_retries"]
RETRY_BACKOFF = cfg.scraping["bball_ref"]["retry_backoff_seconds"]

# Activate persistent HTTP cache (SQLite backend)
requests_cache.install_cache(
    cache_name=str(cfg.project_root / cfg.scraping["bball_ref"]["cache_name"]),
    backend="sqlite",
    expire_after=None,  # never expire — raw data is immutable
)


def _get(url: str) -> BeautifulSoup:
    """Fetch a URL with rate limiting and retries. Returns parsed BeautifulSoup."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            if not getattr(response, "from_cache", False):
                logger.debug("Fetched (live): %s", url)
                time.sleep(RATE_LIMIT)
            else:
                logger.debug("Fetched (cache): %s", url)
            return BeautifulSoup(response.text, "lxml")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                wait = RETRY_BACKOFF * (attempt + 1)
                logger.warning("Rate limited. Waiting %ds before retry %d.", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES - 1:
                logger.warning("Request failed (%s). Retrying in %ds.", exc, RETRY_BACKOFF)
                time.sleep(RETRY_BACKOFF)
            else:
                raise
    raise RuntimeError(f"All retries exhausted for URL: {url}")


def _parse_html_table(soup: BeautifulSoup, table_id: str) -> Optional[pd.DataFrame]:
    """Extract a named HTML table from bball-ref and return as DataFrame."""
    table = soup.find("table", {"id": table_id})
    if table is None:
        logger.warning("Table '%s' not found on page.", table_id)
        return None
    # bball-ref embeds some tables in HTML comments — unwrap if needed
    try:
        df = pd.read_html(str(table), header=0)[0]
    except Exception as exc:
        logger.error("Failed to parse table '%s': %s", table_id, exc)
        return None
    return df


# =============================================================================
# Team advanced stats
# =============================================================================

def fetch_team_advanced(season: int) -> pd.DataFrame:
    """Fetch team advanced stats for a given season year (e.g. 2025 = 2024-25).

    Returns DataFrame with columns:
        season, Team (canonical abbrev), Rk, Age, W, L, Pace, ORtg, DRtg,
        NRtg, eFG%, TOV%, ORB%, FT/FGA, eFG%_opp, TOV%_opp, DRB%_opp, FT/FGA_opp
    """
    url = f"{BASE_URL}/leagues/NBA_{season}_advanced.html"
    soup = _get(url)
    df = _parse_html_table(soup, "advanced-team")
    if df is None:
        return pd.DataFrame()

    # Drop separator rows (where Rk == "Rk")
    df = df[df["Rk"].astype(str) != "Rk"].copy()
    df = df[df["Team"].notna()].copy()

    # Normalize team names to canonical abbreviations
    df["Team"] = df["Team"].str.replace(r"\*", "", regex=True).str.strip()
    df["Team_abbrev"] = df["Team"].map(
        lambda name: _safe_normalize(name)
    )

    df["season"] = season
    logger.info("Fetched team advanced stats for season %d: %d teams", season, len(df))
    return df


def _safe_normalize(name: str) -> str:
    try:
        return cfg.normalize_team(name)
    except KeyError:
        logger.warning("Unknown team name: '%s' — leaving as-is", name)
        return name


def fetch_all_team_advanced(start: int, end: int, out_dir: Path) -> None:
    """Fetch and save team advanced stats for all seasons in range."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for season in range(start, end + 1):
        logger.info("Fetching team advanced stats: season %d", season)
        df = fetch_team_advanced(season)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_dir / "team_advanced_all.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("Saved %d team-season rows to %s", len(combined), out_path)


# =============================================================================
# Player per-game and advanced stats
# =============================================================================

def fetch_player_pergame(season: int) -> pd.DataFrame:
    """Fetch player per-game stats for a given season."""
    url = f"{BASE_URL}/leagues/NBA_{season}_per_game.html"
    soup = _get(url)
    df = _parse_html_table(soup, "per_game_stats")
    if df is None:
        return pd.DataFrame()

    df = df[df["Rk"].astype(str) != "Rk"].copy()
    df = df[df["Player"].notna()].copy()
    df["season"] = season
    df["Team_abbrev"] = df["Tm"].map(lambda t: _safe_normalize(str(t)))
    return df


def fetch_player_advanced(season: int) -> pd.DataFrame:
    """Fetch player advanced stats (BPM, VORP, WS, PER, etc.) for a given season."""
    url = f"{BASE_URL}/leagues/NBA_{season}_advanced.html"
    soup = _get(url)
    df = _parse_html_table(soup, "advanced_stats")
    if df is None:
        return pd.DataFrame()

    df = df[df["Rk"].astype(str) != "Rk"].copy()
    df = df[df["Player"].notna()].copy()
    df["season"] = season
    df["Team_abbrev"] = df["Tm"].map(lambda t: _safe_normalize(str(t)))
    return df


def fetch_all_player_stats(start: int, end: int, out_dir: Path) -> None:
    """Fetch and save player per-game and advanced stats for all seasons."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pg_dfs, adv_dfs = [], []
    for season in range(start, end + 1):
        logger.info("Fetching player stats: season %d", season)
        pg_dfs.append(fetch_player_pergame(season))
        adv_dfs.append(fetch_player_advanced(season))

    def _save(dfs: list[pd.DataFrame], name: str) -> None:
        combined = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
        path = out_dir / f"{name}.parquet"
        combined.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(combined), path)

    _save(pg_dfs, "player_pergame_all")
    _save(adv_dfs, "player_advanced_all")


# =============================================================================
# Playoff series results
# =============================================================================

def fetch_playoff_bracket(season: int) -> pd.DataFrame:
    """Fetch playoff series results for a given season.

    Returns DataFrame with one row per series:
        season, round, higher_seed, lower_seed,
        higher_seed_wins, lower_seed_wins, series_winner, series_length
    """
    url = f"{BASE_URL}/playoffs/NBA_{season}.html"
    soup = _get(url)

    series_rows = []
    # bball-ref playoff bracket page uses divs/tables — parse series results
    # The bracket is in <div id="all_playoffs"> which may be comment-wrapped
    bracket_div = soup.find("div", {"id": "all_playoffs"})
    if bracket_div is None:
        # Try unwrapping HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, str) and "playoffs" in t):
            inner_soup = BeautifulSoup(str(comment), "lxml")
            bracket_div = inner_soup.find("div", {"id": "all_playoffs"})
            if bracket_div:
                break

    if bracket_div is None:
        logger.warning("Could not find playoff bracket for season %d", season)
        return pd.DataFrame()

    # Each series is represented as a table within the bracket
    series_tables = bracket_div.find_all("table")
    for tbl in series_tables:
        rows = tbl.find_all("tr")
        if len(rows) < 2:
            continue
        # Parse team names and win counts from table rows
        teams, wins = [], []
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            team_cell = cells[0].get_text(strip=True)
            win_cell = cells[-1].get_text(strip=True) if len(cells) > 1 else ""
            if team_cell and team_cell.isalpha() is False:
                teams.append(team_cell)
                try:
                    wins.append(int(win_cell))
                except ValueError:
                    wins.append(0)

        if len(teams) >= 2 and len(wins) >= 2:
            # Higher wins = winner; use seed ordering from table position
            w0, w1 = wins[0], wins[1]
            total = w0 + w1
            if total not in (4, 5, 6, 7, 8, 9, 10, 11):  # sanity check
                continue
            winner = teams[0] if w0 > w1 else teams[1]
            series_rows.append(
                {
                    "season": season,
                    "team_a": _safe_normalize(teams[0]),
                    "team_b": _safe_normalize(teams[1]),
                    "team_a_wins": w0,
                    "team_b_wins": w1,
                    "series_winner": _safe_normalize(winner),
                    "series_length": total,
                }
            )

    df = pd.DataFrame(series_rows)
    logger.info("Fetched %d series for playoff season %d", len(df), season)
    return df


def fetch_all_playoff_series(start: int, end: int, out_dir: Path) -> None:
    """Fetch and save playoff series results for all seasons."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for season in range(start, end + 1):
        logger.info("Fetching playoff bracket: season %d", season)
        df = fetch_playoff_bracket(season)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_dir / "playoff_series_all.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("Saved %d playoff series to %s", len(combined), out_path)


# =============================================================================
# Team game logs (for momentum features)
# =============================================================================

def fetch_team_game_log(team_abbrev: str, season: int) -> pd.DataFrame:
    """Fetch all game log rows for a team in a given season."""
    url = f"{BASE_URL}/teams/{team_abbrev}/{season}_games.html"
    soup = _get(url)
    df = _parse_html_table(soup, "games")
    if df is None:
        return pd.DataFrame()
    df["season"] = season
    df["Team_abbrev"] = team_abbrev
    return df


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Basketball Reference data")
    parser.add_argument("--start-season", type=int, default=cfg.seasons["start"])
    parser.add_argument("--end-season", type=int, default=cfg.seasons["end"])
    parser.add_argument("--only", choices=["teams", "players", "playoffs"], default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    start, end = args.start_season, args.end_season
    logger.info("Basketball Reference fetch: seasons %d–%d", start, end)

    team_out = cfg.project_root / cfg.paths["raw"]["bball_ref"]["team_stats"]
    player_out = cfg.project_root / cfg.paths["raw"]["bball_ref"]["player_stats"]
    playoff_out = cfg.project_root / cfg.paths["raw"]["bball_ref"]["playoff_series"]

    if args.only in (None, "teams"):
        fetch_all_team_advanced(start, end, team_out)
    if args.only in (None, "players"):
        fetch_all_player_stats(start, end, player_out)
    if args.only in (None, "playoffs"):
        fetch_all_playoff_series(start, end, playoff_out)

    logger.info("Basketball Reference fetch complete.")


if __name__ == "__main__":
    main()
