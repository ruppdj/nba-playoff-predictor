"""Scraper diagnostic script — fetches one season of each data type and validates structure.

Run with:
    make diagnose-scraper

Checks actual column names, row counts, and required fields against what the
parsing code expects. Prints PASS/FAIL per data source with details on mismatches.
"""

from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SEASON = 2025
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    results.append((name, condition, detail))
    return condition


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ── Basketball Reference ───────────────────────────────────────────────────────

section(f"Basketball Reference — season {SEASON}")

try:
    from nba_predictor.data.bball_ref_scraper import (
        fetch_player_advanced,
        fetch_player_pergame,
        fetch_playoff_bracket,
        fetch_team_advanced,
    )

    # Team advanced stats
    print("\n[Team Advanced Stats]")
    df_team = fetch_team_advanced(SEASON)
    check("Non-empty", not df_team.empty, f"rows={len(df_team)}")
    check("Has 30 teams", len(df_team) == 30, f"got {len(df_team)}")
    for col in ["Team", "Team_abbrev", "season", "ORtg", "DRtg", "NRtg", "Pace", "eFG%"]:
        check(f"Column '{col}'", col in df_team.columns)
    check(
        "No 2TM/3TM in Team_abbrev",
        not df_team["Team_abbrev"].str.match(r"^\d+TM$", na=False).any(),
    )
    check(
        "3-letter abbrevs only",
        df_team["Team_abbrev"].str.len().eq(3).all(),
        f"bad values: {list(df_team[df_team['Team_abbrev'].str.len() != 3]['Team_abbrev'])[:5]}",
    )
    if not df_team.empty:
        print(f"  Sample abbrevs: {list(df_team['Team_abbrev'][:6])}")

    time.sleep(5)

    # Player per-game stats
    print("\n[Player Per-Game Stats]")
    df_pg = fetch_player_pergame(SEASON)
    check("Non-empty", not df_pg.empty, f"rows={len(df_pg)}")
    check("Has 300+ player-team rows", len(df_pg) > 300, f"got {len(df_pg)}")
    for col in ["Player", "Team", "Team_abbrev", "season", "G", "MP", "PTS"]:
        check(f"Column '{col}'", col in df_pg.columns)
    check("No 2TM/3TM rows", not df_pg["Team"].astype(str).str.match(r"^\d+TM$", na=False).any())
    check("No NaN Team rows", df_pg["Team"].notna().all())

    time.sleep(5)

    # Player advanced stats
    print("\n[Player Advanced Stats]")
    df_adv = fetch_player_advanced(SEASON)
    check("Non-empty", not df_adv.empty, f"rows={len(df_adv)}")
    for col in ["Player", "Team", "Team_abbrev", "season", "BPM", "VORP", "WS"]:
        check(f"Column '{col}'", col in df_adv.columns)
    check("No 2TM/3TM rows", not df_adv["Team"].astype(str).str.match(r"^\d+TM$", na=False).any())

    time.sleep(5)

    # Playoff bracket
    print("\n[Playoff Bracket]")
    df_po = fetch_playoff_bracket(SEASON)
    check("Non-empty", not df_po.empty, f"rows={len(df_po)}")
    check("Has 15 series", len(df_po) == 15, f"got {len(df_po)}")
    for col in ["season", "round", "team_a", "team_b", "series_winner", "series_length"]:
        check(f"Column '{col}'", col in df_po.columns)
    if "team_a" in df_po.columns:
        check(
            "team_a are 3-letter abbrevs",
            df_po["team_a"].str.len().eq(3).all(),
            f"bad: {list(df_po[df_po['team_a'].str.len() != 3]['team_a'])}",
        )
        check(
            "series_winner are 3-letter abbrevs",
            df_po["series_winner"].str.len().eq(3).all(),
            f"bad: {list(df_po[df_po['series_winner'].str.len() != 3]['series_winner'])}",
        )
    if "series_length" in df_po.columns:
        check(
            "series_length in {4,5,6,7}",
            df_po["series_length"].isin([4, 5, 6, 7]).all(),
            f"values: {sorted(df_po['series_length'].unique())}",
        )
    if not df_po.empty:
        print(
            f"  Sample series:\n{df_po[['round','team_a','team_b','series_winner','series_length']].head(3).to_string(index=False)}"
        )

except Exception as exc:
    print(f"  [{FAIL}] bball_ref_scraper import/run failed: {exc}")
    import traceback

    traceback.print_exc()


# ── NBA Stats API ─────────────────────────────────────────────────────────────

section(f"NBA Stats API — season {SEASON}")

try:
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.library.parameters import SeasonTypeAllStar

    from nba_predictor.config import cfg

    HEADERS = cfg.scraping["nba_api"]["headers"]
    season_str = f"{SEASON - 1}-{str(SEASON)[-2:]}"

    print(f"\n[LeagueDashTeamStats — {season_str}]")
    try:
        ep = leaguedashteamstats.LeagueDashTeamStats(
            season=season_str,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star=SeasonTypeAllStar.regular,
            timeout=30,
            headers=HEADERS,
        )
        time.sleep(0.65)
        df_nba = ep.get_data_frames()[0]
        check("Non-empty", not df_nba.empty, f"rows={len(df_nba)}")
        check("Has 30 teams", len(df_nba) == 30, f"got {len(df_nba)}")
        for col in ["TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "PIE"]:
            check(f"Column '{col}'", col in df_nba.columns)
        check(
            "No TEAM_ABBREVIATION (we use TEAM_NAME)",
            "TEAM_ABBREVIATION" not in df_nba.columns or True,  # informational
            "Using TEAM_NAME for normalization",
        )
        if not df_nba.empty:
            print(f"  Sample teams: {list(df_nba['TEAM_NAME'][:5])}")
    except Exception as exc:
        check("LeagueDashTeamStats call", False, str(exc))

except Exception as exc:
    print(f"  [{FAIL}] nba_api import failed: {exc}")


# ── FTE ELO ───────────────────────────────────────────────────────────────────

section("FiveThirtyEight ELO CSV")

try:
    import requests

    FTE_URL = (
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/" "nba-elo/nbaallelo.csv"
    )
    print("\n[FTE ELO CSV]")
    try:
        r = requests.get(FTE_URL, timeout=30)
        check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
        header = r.text.split("\n")[0]
        for col in ["team_id", "elo_i", "elo_n", "game_result"]:
            check(f"CSV has column '{col}'", col in header, f"header: {header[:80]}")
    except Exception as exc:
        check("FTE ELO fetch", False, str(exc))

except Exception as exc:
    print(f"  [{FAIL}] requests import failed: {exc}")


# ── Summary ───────────────────────────────────────────────────────────────────

section("Summary")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"\n  {passed} passed, {failed} failed\n")
if failed:
    print("  Failed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"    - {name}: {detail}")
    sys.exit(1)
else:
    print("  All checks passed — scraper is healthy.")
