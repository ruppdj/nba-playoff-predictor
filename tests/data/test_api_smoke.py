"""API smoke tests — validate data source connectivity and response schema.

These tests make REAL network calls to verify API endpoints work correctly
before running the full fetch pipeline. They are skipped in CI by default
(marked with @pytest.mark.smoke) and should be run manually:

    pytest tests/data/test_api_smoke.py -m smoke -v

Requirements:
  - Active internet connection
  - nba_api package installed
  - No API keys needed for bball-ref or nba_api

Run before implementing the full fetch to confirm all endpoints respond.
"""

from __future__ import annotations

import pytest
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


pytestmark = pytest.mark.smoke  # skip unless explicitly requested


# ── Basketball Reference ──────────────────────────────────────────────────────

@pytest.mark.smoke
def test_bball_ref_team_advanced_accessible():
    """Verify Basketball Reference team advanced stats page is accessible."""
    url = "https://www.basketball-reference.com/leagues/NBA_2025_advanced.html"
    response = requests.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (educational research project)"
    })
    assert response.status_code == 200, (
        f"Basketball Reference returned {response.status_code}"
    )
    assert "advanced" in response.text.lower(), (
        "Expected 'advanced' in response — page may have changed"
    )


@pytest.mark.smoke
def test_bball_ref_playoff_bracket_accessible():
    """Verify Basketball Reference playoff bracket page is accessible."""
    url = "https://www.basketball-reference.com/playoffs/NBA_2025.html"
    response = requests.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (educational research project)"
    })
    assert response.status_code == 200, (
        f"BBRef playoff page returned {response.status_code}"
    )


@pytest.mark.smoke
def test_bball_ref_player_advanced_accessible():
    """Verify Basketball Reference player advanced stats page is accessible."""
    url = "https://www.basketball-reference.com/leagues/NBA_2025_advanced.html"
    response = requests.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (educational research project)"
    })
    assert response.status_code == 200


# ── NBA Stats API ─────────────────────────────────────────────────────────────

@pytest.mark.smoke
def test_nba_api_team_stats_accessible():
    """Verify nba_api LeagueDashTeamStats endpoint works."""
    pytest.importorskip("nba_api", reason="nba_api not installed")
    from nba_api.stats.endpoints import leaguedashteamstats
    import time

    df = None
    try:
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season="2024-25",
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
            timeout=30,
        )
        time.sleep(0.65)
        dfs = endpoint.get_data_frames()
        df = dfs[0] if dfs else None
    except Exception as exc:
        pytest.fail(f"nba_api LeagueDashTeamStats failed: {exc}")

    assert df is not None and not df.empty, "Expected non-empty team stats DataFrame"
    assert "TEAM_ABBREVIATION" in df.columns, "Expected TEAM_ABBREVIATION column"
    assert "OFF_RATING" in df.columns or "NET_RATING" in df.columns, (
        "Expected offensive/net rating columns"
    )
    assert len(df) >= 28, f"Expected at least 28 teams, got {len(df)}"


@pytest.mark.smoke
def test_nba_api_player_stats_accessible():
    """Verify nba_api LeagueDashPlayerStats endpoint works."""
    pytest.importorskip("nba_api", reason="nba_api not installed")
    from nba_api.stats.endpoints import leaguedashplayerstats
    import time

    try:
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2024-25",
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
            timeout=30,
        )
        time.sleep(0.65)
        df = endpoint.get_data_frames()[0]
    except Exception as exc:
        pytest.fail(f"nba_api LeagueDashPlayerStats failed: {exc}")

    assert not df.empty, "Expected non-empty player stats DataFrame"
    assert len(df) > 200, f"Expected 200+ players, got {len(df)}"


@pytest.mark.smoke
def test_nba_api_team_game_log_accessible():
    """Verify nba_api TeamGameLog endpoint works for a specific team."""
    pytest.importorskip("nba_api", reason="nba_api not installed")
    from nba_api.stats.endpoints import teamgamelog
    from nba_api.stats.static import teams as nba_teams
    import time

    # Use Boston Celtics (a reliable long-standing franchise)
    bos = next(t for t in nba_teams.get_teams() if t["abbreviation"] == "BOS")
    try:
        endpoint = teamgamelog.TeamGameLog(
            team_id=bos["id"],
            season="2024-25",
            timeout=30,
        )
        time.sleep(0.65)
        df = endpoint.get_data_frames()[0]
    except Exception as exc:
        pytest.fail(f"nba_api TeamGameLog failed: {exc}")

    assert not df.empty, "Expected non-empty game log"
    assert "WL" in df.columns, "Expected WL (win/loss) column"
    assert len(df) > 50, f"Expected 50+ games, got {len(df)}"


@pytest.mark.smoke
def test_fte_elo_csv_accessible():
    """Verify FiveThirtyEight ELO CSV is accessible on GitHub."""
    url = (
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
        "nba-elo/nbaallelo.csv"
    )
    response = requests.get(url, timeout=30)
    assert response.status_code == 200, f"FTE ELO returned {response.status_code}"
    # Validate CSV has expected columns
    first_line = response.text.split("\n")[0]
    assert "team_id" in first_line or "team" in first_line.lower(), (
        f"Unexpected CSV header: {first_line}"
    )
