"""Kaggle / supplemental data loader.

Downloads:
  - wyattowalsh/basketball — SQLite DB with comprehensive historical box scores
  - FiveThirtyEight ELO ratings (archived CSV)

Requires ~/.kaggle/kaggle.json credentials for the Kaggle datasets.
FiveThirtyEight data is fetched directly from GitHub (no key required).

NOTE: Will prompt the user if kaggle.json is not found.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd
import requests

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "wyattowalsh/basketball"
FTE_ELO_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "nba-elo/nbaallelo.csv"
)


def _check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error(
            "Kaggle credentials not found at %s.\n"
            "To set up Kaggle access:\n"
            "  1. Go to https://www.kaggle.com/settings/account\n"
            "  2. Click 'Create New Token' to download kaggle.json\n"
            "  3. Move it to ~/.kaggle/kaggle.json\n"
            "  4. Run: chmod 600 ~/.kaggle/kaggle.json\n"
            "Then re-run: make fetch",
            kaggle_json,
        )
        return False
    return True


def download_kaggle_basketball_db(out_dir: Path) -> Path | None:
    """Download the wyattowalsh/basketball SQLite dataset from Kaggle."""
    if not _check_kaggle_credentials():
        return None

    try:
        import kaggle  # noqa: F401 — only available if kaggle package installed
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Kaggle dataset '%s'...", KAGGLE_DATASET)
    import subprocess
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "--unzip", "-p", str(out_dir)],
        check=True,
    )
    db_path = out_dir / "basketball.sqlite"
    if db_path.exists():
        logger.info("Kaggle dataset downloaded to %s", db_path)
        return db_path
    logger.warning("Expected SQLite file not found at %s", db_path)
    return None


def load_kaggle_game_data(db_path: Path) -> pd.DataFrame:
    """Load game-level data from the basketball.sqlite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT game_id, season_id, team_id_home, team_id_away,
               game_date, pts_home, pts_away, wl_home,
               fg_pct_home, fg3_pct_home, ft_pct_home,
               reb_home, ast_home, stl_home, blk_home, tov_home,
               fg_pct_away, fg3_pct_away, ft_pct_away,
               reb_away, ast_away, stl_away, blk_away, tov_away
        FROM game
        WHERE season_id >= 21984
        ORDER BY game_date
        """,
        conn,
    )
    conn.close()
    logger.info("Loaded %d games from Kaggle basketball.sqlite", len(df))
    return df


def download_fte_elo(out_dir: Path) -> Path:
    """Download FiveThirtyEight NBA ELO ratings CSV (no API key required)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fte_nba_elo.csv"
    if out_path.exists():
        logger.info("FTE ELO file already exists: %s", out_path)
        return out_path

    logger.info("Downloading FTE ELO ratings from GitHub...")
    response = requests.get(FTE_ELO_URL, timeout=60)
    response.raise_for_status()
    out_path.write_text(response.text, encoding="utf-8")
    logger.info("Saved FTE ELO ratings to %s", out_path)
    return out_path


def load_fte_elo(path: Path) -> pd.DataFrame:
    """Load and parse the FTE ELO ratings CSV."""
    df = pd.read_csv(path)
    # Convert date strings and filter to modern era
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= 1984].copy()
    logger.info("Loaded %d FTE ELO rows (1984+)", len(df))
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = cfg.project_root / cfg.paths["raw"]["kaggle"]["supplemental"]

    # FTE ELO — no credentials needed
    elo_path = download_fte_elo(out_dir)
    if elo_path:
        elo_df = load_fte_elo(elo_path)
        elo_df.to_parquet(out_dir / "fte_elo.parquet", index=False)
        logger.info("Saved FTE ELO to parquet.")

    # Kaggle basketball DB — requires credentials
    db_path = download_kaggle_basketball_db(out_dir)
    if db_path:
        games_df = load_kaggle_game_data(db_path)
        games_df.to_parquet(out_dir / "kaggle_games.parquet", index=False)
        logger.info("Saved Kaggle game data to parquet.")


if __name__ == "__main__":
    main()
