"""Feature pipeline — assembles the final series_dataset.parquet.

This is the central integration module. It:
  1. Loads all raw parquet files from data/raw/
  2. Calls team, player, injury, and matchup feature builders
  3. Writes the final series_dataset.parquet to data/processed/
  4. Writes per-step intermediate files (team_season_features.parquet, etc.)
  5. Computes and stores MD5 checksums for reproducibility

Run via: python -m nba_predictor.features.feature_pipeline
      or: make process
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from nba_predictor.config import cfg
from nba_predictor.features.era_normalizer import normalize_team_stats
from nba_predictor.features.injury_features import compute_injury_features
from nba_predictor.features.matchup_features import build_matchup_dataset
from nba_predictor.features.player_features import (
    aggregate_player_to_team,
    compute_player_momentum,
)
from nba_predictor.features.team_features import build_team_season_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = cfg.project_root


def _load_parquet_safe(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        logger.warning("File not found (returning empty): %s [%s]", path, label)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info("Loaded %s: %d rows", label, len(df))
    return df


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_checksums(paths: list[Path], checksum_path: Path) -> None:
    lines = []
    for p in paths:
        if p.exists():
            lines.append(f"{_md5(p)}  {p.name}")
    checksum_path.write_text("\n".join(lines) + "\n")
    logger.info("Checksums written to %s", checksum_path)


def run_pipeline() -> pd.DataFrame:
    """Run the full feature pipeline and return the series dataset."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = PROJECT_ROOT / "data" / "raw"

    # ── Load raw data ──────────────────────────────────────────────────────
    team_adv_bref = _load_parquet_safe(
        raw / "bball_ref" / "team_stats" / "team_advanced_all.parquet",
        "bball_ref team advanced",
    )
    player_adv_bref = _load_parquet_safe(
        raw / "bball_ref" / "player_stats" / "player_advanced_all.parquet",
        "bball_ref player advanced",
    )
    player_pg_bref = _load_parquet_safe(
        raw / "bball_ref" / "player_stats" / "player_pergame_all.parquet",
        "bball_ref player per-game",
    )
    playoff_series = _load_parquet_safe(
        raw / "bball_ref" / "playoff_series" / "playoff_series_all.parquet",
        "playoff series",
    )
    team_game_logs = _load_parquet_safe(
        raw / "nba_api" / "team_game_logs" / "team_game_logs_all.parquet",
        "nba_api team game logs",
    )
    player_game_logs = _load_parquet_safe(
        raw / "nba_api" / "player_game_logs" / "player_game_logs_all.parquet",
        "nba_api player game logs",
    )
    rosters = _load_parquet_safe(
        raw / "nba_api" / "team_game_logs" / "rosters_all.parquet",
        "nba_api rosters",
    )

    if team_adv_bref.empty:
        raise RuntimeError(
            "No team stats found. Run 'make fetch' first to download raw data."
        )

    if playoff_series.empty:
        raise RuntimeError(
            "No playoff series data found. Run 'make fetch' first."
        )

    # ── Step 1: Team season features ──────────────────────────────────────
    logger.info("Step 1: Building team season features...")
    team_features = build_team_season_features(
        team_adv_bref, team_game_logs, playoff_series
    )
    team_out = processed_dir / "team_season_features.parquet"
    team_features.to_parquet(team_out, index=False)
    logger.info("Saved team features: %s", team_out)

    # ── Step 2: Player season features ────────────────────────────────────
    logger.info("Step 2: Building player season features...")
    # Merge per-game and advanced stats
    if not player_adv_bref.empty and not player_pg_bref.empty:
        player_combined = player_adv_bref.merge(
            player_pg_bref[
                [c for c in ["season", "Player", "Tm", "G", "GS", "MP",
                             "PTS", "TRB", "AST", "STL", "BLK", "TOV"] if c in player_pg_bref.columns]
            ],
            on=["season", "Player", "Tm"],
            how="left",
        )
    else:
        player_combined = player_adv_bref if not player_adv_bref.empty else player_pg_bref

    # Ensure Team_abbrev is consistent
    if "Tm" in player_combined.columns and "Team_abbrev" not in player_combined.columns:
        player_combined["Team_abbrev"] = player_combined["Tm"].map(
            lambda t: _safe_normalize(str(t))
        )

    player_team_features = aggregate_player_to_team(player_combined)
    player_momentum = compute_player_momentum(
        player_game_logs, player_combined
    )

    # Merge momentum into player team features
    if not player_momentum.empty:
        player_team_features = player_team_features.merge(
            player_momentum, on=["season", "Team_abbrev"], how="left"
        )

    player_out = processed_dir / "player_season_features.parquet"
    player_team_features.to_parquet(player_out, index=False)
    logger.info("Saved player features: %s", player_out)

    # ── Step 3: Injury features ────────────────────────────────────────────
    logger.info("Step 3: Building injury features...")
    # Use roster data if available, otherwise build from player advanced
    if not rosters.empty:
        roster_df = rosters.copy()
        if "Team_abbrev" not in roster_df.columns and "TEAM_ABBREVIATION" in roster_df.columns:
            roster_df["Team_abbrev"] = roster_df["TEAM_ABBREVIATION"].map(_safe_normalize)
        if "PLAYER_NAME" not in roster_df.columns and "PLAYER" in roster_df.columns:
            roster_df["PLAYER_NAME"] = roster_df["PLAYER"]
    else:
        # Reconstruct minimal roster from player advanced
        roster_df = player_combined[
            [c for c in ["season", "Team_abbrev", "Player"] if c in player_combined.columns]
        ].rename(columns={"Player": "PLAYER_NAME"}).drop_duplicates()

    # No structured injury report — will fall back to GP-based estimates for pre-2010
    injury_features = compute_injury_features(
        roster_df=roster_df,
        injury_report=None,   # TODO: load from data/raw/nba_api/injury_reports/ when available
        player_advanced=player_combined,
    )

    injury_out = processed_dir / "injury_adjusted.parquet"
    injury_features.to_parquet(injury_out, index=False)
    logger.info("Saved injury features: %s", injury_out)

    # ── Step 4: Build matchup dataset ─────────────────────────────────────
    logger.info("Step 4: Building matchup (series) dataset...")
    series_dataset = build_matchup_dataset(
        playoff_series=playoff_series,
        team_features=team_features,
        player_features=player_team_features,
        injury_features=injury_features,
        game_logs=team_game_logs if not team_game_logs.empty else None,
    )

    series_out = processed_dir / "series_dataset.parquet"
    series_dataset.to_parquet(series_out, index=False)
    logger.info(
        "Saved series dataset: %d rows, %d columns — %s",
        len(series_dataset), len(series_dataset.columns), series_out,
    )

    # ── Step 5: Write checksums ────────────────────────────────────────────
    checksum_path = processed_dir / "checksums.txt"
    _write_checksums(
        [team_out, player_out, injury_out, series_out],
        checksum_path,
    )

    logger.info("Feature pipeline complete.")
    return series_dataset


def _safe_normalize(name: str) -> str:
    try:
        return cfg.normalize_team(name)
    except KeyError:
        return name


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_pipeline()


if __name__ == "__main__":
    main()
