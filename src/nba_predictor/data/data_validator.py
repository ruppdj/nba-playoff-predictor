"""Pandera schema validation for processed parquet files.

Run via: python -m nba_predictor.data.data_validator
       or: make validate
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)


# =============================================================================
# Schema definitions
# =============================================================================

SERIES_DATASET_SCHEMA = DataFrameSchema(
    {
        "season": Column(int, Check.between(1984, 2030)),
        "team_a": Column(str, Check.str_length(2, 4)),
        "team_b": Column(str, Check.str_length(2, 4)),
        "series_winner": Column(str, Check.str_length(2, 4)),
        "series_length": Column(int, Check.isin([4, 5, 6, 7])),
        "higher_seed_wins": Column(int, Check.isin([0, 1])),
        "NRtg_norm_a": Column(float, nullable=True),
        "NRtg_norm_b": Column(float, nullable=True),
        "VORP_sum_a": Column(float, nullable=True),
        "VORP_sum_b": Column(float, nullable=True),
    },
    coerce=True,
    strict=False,  # allow extra columns not listed here
)

TEAM_FEATURES_SCHEMA = DataFrameSchema(
    {
        "season": Column(int, Check.between(1984, 2030)),
        "Team_abbrev": Column(str, Check.str_length(2, 4)),
        "NRtg_norm": Column(float, nullable=True),
        "ORtg_norm": Column(float, nullable=True),
        "DRtg_norm": Column(float, nullable=True),
        "Win_pct": Column(float, Check.between(0.0, 1.0), nullable=True),
    },
    coerce=True,
    strict=False,
)


# =============================================================================
# Validation runner
# =============================================================================

def validate_file(schema: pa.DataFrameSchema, path: Path, label: str) -> bool:
    if not path.exists():
        logger.warning("Skipping %s — file not found: %s", label, path)
        return True  # not an error if not yet generated

    logger.info("Validating %s: %s", label, path)
    df = pd.read_parquet(path)
    try:
        schema.validate(df, lazy=True)
        logger.info("  ✓ %s passed validation (%d rows)", label, len(df))
        return True
    except pa.errors.SchemaErrors as exc:
        logger.error("  ✗ %s FAILED validation:\n%s", label, exc.failure_cases)
        return False


def run_all_validations() -> None:
    all_passed = True

    checks = [
        (
            TEAM_FEATURES_SCHEMA,
            cfg.path("processed", "team_features"),
            "team_season_features",
        ),
        (
            SERIES_DATASET_SCHEMA,
            cfg.path("processed", "series_dataset"),
            "series_dataset",
        ),
    ]

    for schema, path, label in checks:
        ok = validate_file(schema, path, label)
        if not ok:
            all_passed = False

    if all_passed:
        logger.info("All validations passed.")
    else:
        raise SystemExit("One or more validation checks failed — see logs above.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_all_validations()


if __name__ == "__main__":
    main()
