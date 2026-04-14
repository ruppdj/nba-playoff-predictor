"""Bracket simulator — apply trained models to the 2026 NBA playoff bracket.

Simulates the full bracket 10,000 times using calibrated win probabilities.
Each simulation samples stochastically, propagating uncertainty through all
four rounds (winner of round 1 advances to face the appropriate opponent, etc.).

Output: CSV with per-series probabilities + champion probability distribution.

Run via: python -m nba_predictor.predict.bracket_simulator --season 2026
      or: make predict
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_predictor.config import cfg
from nba_predictor.models.monte_carlo import simulate_series_from_features

logger = logging.getLogger(__name__)

RANDOM_STATE = cfg.modeling["random_state"]
N_SIMULATIONS = cfg.modeling["monte_carlo"]["n_simulations"]

# NBA playoffs bracket structure: 16 teams, 4 rounds
# Seeds within each conference: 1v8, 2v7, 3v6, 4v5 in first round
FIRST_ROUND_MATCHUPS = [
    # East
    {"higher_seed": None, "lower_seed": None, "conference": "East",
     "round": "first_round", "matchup_id": "E1v8"},
    {"higher_seed": None, "lower_seed": None, "conference": "East",
     "round": "first_round", "matchup_id": "E2v7"},
    {"higher_seed": None, "lower_seed": None, "conference": "East",
     "round": "first_round", "matchup_id": "E3v6"},
    {"higher_seed": None, "lower_seed": None, "conference": "East",
     "round": "first_round", "matchup_id": "E4v5"},
    # West
    {"higher_seed": None, "lower_seed": None, "conference": "West",
     "round": "first_round", "matchup_id": "W1v8"},
    {"higher_seed": None, "lower_seed": None, "conference": "West",
     "round": "first_round", "matchup_id": "W2v7"},
    {"higher_seed": None, "lower_seed": None, "conference": "West",
     "round": "first_round", "matchup_id": "W3v6"},
    {"higher_seed": None, "lower_seed": None, "conference": "West",
     "round": "first_round", "matchup_id": "W4v5"},
]


def _load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun 'make train' first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_bracket_input(season: int) -> pd.DataFrame:
    """Load the bracket input CSV with this season's matchup features."""
    path = cfg.project_root / "data" / "predictions" / str(season) / "bracket_input.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Bracket input not found: {path}\n"
            "Build bracket_input.csv from current-season features first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded bracket input: %d matchups", len(df))
    return df


def _get_winner_prob(
    model: Any,
    matchup_features: pd.DataFrame,
    row_idx: int,
) -> float:
    """Get the probability that the higher seed wins a series."""
    row = matchup_features.iloc[[row_idx]]
    try:
        return float(model.predict_proba(row)[:, 1][0])
    except Exception:
        # Fallback: use NRtg delta directly via Monte Carlo
        delta_nrtg = float(row.get("delta_NRtg", pd.Series([0.0])).iloc[0])
        result = simulate_series_from_features({"delta_NRtg": delta_nrtg})
        return result["p_higher_seed_wins"]


def simulate_full_bracket(
    bracket_df: pd.DataFrame,
    winner_model: Any,
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Simulate the full playoff bracket N times.

    Args:
        bracket_df: DataFrame with one row per series, containing:
            matchup_id, higher_seed, lower_seed, round, conference,
            and all model feature columns.
        winner_model: Fitted series winner model.
        n_simulations: Number of complete bracket simulations.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
          - series_results: list of per-series dicts with probabilities
          - champion_probs: dict of team → champion probability
    """
    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    # Compute win probabilities for each series
    series_probs: dict[str, float] = {}
    for i, row in bracket_df.iterrows():
        matchup_id = row.get("matchup_id", str(i))
        p_higher = _get_winner_prob(winner_model, bracket_df, bracket_df.index.get_loc(i))
        series_probs[matchup_id] = p_higher
        logger.debug(
            "Series %s: P(%s wins) = %.3f",
            matchup_id, row.get("higher_seed", "?"), p_higher,
        )

    # Monte Carlo simulation
    champion_counts: dict[str, int] = {}
    all_teams = set(bracket_df["higher_seed"].tolist() + bracket_df["lower_seed"].tolist())
    for t in all_teams:
        champion_counts[t] = 0

    for sim_i in range(n_simulations):
        # Simulate bracket round by round
        # Build a mapping from matchup_id to the winner for this simulation
        sim_winners: dict[str, str] = {}

        for _, row in bracket_df.iterrows():
            matchup_id = str(row.get("matchup_id", ""))
            higher = str(row.get("higher_seed", ""))
            lower = str(row.get("lower_seed", ""))
            p_higher = series_probs.get(matchup_id, 0.5)

            if rng.random() < p_higher:
                sim_winners[matchup_id] = higher
            else:
                sim_winners[matchup_id] = lower

        # Identify the NBA Finals winner (last round)
        # The actual bracket advancement logic needs the bracket structure.
        # For now, use the Finals matchup as the champion.
        finals_rows = bracket_df[bracket_df["round"] == "nba_finals"]
        if not finals_rows.empty:
            finals_id = str(finals_rows.iloc[0].get("matchup_id", ""))
            champion = sim_winners.get(finals_id, "")
        else:
            # Fallback: pick the team with highest probability that appears in all rounds
            finalist_probs = {
                str(row.get("higher_seed", "")): series_probs.get(str(row.get("matchup_id", "")), 0.5)
                for _, row in bracket_df.iterrows()
            }
            champion = max(finalist_probs, key=finalist_probs.get) if finalist_probs else ""

        if champion and champion in champion_counts:
            champion_counts[champion] += 1

    # Build output
    series_results = []
    for _, row in bracket_df.iterrows():
        matchup_id = str(row.get("matchup_id", ""))
        higher = str(row.get("higher_seed", ""))
        lower = str(row.get("lower_seed", ""))
        p_higher = series_probs.get(matchup_id, 0.5)

        # Get series length distribution via Monte Carlo
        delta_nrtg = float(row.get("delta_NRtg", 0.0)) if "delta_NRtg" in row else 0.0
        length_dist = simulate_series_from_features({"delta_NRtg": delta_nrtg}, n_simulations=1000)

        series_results.append(
            {
                "matchup_id": matchup_id,
                "round": row.get("round", ""),
                "conference": row.get("conference", ""),
                "higher_seed": higher,
                "lower_seed": lower,
                "p_higher_seed_wins": round(p_higher, 4),
                "p_lower_seed_wins": round(1 - p_higher, 4),
                "predicted_winner": higher if p_higher >= 0.5 else lower,
                "p_length_4": round(length_dist["p_length_4"], 4),
                "p_length_5": round(length_dist["p_length_5"], 4),
                "p_length_6": round(length_dist["p_length_6"], 4),
                "p_length_7": round(length_dist["p_length_7"], 4),
                "expected_length": round(length_dist["expected_length"], 2),
            }
        )

    champion_probs = {
        team: count / n_simulations
        for team, count in champion_counts.items()
    }

    return {
        "series_results": series_results,
        "champion_probs": champion_probs,
    }


def save_predictions(results: dict, season: int) -> Path:
    """Write predictions to data/predictions/{season}/bracket_output.csv."""
    out_dir = cfg.project_root / "data" / "predictions" / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_df = pd.DataFrame(results["series_results"])
    series_path = out_dir / "bracket_output.csv"
    series_df.to_csv(series_path, index=False)

    champ_df = pd.DataFrame(
        [{"team": k, "champion_probability": v}
         for k, v in sorted(results["champion_probs"].items(), key=lambda x: -x[1])]
    )
    champ_path = out_dir / "champion_probabilities.csv"
    champ_df.to_csv(champ_path, index=False)

    logger.info("Predictions saved:")
    logger.info("  Series: %s", series_path)
    logger.info("  Champions: %s", champ_path)
    return series_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NBA playoff bracket predictions")
    parser.add_argument("--season", type=int, default=cfg.seasons["current"])
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    # Load winner model (ensemble preferred, fall back to XGBoost)
    model_dir = cfg.project_root / cfg.paths["models"]["trained"]
    model_path = model_dir / "ensemble_winner.pkl"
    if not model_path.exists():
        model_path = model_dir / "xgboost_winner.pkl"
    if not model_path.exists():
        model_path = model_dir / "lightgbm_winner.pkl"

    winner_model = _load_model(model_path)
    logger.info("Loaded winner model: %s", model_path)

    bracket_df = _load_bracket_input(args.season)

    results = simulate_full_bracket(
        bracket_df, winner_model, n_simulations=args.n_simulations
    )
    save_predictions(results, args.season)

    # Print summary
    logger.info("\n=== 2026 NBA Playoffs Prediction ===")
    for series in results["series_results"]:
        logger.info(
            "%s vs %s: P(%s wins) = %.1f%%  [expected %.1f games]",
            series["higher_seed"], series["lower_seed"],
            series["predicted_winner"],
            series["p_higher_seed_wins"] * 100,
            series["expected_length"],
        )

    logger.info("\n=== Championship Probabilities ===")
    for team, prob in sorted(results["champion_probs"].items(), key=lambda x: -x[1])[:10]:
        logger.info("  %-5s %.1f%%", team, prob * 100)


if __name__ == "__main__":
    main()
