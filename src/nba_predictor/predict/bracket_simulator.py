"""Bracket simulator — apply trained models to the 2026 NBA playoff bracket.

Simulates all four rounds of the NBA playoffs 10,000 times using calibrated
win probabilities from the ensemble model, properly advancing winners through:
  Round 1  → Round 2  → Conference Finals  → NBA Finals

Win probabilities for ALL possible team pairings are pre-computed once before
the Monte Carlo loop (16 teams → 120 pairs), so simulation is fast (~1 sec).

Run via: python -m nba_predictor.predict.bracket_simulator --season 2026
      or: make predict

Options:
  --upset-threshold FLOAT   Pick the lower seed when P(higher seed wins) < this
                            value. Default 0.5 (standard). The 2026 final
                            prediction uses 0.532, which adds TOR over CLE and
                            MIN over DEN on top of the HOU over LAL call.
                            Upset picks are marked with * in printed output.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_predictor.config import cfg
from nba_predictor.models.ensemble import StackingEnsemble  # noqa: F401 — needed for pickle
from nba_predictor.models.monte_carlo import simulate_series_from_features
from nba_predictor.predict.build_bracket_input import _build_matchup_row

logger = logging.getLogger(__name__)

RANDOM_STATE = cfg.modeling["random_state"]
N_SIMULATIONS = cfg.modeling["monte_carlo"]["n_simulations"]

ROUND_NAMES = ["first_round", "second_round", "conf_finals", "nba_finals"]


# ── Model loading ──────────────────────────────────────────────────────────────


def _load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun 'make train' first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_bracket_files(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = cfg.project_root / "data" / "predictions" / str(season)
    input_path = base / "bracket_input.csv"
    teams_path = base / "bracket_teams.csv"
    for p in (input_path, teams_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found: {p}\n"
                "Run: python -m nba_predictor.predict.build_bracket_input"
            )
    return pd.read_csv(input_path), pd.read_csv(teams_path)


# ── Feature / probability helpers ─────────────────────────────────────────────


def _get_model_feature_cols(df: pd.DataFrame) -> list[str]:
    all_feat = (
        cfg.features.get("matchup", [])
        + cfg.features.get("meta", [])
        + [f"higher_{c}" for c in cfg.features.get("injury", [])]
        + [f"lower_{c}" for c in cfg.features.get("injury", [])]
    )
    return [c for c in all_feat if c in df.columns]


def _model_prob(model: Any, feature_row: pd.DataFrame) -> float:
    """Return P(higher_seed wins) for a single matchup row."""
    feat_cols = _get_model_feature_cols(feature_row)
    X = feature_row[feat_cols].fillna(0)
    try:
        return float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        logger.debug("predict_proba fallback (%s)", e)
        delta_nrtg = float(X.get("delta_NRtg", pd.Series([0.0])).iloc[0])
        return simulate_series_from_features({"delta_NRtg": delta_nrtg})["p_higher_seed_wins"]


def _team_attr(team_store: pd.DataFrame, team: str, col: str):
    rows = team_store[team_store["team"] == team]
    return rows.iloc[0][col] if not rows.empty else None


LENGTH_CLASSES = [4, 5, 6, 7]


def _predict_length(length_model: Any | None, feat_row: pd.DataFrame) -> dict:
    """Predict series length distribution using the trained lgbm_length model.

    Falls back to NRtg-based Monte Carlo if the model is unavailable.
    Returns dict with p_length_4/5/6/7, expected_length, modal_length.
    """
    if length_model is not None:
        try:
            feat_names = list(length_model.feature_name_)
            X = feat_row.reindex(columns=feat_names).fillna(0)
            probs = length_model.predict_proba(X)[0]  # shape (4,)
            classes = length_model.classes_  # [4, 5, 6, 7]
            prob_map = dict(zip(classes, probs, strict=False))
            expected = sum(g * prob_map.get(g, 0) for g in LENGTH_CLASSES)
            modal = max(LENGTH_CLASSES, key=lambda g: prob_map.get(g, 0))
            return {
                "p_length_4": round(float(prob_map.get(4, 0)), 4),
                "p_length_5": round(float(prob_map.get(5, 0)), 4),
                "p_length_6": round(float(prob_map.get(6, 0)), 4),
                "p_length_7": round(float(prob_map.get(7, 0)), 4),
                "expected_length": round(expected, 2),
                "modal_length": int(modal),
            }
        except Exception as e:
            logger.debug("Length model fallback (%s)", e)

    # Fallback: NRtg-based Monte Carlo
    delta_nrtg = float(feat_row.get("delta_NRtg", pd.Series([0.0])).iloc[0])
    dist = simulate_series_from_features({"delta_NRtg": delta_nrtg}, n_simulations=2000)
    probs = [dist[f"p_length_{g}"] for g in LENGTH_CLASSES]
    modal = LENGTH_CLASSES[int(np.argmax(probs))]
    return {
        "p_length_4": round(dist["p_length_4"], 4),
        "p_length_5": round(dist["p_length_5"], 4),
        "p_length_6": round(dist["p_length_6"], 4),
        "p_length_7": round(dist["p_length_7"], 4),
        "expected_length": round(dist["expected_length"], 2),
        "modal_length": modal,
    }


def _precompute_all_probs(
    model: Any,
    team_store: pd.DataFrame,
    season: int,
) -> dict[tuple[str, str], float]:
    """Pre-compute P(higher_seed wins) for every possible team pair.

    Key: (higher_seed, lower_seed) — lower seed number = home court.
    Returns a dict of (higher, lower) → float probability.
    """
    teams = team_store["team"].tolist()
    prob_table: dict[tuple[str, str], float] = {}

    # Use generic round name; model features don't differ much by round label
    round_name = "second_round"

    for team_a, team_b in itertools.combinations(teams, 2):
        seed_a = int(_team_attr(team_store, team_a, "seed") or 99)
        seed_b = int(_team_attr(team_store, team_b, "seed") or 99)
        conf_a = str(_team_attr(team_store, team_a, "conference") or "")
        conf_b = str(_team_attr(team_store, team_b, "conference") or "")
        conf = conf_a if conf_a else conf_b

        if seed_a <= seed_b:
            higher, lower, seed_diff = team_a, team_b, float(seed_b - seed_a)
        else:
            higher, lower, seed_diff = team_b, team_a, float(seed_a - seed_b)

        feat_row = pd.DataFrame(
            [
                _build_matchup_row(
                    higher,
                    lower,
                    seed_diff,
                    conf,
                    f"{higher}v{lower}",
                    round_name,
                    season,
                    team_store,
                )
            ]
        )
        p = _model_prob(model, feat_row)
        prob_table[(higher, lower)] = p

    logger.info("Pre-computed win probabilities for %d team pairs", len(prob_table))
    return prob_table


def _lookup_prob(
    prob_table: dict[tuple[str, str], float],
    team_a: str,
    team_b: str,
    team_store: pd.DataFrame,
) -> tuple[str, str, float]:
    """Return (higher_seed, lower_seed, p_higher_wins) for any team pair."""
    seed_a = int(_team_attr(team_store, team_a, "seed") or 99)
    seed_b = int(_team_attr(team_store, team_b, "seed") or 99)

    if seed_a <= seed_b:
        higher, lower = team_a, team_b
    else:
        higher, lower = team_b, team_a

    key = (higher, lower)
    # Try both orderings in case something was stored reversed
    p = prob_table.get(key, prob_table.get((lower, higher), 0.5))
    # If we got the reversed key, flip the probability
    if key not in prob_table and (lower, higher) in prob_table:
        p = 1.0 - p

    return higher, lower, p


# ── Main simulation ────────────────────────────────────────────────────────────


def simulate_full_bracket(
    bracket_input: pd.DataFrame,
    team_store: pd.DataFrame,
    winner_model: Any,
    season: int,
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_STATE,
    length_model: Any | None = None,
) -> dict[str, Any]:
    """Simulate all four playoff rounds N times.

    Bracket structure (traditional, no re-seeding between rounds):
      R1:  1v8, 2v7, 3v6, 4v5  (each conference)
      R2:  winner(1v8) vs winner(4v5),  winner(2v7) vs winner(3v6)
      R3:  R2 upper winner vs R2 lower winner  (Conference Finals)
      R4:  East champion vs West champion  (NBA Finals)
    """
    rng = random.Random(random_seed)

    # ── Pre-compute all pairwise win probabilities (120 pairs) ───────────────
    logger.info("Pre-computing win probabilities for all team pairs...")
    prob_table = _precompute_all_probs(winner_model, team_store, season)

    # ── Extract R1 info from bracket_input ───────────────────────────────────
    # List per conference: [(higher, lower, matchup_id), ...]
    # Order must be: [1v8, 2v7, 3v6, 4v5]
    conf_r1: dict[str, list[tuple[str, str, str]]] = {"East": [], "West": []}
    for _, row in bracket_input.iterrows():
        conf = str(row["conference"])
        conf_r1[conf].append(
            (str(row["higher_seed"]), str(row["lower_seed"]), str(row["matchup_id"]))
        )

    # ── Build per-R1-matchup probabilities for output reporting ─────────────
    r1_probs: dict[str, tuple[str, str, float]] = {}
    for _conf, matchups in conf_r1.items():
        for higher, lower, mid in matchups:
            p = prob_table.get((higher, lower), prob_table.get((lower, higher), 0.5))
            if (higher, lower) not in prob_table:
                p = 1.0 - p
            r1_probs[mid] = (higher, lower, p)
            logger.info(
                "R1 %s: %s vs %s  →  P(%s wins) = %.1f%%", mid, higher, lower, higher, p * 100
            )

    # ── Monte Carlo loop ─────────────────────────────────────────────────────
    all_teams = set(team_store["team"].tolist())
    champion_counts: dict[str, int] = dict.fromkeys(all_teams, 0)
    adv: dict[str, dict[str, int]] = {
        "r2": dict.fromkeys(all_teams, 0),
        "conf_finals": dict.fromkeys(all_teams, 0),
        "finals": dict.fromkeys(all_teams, 0),
    }

    for _ in range(n_simulations):
        conf_champs: dict[str, str] = {}

        for conf in ("East", "West"):
            matchups = conf_r1[conf]  # [(higher, lower, mid), ...]  order: 1v8,2v7,3v6,4v5

            # R1
            r1_w = []
            for higher, lower, mid in matchups:
                p = r1_probs[mid][2]
                r1_w.append(higher if rng.random() < p else lower)
            # r1_w: [w_1v8, w_2v7, w_3v6, w_4v5]

            # R2 — Semifinal A: w(1v8) vs w(4v5), Semifinal B: w(2v7) vs w(3v6)
            h_a, l_a, p_a = _lookup_prob(prob_table, r1_w[0], r1_w[3], team_store)
            h_b, l_b, p_b = _lookup_prob(prob_table, r1_w[1], r1_w[2], team_store)
            w_semi_a = h_a if rng.random() < p_a else l_a
            w_semi_b = h_b if rng.random() < p_b else l_b
            adv["r2"][w_semi_a] += 1
            adv["r2"][w_semi_b] += 1

            # Conference Finals
            h_cf, l_cf, p_cf = _lookup_prob(prob_table, w_semi_a, w_semi_b, team_store)
            conf_champ = h_cf if rng.random() < p_cf else l_cf
            adv["conf_finals"][conf_champ] += 1
            conf_champs[conf] = conf_champ

        # NBA Finals
        east, west = conf_champs["East"], conf_champs["West"]
        h_fin, l_fin, p_fin = _lookup_prob(prob_table, east, west, team_store)
        adv["finals"][h_fin] += 1
        adv["finals"][l_fin] += 1
        champion = h_fin if rng.random() < p_fin else l_fin
        champion_counts[champion] += 1

    # ── Build outputs ────────────────────────────────────────────────────────
    series_results = []
    for conf, matchups in conf_r1.items():
        for higher, lower, mid in matchups:
            p_higher = r1_probs[mid][2]
            feat_row = bracket_input[bracket_input["matchup_id"] == mid]
            length_info = _predict_length(length_model, feat_row)
            series_results.append(
                {
                    "matchup_id": mid,
                    "round": "first_round",
                    "conference": conf,
                    "higher_seed": higher,
                    "lower_seed": lower,
                    "p_higher_seed_wins": round(p_higher, 4),
                    "p_lower_seed_wins": round(1 - p_higher, 4),
                    "predicted_winner": higher if p_higher >= 0.5 else lower,
                    "p_length_4": length_info["p_length_4"],
                    "p_length_5": length_info["p_length_5"],
                    "p_length_6": length_info["p_length_6"],
                    "p_length_7": length_info["p_length_7"],
                    "expected_length": length_info["expected_length"],
                    "modal_length": length_info["modal_length"],
                }
            )

    n = n_simulations
    champion_probs = {t: c / n for t, c in champion_counts.items() if c > 0}
    r2_probs = {t: c / n for t, c in adv["r2"].items() if c > 0}
    cf_probs = {t: c / n for t, c in adv["conf_finals"].items() if c > 0}
    finals_probs = {t: c / n for t, c in adv["finals"].items() if c > 0}

    return {
        "series_results": series_results,
        "champion_probs": champion_probs,
        "r2_probs": r2_probs,
        "conf_finals_probs": cf_probs,
        "finals_probs": finals_probs,
        "prob_table": prob_table,
    }


# ── Greedy bracket builder ────────────────────────────────────────────────────


def _series_pick(
    team_a: str,
    team_b: str,
    prob_table: dict[tuple[str, str], float],
    team_store: pd.DataFrame,
    round_name: str,
    conference: str,
    length_model: Any | None = None,
    season: int = 2026,
    upset_threshold: float = 0.5,
) -> dict:
    """Return a series pick dict for any team pair.

    Determines higher/lower seed by comparing seed numbers, looks up the
    pre-computed win probability, and estimates series length from NRtg delta.

    upset_threshold: pick the lower seed (upset) when p_higher_seed < this value.
      Default 0.5 = standard behaviour. Set e.g. 0.55 to call any sub-55% matchup
      an upset, giving the bracket more predicted upsets.
    """
    seed_a = int(_team_attr(team_store, team_a, "seed") or 99)
    seed_b = int(_team_attr(team_store, team_b, "seed") or 99)

    if seed_a <= seed_b:
        higher, lower = team_a, team_b
    else:
        higher, lower = team_b, team_a

    key = (higher, lower)
    p_higher = prob_table.get(key, prob_table.get((lower, higher), 0.5))
    if key not in prob_table and (lower, higher) in prob_table:
        p_higher = 1.0 - p_higher

    winner = higher if p_higher >= upset_threshold else lower
    p_winner = p_higher if p_higher >= upset_threshold else 1.0 - p_higher

    seed_diff = float(abs(seed_b - seed_a))
    feat_row = pd.DataFrame(
        [
            _build_matchup_row(
                higher,
                lower,
                seed_diff,
                conference,
                f"{higher}v{lower}",
                round_name,
                season,
                team_store,
            )
        ]
    )
    length_info = _predict_length(length_model, feat_row)

    return {
        "round": round_name,
        "conference": conference,
        "higher_seed": higher,
        "lower_seed": lower,
        "predicted_winner": winner,
        "p_winner": round(p_winner, 4),
        "p_higher_seed_wins": round(p_higher, 4),
        "expected_length": length_info["expected_length"],
        "modal_length": length_info["modal_length"],
        "p_length_4": length_info["p_length_4"],
        "p_length_5": length_info["p_length_5"],
        "p_length_6": length_info["p_length_6"],
        "p_length_7": length_info["p_length_7"],
    }


def build_greedy_bracket(
    results: dict,
    team_store: pd.DataFrame,
    length_model: Any | None = None,
    season: int = 2026,
    upset_threshold: float = 0.5,
) -> list[dict]:
    """Build the full greedy bracket (all 4 rounds) from pre-computed probs.

    Always picks the model's favorite at each stage.  R2/CF/Finals matchups
    flow from the predicted winners of the prior round.

    Returns a list of 15 series dicts in round order:
      8 × first_round, 4 × second_round, 2 × conf_finals, 1 × nba_finals
    """
    prob_table = results["prob_table"]
    r1_series = results["series_results"]

    all_picks: list[dict] = []
    cf_winners: dict[str, str] = {}

    for conf in ("East", "West"):
        # R1 series for this conf, ordered by slot (matchup_id sorts correctly: E1v8 < E2v7 …)
        conf_r1 = sorted(
            [s for s in r1_series if s["conference"] == conf],
            key=lambda s: s["matchup_id"],
        )
        # Slots: 0=1v8, 1=2v7, 2=3v6, 3=4v5
        for s in conf_r1:
            p_h = round(s["p_higher_seed_wins"], 4)
            r1_winner = s["lower_seed"] if p_h < upset_threshold else s["higher_seed"]
            all_picks.append(
                {
                    "round": "first_round",
                    "conference": conf,
                    "higher_seed": s["higher_seed"],
                    "lower_seed": s["lower_seed"],
                    "predicted_winner": r1_winner,
                    "p_winner": round(max(p_h, 1 - p_h), 4),
                    "p_higher_seed_wins": p_h,
                    "expected_length": s.get("expected_length", 0.0),
                    "modal_length": s.get("modal_length", 6),
                    "p_length_4": s.get("p_length_4", 0.0),
                    "p_length_5": s.get("p_length_5", 0.0),
                    "p_length_6": s.get("p_length_6", 0.0),
                    "p_length_7": s.get("p_length_7", 0.0),
                }
            )

        r1_winners = [s["predicted_winner"] for s in all_picks[-4:]]

        # R2 — Semi A: winner(1v8) vs winner(4v5), Semi B: winner(2v7) vs winner(3v6)
        semi_a = _series_pick(
            r1_winners[0],
            r1_winners[3],
            prob_table,
            team_store,
            "second_round",
            conf,
            length_model,
            season,
            upset_threshold,
        )
        semi_b = _series_pick(
            r1_winners[1],
            r1_winners[2],
            prob_table,
            team_store,
            "second_round",
            conf,
            length_model,
            season,
            upset_threshold,
        )
        all_picks.extend([semi_a, semi_b])
        r2_winners = [semi_a["predicted_winner"], semi_b["predicted_winner"]]

        # Conference Finals
        cf = _series_pick(
            r2_winners[0],
            r2_winners[1],
            prob_table,
            team_store,
            "conf_finals",
            conf,
            length_model,
            season,
            upset_threshold,
        )
        all_picks.append(cf)
        cf_winners[conf] = cf["predicted_winner"]

    # NBA Finals
    finals = _series_pick(
        cf_winners["East"],
        cf_winners["West"],
        prob_table,
        team_store,
        "nba_finals",
        "Finals",
        length_model,
        season,
        upset_threshold,
    )
    all_picks.append(finals)
    return all_picks


# ── Output ────────────────────────────────────────────────────────────────────


def save_predictions(results: dict, season: int) -> Path:
    out_dir = cfg.project_root / "data" / "predictions" / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full bracket (all 4 rounds), not just R1
    bracket_df = pd.DataFrame(results.get("full_bracket", results["series_results"]))
    series_path = out_dir / "bracket_output.csv"
    bracket_df.to_csv(series_path, index=False)

    all_teams = sorted(
        set(results["r2_probs"])
        | set(results["conf_finals_probs"])
        | set(results["finals_probs"])
        | set(results["champion_probs"])
    )
    rows = [
        {
            "team": t,
            "p_r2": round(results["r2_probs"].get(t, 0.0), 4),
            "p_conf_finals": round(results["conf_finals_probs"].get(t, 0.0), 4),
            "p_finals": round(results["finals_probs"].get(t, 0.0), 4),
            "p_champion": round(results["champion_probs"].get(t, 0.0), 4),
        }
        for t in all_teams
    ]
    rows.sort(key=lambda r: -r["p_champion"])
    champ_df = pd.DataFrame(rows)
    champ_path = out_dir / "champion_probabilities.csv"
    champ_df.to_csv(champ_path, index=False)

    logger.info("Saved: %s | %s", series_path.name, champ_path.name)
    return series_path


def print_full_bracket(results: dict, season: int) -> None:
    """Print the full greedy bracket plus probability table."""
    bracket = results["full_bracket"]

    ROUND_LABELS = {
        "first_round": "Round 1",
        "second_round": "Round 2",
        "conf_finals": "Conference Finals",
        "nba_finals": "NBA Finals",
    }
    ROUND_ORDER = ["first_round", "second_round", "conf_finals"]

    print(f"\n{'='*60}")
    print(f"  {season} NBA PLAYOFFS — FULL BRACKET")
    print(f"{'='*60}")

    for conf in ("East", "West"):
        print(f"\n  {conf.upper()}")
        for rnd in ROUND_ORDER:
            series_in_round = [s for s in bracket if s["round"] == rnd and s["conference"] == conf]
            if not series_in_round:
                continue
            print(f"  {'─'*3} {ROUND_LABELS[rnd]} {'─'*40}"[:55])
            for s in series_in_round:
                upset_marker = " *" if s["predicted_winner"] == s["lower_seed"] else ""
                print(
                    f"    {s['higher_seed']:3s} vs {s['lower_seed']:3s}"
                    f"  →  {s['predicted_winner']:3s}"
                    f"  ({s['p_winner']*100:.1f}%, in {s['modal_length']})"
                    f"{upset_marker}"
                )

    # NBA Finals
    finals_list = [s for s in bracket if s["round"] == "nba_finals"]
    if finals_list:
        f = finals_list[0]
        upset_marker = " *" if f["predicted_winner"] == f["lower_seed"] else ""
        print(f"\n  {'─'*3} NBA FINALS {'─'*43}"[:55])
        print(
            f"    {f['higher_seed']:3s} vs {f['lower_seed']:3s}"
            f"  →  {f['predicted_winner']:3s}"
            f"  ({f['p_winner']*100:.1f}%, in {f['modal_length']})"
            f"{upset_marker}"
        )
    if results.get("upset_threshold", 0.5) > 0.5:
        print(
            f"\n  (* = upset pick — lower seed predicted to win; threshold={results['upset_threshold']:.0%})"
        )

    # Probability table
    print(f"\n{'='*60}")
    print(f"  ADVANCEMENT PROBABILITIES (from {N_SIMULATIONS:,} simulations)")
    print(f"{'='*60}")
    print(f"  {'Team':<6} {'R2':>6} {'ConfF':>7} {'Finals':>7} {'Champ':>7}")
    print(f"  {'─'*40}")
    seen: set[str] = set()
    for team, _ in sorted(results["champion_probs"].items(), key=lambda x: -x[1]):
        seen.add(team)
        print(
            f"  {team:<6}"
            f" {results['r2_probs'].get(team, 0)*100:5.1f}%"
            f" {results['conf_finals_probs'].get(team, 0)*100:6.1f}%"
            f" {results['finals_probs'].get(team, 0)*100:6.1f}%"
            f" {results['champion_probs'].get(team, 0)*100:6.1f}%"
        )
    for team in sorted(results["r2_probs"], key=lambda t: -results["r2_probs"][t]):
        if team not in seen:
            seen.add(team)
            print(
                f"  {team:<6}"
                f" {results['r2_probs'].get(team, 0)*100:5.1f}%"
                f" {results['conf_finals_probs'].get(team, 0)*100:6.1f}%"
                f" {results['finals_probs'].get(team, 0)*100:6.1f}%"
                f"   <0.1%"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=cfg.seasons["current"])
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument(
        "--upset-threshold",
        type=float,
        default=0.5,
        help="Pick the lower seed (upset) when P(higher seed wins) < this value. "
        "Default 0.5 = standard. Try 0.55 for more upset picks.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    model_dir = cfg.project_root / cfg.paths["models"]["trained"]
    model_path = model_dir / "ensemble_winner.pkl"
    if not model_path.exists():
        model_path = model_dir / "xgboost_winner.pkl"
    if not model_path.exists():
        model_path = model_dir / "lightgbm_winner.pkl"

    winner_model = _load_model(model_path)
    logger.info("Loaded winner model: %s", model_path.name)

    length_model = None
    length_path = model_dir / "lgbm_length.pkl"
    if length_path.exists():
        try:
            length_model = _load_model(length_path)
            logger.info("Loaded length model: %s", length_path.name)
        except Exception as e:
            logger.warning("Could not load length model (%s) — falling back to Monte Carlo", e)
    else:
        logger.warning("Length model not found at %s — falling back to Monte Carlo", length_path)

    bracket_input, team_store = _load_bracket_files(args.season)
    logger.info("Bracket: %d R1 matchups, %d teams", len(bracket_input), len(team_store))

    results = simulate_full_bracket(
        bracket_input,
        team_store,
        winner_model,
        season=args.season,
        n_simulations=args.n_simulations,
        length_model=length_model,
    )
    results["upset_threshold"] = args.upset_threshold
    results["full_bracket"] = build_greedy_bracket(
        results,
        team_store,
        length_model=length_model,
        season=args.season,
        upset_threshold=args.upset_threshold,
    )
    save_predictions(results, args.season)
    print_full_bracket(results, args.season)


if __name__ == "__main__":
    main()
