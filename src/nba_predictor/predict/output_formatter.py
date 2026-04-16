"""Bracket prediction output formatting.

Reads bracket_output.csv and champion_probabilities.csv and renders
human-readable summaries in Markdown and HTML format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nba_predictor.config import cfg

logger = logging.getLogger(__name__)

ROUND_ORDER = ["first_round", "second_round", "conf_finals", "nba_finals"]
ROUND_LABELS = {
    "first_round": "First Round",
    "second_round": "Conference Semifinals",
    "conf_finals": "Conference Finals",
    "nba_finals": "NBA Finals",
}


def format_bracket_markdown(season: int) -> str:
    """Generate a Markdown-formatted bracket summary."""
    pred_dir = cfg.project_root / "data" / "predictions" / str(season)
    series_path = pred_dir / "bracket_output.csv"
    champ_path = pred_dir / "champion_probabilities.csv"

    if not series_path.exists():
        return f"No predictions found for season {season}. Run `make predict` first."

    series_df = pd.read_csv(series_path)
    champ_df = pd.read_csv(champ_path) if champ_path.exists() else pd.DataFrame()

    lines = [f"# {season} NBA Playoffs — Predicted Bracket", ""]

    for conf in ["East", "West"]:
        lines.append(f"## {conf}ern Conference")
        for round_key in ROUND_ORDER[:-1]:
            round_label = ROUND_LABELS[round_key]
            subset = series_df[
                (series_df["conference"] == conf) & (series_df["round"] == round_key)
            ]
            if subset.empty:
                continue
            lines.append(f"\n### {round_label}")
            for _, row in subset.iterrows():
                winner = row["predicted_winner"]
                loser = row["lower_seed"] if winner == row["higher_seed"] else row["higher_seed"]
                p_win = (
                    row["p_higher_seed_wins"]
                    if winner == row["higher_seed"]
                    else 1 - row["p_higher_seed_wins"]
                )
                length_str = (
                    f"in {int(row['modal_length'])}"
                    if "modal_length" in row.index and pd.notna(row["modal_length"])
                    else f"~{row['expected_length']:.1f}g"
                )
                lines.append(f"- **{winner}** def. {loser} " f"(P={p_win:.1%}, {length_str})")

    # NBA Finals
    finals = series_df[series_df["round"] == "nba_finals"]
    if not finals.empty:
        lines.append("\n## NBA Finals")
        row = finals.iloc[0]
        winner = row["predicted_winner"]
        loser = row["lower_seed"] if winner == row["higher_seed"] else row["higher_seed"]
        p_win = (
            row["p_higher_seed_wins"]
            if winner == row["higher_seed"]
            else 1 - row["p_higher_seed_wins"]
        )
        length_str = (
            f"in {int(row['modal_length'])}"
            if "modal_length" in row.index and pd.notna(row["modal_length"])
            else f"~{row['expected_length']:.1f}g"
        )
        lines.append(f"**🏆 {winner}** def. {loser} " f"(P={p_win:.1%}, {length_str})")

    # Champion probabilities
    if not champ_df.empty:
        lines.append("\n## Championship Probabilities (Monte Carlo)")
        lines.append("| Team | Probability |")
        lines.append("|------|-------------|")
        champ_col = "p_champion" if "p_champion" in champ_df.columns else "champion_probability"
        for _, row in champ_df.head(16).iterrows():
            lines.append(f"| {row['team']} | {row[champ_col]:.1%} |")

    return "\n".join(lines)


def save_markdown_report(season: int) -> Path:
    """Write formatted bracket Markdown to reports/."""
    md = format_bracket_markdown(season)
    out_path = cfg.project_root / "reports" / f"bracket_{season}.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info("Bracket report saved: %s", out_path)
    return out_path
