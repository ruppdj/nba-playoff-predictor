"""Evaluation metrics for series winner and series length prediction.

Metrics:
  - accuracy: fraction of series correctly predicted
  - log_loss: probabilistic loss (lower = better calibrated)
  - brier_score: mean squared probability error
  - upset_recall: how well the model identifies upsets (lower seed wins)
  - bracket_score: ESPN-style bracket scoring (weighted by round)
  - ece: Expected Calibration Error (calibration quality)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def compute_winner_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    series_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute all series-winner evaluation metrics.

    Args:
        y_true: Ground truth binary labels (1 = higher seed wins).
        y_pred: Predicted binary labels.
        y_prob: Predicted probability that higher seed wins.
        series_df: Optional full series DataFrame for upset analysis.

    Returns:
        Dictionary of metric name → value.
    """
    metrics: dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["log_loss"] = float(log_loss(y_true, y_prob))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # Naive baseline: always predict higher seed wins
    naive_pred = np.ones_like(y_true)
    metrics["naive_accuracy"] = float(accuracy_score(y_true, naive_pred))
    metrics["accuracy_vs_naive"] = metrics["accuracy"] - metrics["naive_accuracy"]

    # Upset analysis
    upsets = (y_true == 0)  # lower seed won
    if upsets.sum() > 0:
        upset_preds = (y_pred == 0)  # model predicted lower seed wins
        # Recall: of actual upsets, how many did we catch?
        metrics["upset_recall"] = float(
            (upsets & upset_preds).sum() / upsets.sum()
        )
        # Precision: of predicted upsets, how many were correct?
        if upset_preds.sum() > 0:
            metrics["upset_precision"] = float(
                (upsets & upset_preds).sum() / upset_preds.sum()
            )
        else:
            metrics["upset_precision"] = 0.0
        metrics["n_upsets"] = int(upsets.sum())
        metrics["n_upset_predictions"] = int(upset_preds.sum())
    else:
        metrics["upset_recall"] = np.nan
        metrics["upset_precision"] = np.nan

    # Calibration: ECE (simple binned version)
    metrics["ece"] = float(expected_calibration_error(y_true, y_prob))

    return metrics


def compute_length_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute series length prediction metrics.

    Args:
        y_true: True series lengths (4, 5, 6, or 7).
        y_pred: Predicted series lengths.
        y_prob: Optional class probability matrix (n_samples x 4).

    Returns:
        Dictionary of metric name → value.
    """
    metrics: dict[str, float] = {}

    metrics["exact_accuracy"] = float(accuracy_score(y_true, y_pred))

    # Within-1 accuracy (ordinal metric — off by one is "close")
    within_1 = np.abs(np.array(y_true) - np.array(y_pred)) <= 1
    metrics["within1_accuracy"] = float(within_1.mean())

    # Naive baseline: always predict 6 games (most common length)
    naive_pred = np.full_like(y_pred, 6)
    metrics["naive_exact_accuracy"] = float(accuracy_score(y_true, naive_pred))
    metrics["accuracy_vs_naive"] = metrics["exact_accuracy"] - metrics["naive_exact_accuracy"]

    # Mean absolute error
    metrics["mae"] = float(np.abs(np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)).mean())

    if y_prob is not None:
        classes = [4, 5, 6, 7]
        y_true_encoded = np.array([classes.index(y) for y in y_true])
        metrics["log_loss"] = float(log_loss(y_true_encoded, y_prob, labels=list(range(4))))

    return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the average gap between
    predicted probability and actual accuracy within each bin.

    Target: ECE < 0.05 for a well-calibrated model.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi)
        n_bin = in_bin.sum()
        if n_bin == 0:
            continue
        avg_confidence = y_prob[in_bin].mean()
        avg_accuracy = y_true[in_bin].mean()
        ece += (n_bin / n) * abs(avg_confidence - avg_accuracy)

    return float(ece)


def bracket_score(
    true_bracket: dict[str, str],
    predicted_bracket: dict[str, str],
    round_weights: dict[str, int] | None = None,
) -> int:
    """Compute an ESPN-style bracket score.

    Args:
        true_bracket: Dict mapping series_id → winning team.
        predicted_bracket: Dict mapping series_id → predicted winner.
        round_weights: Points per correct pick per round.
                       Default: {first: 10, second: 20, conf: 40, finals: 80}.

    Returns:
        Total bracket score (integer).
    """
    if round_weights is None:
        round_weights = {
            "first_round": 10,
            "conf_semis": 20,
            "conf_finals": 40,
            "nba_finals": 80,
        }

    score = 0
    for series_id, true_winner in true_bracket.items():
        pred_winner = predicted_bracket.get(series_id, "")
        if pred_winner == true_winner:
            # Determine round from series_id naming convention
            for round_key, pts in round_weights.items():
                if round_key in series_id:
                    score += pts
                    break
            else:
                score += 10  # default
    return score
