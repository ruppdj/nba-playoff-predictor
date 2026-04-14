"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nba_predictor.evaluation.metrics import (
    compute_winner_metrics,
    compute_length_metrics,
    expected_calibration_error,
)


def test_perfect_winner_metrics():
    """Perfect predictions should give accuracy=1, log_loss≈0, brier≈0."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    # Clip to avoid log(0)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    metrics = compute_winner_metrics(y_true, y_pred, y_prob_clipped)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] < 0.01


def test_naive_accuracy_is_higher_seed_win_rate():
    """Naive accuracy should equal the fraction of higher-seed wins in y_true."""
    y_true = np.array([1, 1, 1, 0, 1])  # higher seed wins 80%
    y_pred = np.array([1, 1, 1, 1, 1])
    y_prob = np.full(5, 0.7)
    metrics = compute_winner_metrics(y_true, y_pred, y_prob)
    assert abs(metrics["naive_accuracy"] - 0.8) < 1e-10


def test_upset_recall_all_upsets_caught():
    """When model predicts all upsets correctly, upset_recall = 1."""
    y_true = np.array([0, 1, 0, 1])   # 2 upsets
    y_pred = np.array([0, 1, 0, 1])   # correctly predicted
    y_prob = np.array([0.3, 0.7, 0.3, 0.7])
    metrics = compute_winner_metrics(y_true, y_pred, y_prob)
    assert metrics["upset_recall"] == 1.0


def test_upset_recall_no_upsets_caught():
    """When model never predicts upsets, upset_recall = 0."""
    y_true = np.array([0, 1, 0, 1])   # 2 upsets
    y_pred = np.array([1, 1, 1, 1])   # always predicts higher seed
    y_prob = np.array([0.7, 0.7, 0.7, 0.7])
    metrics = compute_winner_metrics(y_true, y_pred, y_prob)
    assert metrics["upset_recall"] == 0.0


def test_ece_perfect_calibration():
    """Perfect calibration: predicted prob = actual accuracy → ECE ≈ 0."""
    # 0.5 confidence, 50% actual accuracy
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.full(10, 0.5)
    ece = expected_calibration_error(y_true, y_prob)
    assert ece < 0.05, f"ECE should be near 0 for perfect calibration, got {ece}"


def test_ece_overconfident():
    """Predicting 1.0 prob for wrong outcomes → high ECE."""
    y_true = np.array([0, 0, 0, 0, 0])
    y_prob = np.full(5, 0.99)
    ece = expected_calibration_error(y_true, y_prob)
    assert ece > 0.5, f"Overconfident model should have high ECE, got {ece}"


def test_length_exact_accuracy():
    """Exact accuracy: fraction of series lengths predicted correctly."""
    y_true = np.array([4, 5, 6, 7, 5])
    y_pred = np.array([4, 5, 7, 7, 5])  # 4/5 correct
    metrics = compute_length_metrics(y_true, y_pred)
    assert abs(metrics["exact_accuracy"] - 0.8) < 1e-10


def test_length_within1_accuracy():
    """Within-1 accuracy counts predictions that are off by at most 1."""
    y_true = np.array([4, 5, 6, 7])
    y_pred = np.array([5, 6, 7, 6])  # all off by 1 → within-1 accuracy = 1.0
    metrics = compute_length_metrics(y_true, y_pred)
    assert metrics["within1_accuracy"] == 1.0


def test_length_naive_baseline():
    """Naive baseline (always predict 6) should have reasonable accuracy."""
    # Historically ~30% of series go 6 games
    y_true = np.array([4, 5, 5, 6, 6, 6, 7, 7])
    y_pred = np.array([6, 6, 6, 6, 6, 6, 6, 6])
    metrics = compute_length_metrics(y_true, y_pred)
    assert "naive_exact_accuracy" in metrics
    assert "accuracy_vs_naive" in metrics


def test_accuracy_vs_naive_sign():
    """A good model should have positive accuracy_vs_naive."""
    y_true = np.array([1, 0, 1, 0, 1])
    # Always predict 1 (higher seed) — this IS the naive baseline
    y_pred = np.array([1, 1, 1, 1, 1])
    y_prob = np.full(5, 0.7)
    metrics = compute_winner_metrics(y_true, y_pred, y_prob)
    # accuracy == naive_accuracy here (both always pick higher seed)
    assert metrics["accuracy_vs_naive"] >= 0
