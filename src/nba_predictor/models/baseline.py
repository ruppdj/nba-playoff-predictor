"""Baseline models — Logistic Regression series winner predictor.

Serves as the performance floor: every advanced model must beat this.
Also implements the naive "always pick higher seed" baseline.

Historical benchmark:
  - Naive (always higher seed): ~71% accuracy
  - Logistic Regression: typically ~73-75%
  - Good model target: 76-78% accuracy, log_loss < 0.52

Run via: python -m nba_predictor.models.baseline
      or: make train
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_predictor.config import cfg
from nba_predictor.evaluation.cv_strategy import playoff_season_cv_splits
from nba_predictor.evaluation.metrics import compute_winner_metrics
from nba_predictor.tracking.mlflow_logger import log_training_run, setup_mlflow

logger = logging.getLogger(__name__)

RANDOM_STATE = cfg.modeling["random_state"]
EXPERIMENT_NAME = "series_winner_baseline"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all available feature columns from the config."""
    all_feat = (
        cfg.features.get("matchup", [])
        + cfg.features.get("meta", [])
        + [f"higher_{c}" for c in cfg.features.get("injury", [])]
        + [f"lower_{c}" for c in cfg.features.get("injury", [])]
    )
    return [c for c in all_feat if c in df.columns]


def build_lr_pipeline(C: float = 1.0) -> Pipeline:
    """Build a Logistic Regression pipeline with StandardScaler."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def run_cv_baseline(series_df: pd.DataFrame, C: float = 1.0) -> dict[str, list[float]]:
    """Run walk-forward CV for logistic regression baseline.

    Returns dict of metric_name → [fold_values].
    """
    feature_cols = get_feature_cols(series_df)
    target_col = "higher_seed_wins"

    metrics_history: dict[str, list[float]] = {
        "accuracy": [],
        "log_loss": [],
        "brier_score": [],
        "upset_recall": [],
        "ece": [],
        "naive_accuracy": [],
    }

    for fold_i, (train_idx, test_idx) in enumerate(playoff_season_cv_splits(series_df)):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        X_train = train[feature_cols].fillna(0).values
        y_train = train[target_col].values
        X_test = test[feature_cols].fillna(0).values
        y_test = test[target_col].values

        pipe = build_lr_pipeline(C)
        calibrated = CalibratedClassifierCV(pipe, cv=3, method="isotonic")
        try:
            calibrated.fit(X_train, y_train)
        except ValueError:
            # Single-class fold — fit uncalibrated then wrap with prefit calibration
            logger.warning("Fold %d: single class in training set, skipping calibration.", fold_i)
            pipe.fit(X_train, y_train)
            calibrated = CalibratedClassifierCV(pipe, cv="prefit", method="isotonic")
            calibrated.fit(X_train, y_train)

        y_pred = calibrated.predict(X_test)
        y_prob = calibrated.predict_proba(X_test)[:, 1]

        fold_metrics = compute_winner_metrics(y_test, y_pred, y_prob)
        for k in metrics_history:
            if k in fold_metrics:
                metrics_history[k].append(fold_metrics[k])

    return metrics_history


def train_final_model(series_df: pd.DataFrame, C: float = 1.0) -> CalibratedClassifierCV:
    """Train a final logistic regression model on all available data."""
    feature_cols = get_feature_cols(series_df)
    X = series_df[feature_cols].fillna(0).values
    y = series_df["higher_seed_wins"].values

    pipe = build_lr_pipeline(C)
    calibrated = CalibratedClassifierCV(pipe, cv=3, method="isotonic")
    calibrated.fit(X, y)
    return calibrated


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    series_path = cfg.path("processed", "series_dataset")
    if not series_path.exists():
        raise SystemExit(f"Series dataset not found: {series_path}\nRun 'make process' first.")

    series_df = pd.read_parquet(series_path)
    feature_cols = get_feature_cols(series_df)
    logger.info("Training baseline: %d series, %d features", len(series_df), len(feature_cols))

    setup_mlflow(EXPERIMENT_NAME)

    # CV evaluation
    C = 1.0
    cv_metrics = run_cv_baseline(series_df, C)

    # Log CV summary
    logger.info(
        "Baseline CV: acc=%.3f±%.3f, logloss=%.3f±%.3f, brier=%.3f±%.3f",
        np.mean(cv_metrics["accuracy"]),
        np.std(cv_metrics["accuracy"]),
        np.mean(cv_metrics["log_loss"]),
        np.std(cv_metrics["log_loss"]),
        np.mean(cv_metrics["brier_score"]),
        np.std(cv_metrics["brier_score"]),
    )

    # Train final model on all data
    final_model = train_final_model(series_df, C)

    # Log to MLflow
    params = {
        "model": "LogisticRegression",
        "C": C,
        "calibration": "isotonic",
        "target": "series_winner",
        "n_training_series": len(series_df),
    }
    log_training_run(
        model=final_model,
        params=params,
        cv_metrics=cv_metrics,
        feature_names=feature_cols,
        run_name="logistic_regression_baseline",
        register_as=None,  # don't auto-register baseline
    )

    logger.info("Baseline training complete.")


if __name__ == "__main__":
    main()
