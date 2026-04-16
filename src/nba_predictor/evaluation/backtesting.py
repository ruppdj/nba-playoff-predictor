"""Walk-forward backtesting of playoff bracket predictions.

Trains a model on all seasons before the test season, predicts the test
season's bracket, and aggregates performance across all test seasons.

Run via: python -m nba_predictor.evaluation.backtesting
      or: make evaluate
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nba_predictor.config import cfg, get_git_hash
from nba_predictor.evaluation.cv_strategy import playoff_season_cv_splits
from nba_predictor.evaluation.metrics import compute_winner_metrics

logger = logging.getLogger(__name__)

FEATURE_COLS = (
    cfg.features.get("matchup", [])
    + [f"higher_{c}" for c in cfg.features.get("injury", [])]
    + [f"lower_{c}" for c in cfg.features.get("injury", [])]
    + ["era_showtime", "era_defensive", "era_transition", "era_analytics", "season_flag"]
)

TARGET_COL = "higher_seed_wins"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns that actually exist in the DataFrame."""
    return [c for c in FEATURE_COLS if c in df.columns]


def backtest(
    series_df: pd.DataFrame,
    model_cls: Any = LogisticRegression,
    model_kwargs: dict | None = None,
    season_col: str = "season",
) -> pd.DataFrame:
    """Run walk-forward backtesting and return per-fold metrics.

    Args:
        series_df: Full series dataset (one row per historical series).
        model_cls: Sklearn-compatible classifier class.
        model_kwargs: Keyword arguments for the classifier.
        season_col: Name of the season column.

    Returns:
        DataFrame with one row per test fold containing all metrics.
    """
    if model_kwargs is None:
        model_kwargs = {"random_state": cfg.modeling["random_state"], "max_iter": 1000}

    feature_cols = get_feature_cols(series_df)
    logger.info("Backtesting with %d features on %d series", len(feature_cols), len(series_df))

    results = []
    for fold_i, (train_idx, test_idx) in enumerate(playoff_season_cv_splits(series_df, season_col)):
        train = series_df.loc[train_idx].copy()
        test = series_df.loc[test_idx].copy()
        test_season = test[season_col].iloc[0]

        X_train = train[feature_cols].fillna(0)
        y_train = train[TARGET_COL].values
        X_test = test[feature_cols].fillna(0)
        y_test = test[TARGET_COL].values

        # Scale features
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Fit model
        model = model_cls(**model_kwargs)
        model.fit(X_train_sc, y_train)

        # Predict
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]

        # Metrics
        fold_metrics = compute_winner_metrics(y_test, y_pred, y_prob)
        fold_metrics["fold"] = fold_i
        fold_metrics["test_season"] = test_season
        fold_metrics["n_train_series"] = len(train)
        fold_metrics["n_test_series"] = len(test)
        results.append(fold_metrics)

        logger.info(
            "Fold %d (test=%d): acc=%.3f, log_loss=%.3f, brier=%.3f",
            fold_i,
            test_season,
            fold_metrics["accuracy"],
            fold_metrics["log_loss"],
            fold_metrics["brier_score"],
        )

    results_df = pd.DataFrame(results)
    return results_df


def backtest_and_log(
    series_df: pd.DataFrame,
    experiment_name: str = "backtest",
    model_cls: Any = LogisticRegression,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Run backtesting and log aggregate metrics to MLflow."""
    mlflow.set_tracking_uri(str(cfg.project_root / cfg.mlflow["tracking_uri"]))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="walk_forward_backtest"):
        mlflow.set_tag("git_commit", get_git_hash())
        mlflow.set_tag("model_type", model_cls.__name__)

        results_df = backtest(series_df, model_cls, model_kwargs)

        # Log aggregate metrics
        for metric in ["accuracy", "log_loss", "brier_score", "ece", "upset_recall"]:
            if metric in results_df.columns:
                mlflow.log_metric(f"{metric}_mean", results_df[metric].mean())
                mlflow.log_metric(f"{metric}_std", results_df[metric].std())

        # Save results CSV as artifact
        out_path = cfg.project_root / "reports" / "backtest_results.csv"
        results_df.to_csv(out_path, index=False)
        mlflow.log_artifact(str(out_path))

        logger.info(
            "Backtest complete: mean_acc=%.3f ± %.3f, mean_logloss=%.3f ± %.3f",
            results_df["accuracy"].mean(),
            results_df["accuracy"].std(),
            results_df["log_loss"].mean(),
            results_df["log_loss"].std(),
        )

    return results_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    series_path = cfg.path("processed", "series_dataset")
    if not series_path.exists():
        raise SystemExit(f"Series dataset not found: {series_path}\nRun 'make process' first.")

    series_df = pd.read_parquet(series_path)
    backtest_and_log(series_df)


if __name__ == "__main__":
    main()
