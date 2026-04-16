"""XGBoost and LightGBM series winner models with Optuna hyperparameter tuning.

Run via: python -m nba_predictor.models.gradient_boosting --model xgboost
      or: python -m nba_predictor.models.gradient_boosting --model lightgbm
      or: make train
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from nba_predictor.config import cfg
from nba_predictor.evaluation.cv_strategy import playoff_season_cv_splits
from nba_predictor.evaluation.metrics import compute_winner_metrics
from nba_predictor.tracking.mlflow_logger import log_training_run, setup_mlflow

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = cfg.modeling["random_state"]
N_TRIALS = cfg.modeling["optuna"]["n_trials"]
TIMEOUT = cfg.modeling["optuna"]["timeout_seconds"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    all_feat = (
        cfg.features.get("matchup", [])
        + cfg.features.get("meta", [])
        + [f"higher_{c}" for c in cfg.features.get("injury", [])]
        + [f"lower_{c}" for c in cfg.features.get("injury", [])]
    )
    return [c for c in all_feat if c in df.columns]


# =============================================================================
# XGBoost
# =============================================================================


def _xgb_objective(trial: optuna.Trial, series_df: pd.DataFrame, feature_cols: list[str]) -> float:
    """Optuna objective: minimize log_loss on walk-forward CV."""
    import xgboost as xgb

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    log_losses = []
    for train_idx, test_idx in playoff_season_cv_splits(series_df):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        X_train = train[feature_cols].fillna(0)
        y_train = train["higher_seed_wins"].values
        X_test = test[feature_cols].fillna(0)
        y_test = test["higher_seed_wins"].values

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import log_loss as sk_log_loss

        log_losses.append(sk_log_loss(y_test, y_prob))

    return float(np.mean(log_losses))


def tune_xgboost(series_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Run Optuna study for XGBoost. Returns best params."""
    logger.info("Tuning XGBoost with Optuna (%d trials)...", N_TRIALS)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _xgb_objective(trial, series_df, feature_cols),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True,
    )
    logger.info("XGBoost best log_loss: %.4f", study.best_value)
    return study.best_params


# =============================================================================
# LightGBM
# =============================================================================


def _lgbm_objective(trial: optuna.Trial, series_df: pd.DataFrame, feature_cols: list[str]) -> float:
    """Optuna objective for LightGBM."""
    import lightgbm as lgb

    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    log_losses = []
    for train_idx, test_idx in playoff_season_cv_splits(series_df):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        X_train = train[feature_cols].fillna(0)
        y_train = train["higher_seed_wins"].values
        X_test = test[feature_cols].fillna(0)
        y_test = test["higher_seed_wins"].values

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import log_loss as sk_log_loss

        log_losses.append(sk_log_loss(y_test, y_prob))

    return float(np.mean(log_losses))


def tune_lightgbm(series_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Run Optuna study for LightGBM. Returns best params."""
    logger.info("Tuning LightGBM with Optuna (%d trials)...", N_TRIALS)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _lgbm_objective(trial, series_df, feature_cols),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True,
    )
    logger.info("LightGBM best log_loss: %.4f", study.best_value)
    return study.best_params


# =============================================================================
# CV evaluation with best params
# =============================================================================


def run_cv_with_params(
    series_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    best_params: dict,
) -> tuple[dict[str, list[float]], object]:
    """Run final walk-forward CV with best params. Returns metrics and final model."""
    metrics_history: dict[str, list[float]] = {
        "accuracy": [],
        "log_loss": [],
        "brier_score": [],
        "upset_recall": [],
        "ece": [],
    }

    final_model = None
    for _fold_i, (train_idx, test_idx) in enumerate(playoff_season_cv_splits(series_df)):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        X_train = train[feature_cols].fillna(0)
        y_train = train["higher_seed_wins"].values
        X_test = test[feature_cols].fillna(0)
        y_test = test["higher_seed_wins"].values

        if model_type == "xgboost":
            import xgboost as xgb

            model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
        else:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(**best_params, verbose=-1)

        calibrated = CalibratedClassifierCV(model, cv=5, method="isotonic")
        calibrated.fit(X_train, y_train)

        y_pred = calibrated.predict(X_test)
        y_prob = calibrated.predict_proba(X_test)[:, 1]
        fold_metrics = compute_winner_metrics(y_test, y_pred, y_prob)

        for k in metrics_history:
            if k in fold_metrics:
                metrics_history[k].append(fold_metrics[k])

        # Keep the last fold's model as a proxy for the final model
        final_model = calibrated

    return metrics_history, final_model


def save_model(model: object, model_type: str) -> Path:
    """Pickle the final model to models/trained/."""
    out_dir = cfg.project_root / cfg.paths["models"]["trained"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_type}_winner.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved: %s", out_path)
    return out_path


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--n-trials", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    model_type = args.model

    if args.n_trials:
        global N_TRIALS
        N_TRIALS = args.n_trials

    series_path = cfg.path("processed", "series_dataset")
    if not series_path.exists():
        raise SystemExit(f"Series dataset not found: {series_path}\nRun 'make process' first.")

    series_df = pd.read_parquet(series_path)
    feature_cols = get_feature_cols(series_df)
    logger.info(
        "Training %s: %d series, %d features", model_type, len(series_df), len(feature_cols)
    )

    # Hyperparameter tuning
    if model_type == "xgboost":
        best_params = tune_xgboost(series_df, feature_cols)
        experiment_name = "series_winner_xgboost"
        register_name = cfg.mlflow["model_registry"]["winner_model_name"]
    else:
        best_params = tune_lightgbm(series_df, feature_cols)
        experiment_name = "series_winner_lgbm"
        register_name = None  # don't auto-register lgbm — ensemble will be registered

    # CV evaluation with best params
    cv_metrics, final_model = run_cv_with_params(series_df, feature_cols, model_type, best_params)

    logger.info(
        "%s CV: acc=%.3f±%.3f, logloss=%.3f±%.3f",
        model_type,
        np.mean(cv_metrics["accuracy"]),
        np.std(cv_metrics["accuracy"]),
        np.mean(cv_metrics["log_loss"]),
        np.std(cv_metrics["log_loss"]),
    )

    # Save model
    save_model(final_model, model_type)

    # Log to MLflow
    setup_mlflow(experiment_name)
    params = {
        "model": model_type,
        "target": "series_winner",
        "calibration": "isotonic",
        "n_training_series": len(series_df),
        "n_optuna_trials": N_TRIALS,
        **best_params,
    }
    log_training_run(
        model=final_model,
        params=params,
        cv_metrics=cv_metrics,
        feature_names=feature_cols,
        run_name=f"{model_type}_optuna_tuned",
        register_as=register_name,
    )

    logger.info("%s training complete.", model_type)


if __name__ == "__main__":
    main()
