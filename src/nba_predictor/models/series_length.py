"""Series length prediction — 4-class model (4, 5, 6, or 7 games).

Two approaches compared:
  1. LightGBM multi-class classifier (class_weight='balanced' for rare sweeps)
  2. Ordinal regression (mord.LogisticIT) — respects natural ordering 4 < 5 < 6 < 7

The ordinal approach is theoretically superior; empirical comparison in notebook 07.

Run via: python -m nba_predictor.models.series_length
      or: make train
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from nba_predictor.config import cfg
from nba_predictor.evaluation.cv_strategy import playoff_season_cv_splits
from nba_predictor.evaluation.metrics import compute_length_metrics
from nba_predictor.tracking.mlflow_logger import setup_mlflow, log_training_run

logger = logging.getLogger(__name__)

RANDOM_STATE = cfg.modeling["random_state"]
EXPERIMENT_NAME = "series_length_lgbm"
LENGTH_CLASSES = [4, 5, 6, 7]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    all_feat = (
        cfg.features.get("matchup", [])
        + cfg.features.get("meta", [])
        + cfg.features.get("series_length", [])
        + [f"higher_{c}" for c in cfg.features.get("injury", [])]
        + [f"lower_{c}" for c in cfg.features.get("injury", [])]
    )
    return [c for c in all_feat if c in df.columns]


def run_lgbm_cv(series_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, list[float]]:
    """Walk-forward CV for LightGBM series length model."""
    import lightgbm as lgb

    metrics_history: dict[str, list[float]] = {
        "exact_accuracy": [], "within1_accuracy": [], "mae": [], "log_loss": [],
    }

    for fold_i, (train_idx, test_idx) in enumerate(playoff_season_cv_splits(series_df)):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        X_train = train[feature_cols].fillna(0)
        y_train = train["series_length"].values
        X_test = test[feature_cols].fillna(0)
        y_test = test["series_length"].values

        # Compute class weights (sweeps = 4 games are rare)
        class_counts = pd.Series(y_train).value_counts()
        total = len(y_train)
        class_weight = {c: total / (len(LENGTH_CLASSES) * class_counts.get(c, 1)) for c in LENGTH_CLASSES}

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            class_weight=class_weight,
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        fold_metrics = compute_length_metrics(y_test, y_pred, y_prob)
        for k in metrics_history:
            if k in fold_metrics:
                metrics_history[k].append(fold_metrics[k])

        logger.info(
            "Length fold %d: exact_acc=%.3f, within1=%.3f",
            fold_i, fold_metrics["exact_accuracy"], fold_metrics["within1_accuracy"],
        )

    return metrics_history


def run_ordinal_cv(series_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, list[float]]:
    """Walk-forward CV for ordinal regression (mord.LogisticIT)."""
    try:
        import mord
    except ImportError:
        logger.warning("mord not installed — skipping ordinal regression.")
        return {}

    metrics_history: dict[str, list[float]] = {
        "exact_accuracy": [], "within1_accuracy": [], "mae": [],
    }

    scaler = StandardScaler()

    # Map length classes to ordinal labels [0, 1, 2, 3]
    label_map = {4: 0, 5: 1, 6: 2, 7: 3}
    inv_map = {0: 4, 1: 5, 2: 6, 3: 7}

    for fold_i, (train_idx, test_idx) in enumerate(playoff_season_cv_splits(series_df)):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        X_train = train[feature_cols].fillna(0)
        y_train_raw = train["series_length"].values
        y_train = np.array([label_map.get(y, 2) for y in y_train_raw])

        X_test = test[feature_cols].fillna(0)
        y_test_raw = test["series_length"].values

        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        model = mord.LogisticIT(alpha=1.0)
        model.fit(X_train_sc, y_train)

        y_pred_ord = model.predict(X_test_sc)
        y_pred = np.array([inv_map.get(p, 6) for p in y_pred_ord])

        fold_metrics = compute_length_metrics(y_test_raw, y_pred)
        for k in metrics_history:
            if k in fold_metrics:
                metrics_history[k].append(fold_metrics[k])

    return metrics_history


def train_final_length_model(series_df: pd.DataFrame, feature_cols: list[str]) -> object:
    """Train final LightGBM length model on all data."""
    import lightgbm as lgb

    X = series_df[feature_cols].fillna(0)
    y = series_df["series_length"].values

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight="balanced",
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    series_path = cfg.path("processed", "series_dataset")
    if not series_path.exists():
        raise SystemExit(f"Series dataset not found: {series_path}\nRun 'make process' first.")

    series_df = pd.read_parquet(series_path)

    # Filter to series with valid length labels
    series_df = series_df[series_df["series_length"].isin(LENGTH_CLASSES)].copy()
    feature_cols = get_feature_cols(series_df)
    logger.info(
        "Series length training: %d series, %d features", len(series_df), len(feature_cols)
    )

    # LGBM CV
    logger.info("Running LightGBM series length CV...")
    lgbm_metrics = run_lgbm_cv(series_df, feature_cols)
    logger.info(
        "LGBM length: exact_acc=%.3f±%.3f, within1=%.3f±%.3f",
        np.mean(lgbm_metrics["exact_accuracy"]), np.std(lgbm_metrics["exact_accuracy"]),
        np.mean(lgbm_metrics["within1_accuracy"]), np.std(lgbm_metrics["within1_accuracy"]),
    )

    # Ordinal CV
    logger.info("Running ordinal regression series length CV...")
    ord_metrics = run_ordinal_cv(series_df, feature_cols)
    if ord_metrics:
        logger.info(
            "Ordinal length: exact_acc=%.3f±%.3f, within1=%.3f±%.3f",
            np.mean(ord_metrics["exact_accuracy"]), np.std(ord_metrics["exact_accuracy"]),
            np.mean(ord_metrics["within1_accuracy"]), np.std(ord_metrics["within1_accuracy"]),
        )

    # Train and save final model
    final_model = train_final_length_model(series_df, feature_cols)
    out_dir = cfg.project_root / cfg.paths["models"]["trained"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "lgbm_length.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info("Length model saved: %s", model_path)

    # Log to MLflow
    setup_mlflow(EXPERIMENT_NAME)
    params = {
        "model": "LGBMClassifier",
        "target": "series_length",
        "classes": "4,5,6,7",
        "class_weight": "balanced",
        "n_training_series": len(series_df),
    }
    log_training_run(
        model=final_model,
        params=params,
        cv_metrics=lgbm_metrics,
        feature_names=feature_cols,
        run_name="lgbm_series_length",
        register_as=cfg.mlflow["model_registry"]["length_model_name"],
    )
    logger.info("Series length training complete.")


if __name__ == "__main__":
    main()
