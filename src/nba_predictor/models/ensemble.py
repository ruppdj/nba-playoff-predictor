"""Stacking ensemble — meta-learner on XGBoost + LightGBM + LogReg predictions.

Uses out-of-fold (OOF) predictions from base models as features for a
Logistic Regression meta-learner. This is the primary champion model
and is registered in the MLflow Model Registry.

Run via: python -m nba_predictor.models.ensemble
      or: make train
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nba_predictor.config import cfg
from nba_predictor.evaluation.cv_strategy import playoff_season_cv_splits
from nba_predictor.evaluation.metrics import compute_winner_metrics
from nba_predictor.tracking.mlflow_logger import log_training_run, setup_mlflow

logger = logging.getLogger(__name__)

RANDOM_STATE = cfg.modeling["random_state"]
EXPERIMENT_NAME = "series_winner_ensemble"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    all_feat = (
        cfg.features.get("matchup", [])
        + cfg.features.get("meta", [])
        + [f"higher_{c}" for c in cfg.features.get("injury", [])]
        + [f"lower_{c}" for c in cfg.features.get("injury", [])]
    )
    return [c for c in all_feat if c in df.columns]


def _load_base_model(name: str) -> object | None:
    """Load a pickled base model from models/trained/."""
    path = cfg.project_root / cfg.paths["models"]["trained"] / f"{name}_winner.pkl"
    if not path.exists():
        logger.warning("Base model not found: %s", path)
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_oof_predictions(
    series_df: pd.DataFrame,
    feature_cols: list[str],
    min_train_seasons: int = 5,
) -> np.ndarray:
    """Generate out-of-fold predictions from all base models.

    Returns an (n_samples, 3) array of [lr_prob, xgb_prob, lgbm_prob]
    using walk-forward CV to avoid data leakage.

    min_train_seasons is kept small (default 5) so this works when called
    from inside an outer CV fold that already has a reduced season window.
    """
    n = len(series_df)
    oof_preds = np.full((n, 3), np.nan)

    for train_idx, test_idx in playoff_season_cv_splits(
        series_df, min_train_seasons=min_train_seasons
    ):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        X_train = train[feature_cols].fillna(0)
        y_train = train["higher_seed_wins"].values
        X_test = test[feature_cols].fillna(0)

        # Map test_idx to positional indices in oof_preds
        pos_idx = [series_df.index.get_loc(i) for i in test_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # 1. Logistic Regression — stronger regularization to prevent overfitting
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=0.05)
        lr.fit(X_train_sc, y_train)
        oof_preds[pos_idx, 0] = lr.predict_proba(X_test_sc)[:, 1]

        # 2. XGBoost — shallower trees + higher min_child_weight to reduce overfitting
        try:
            import xgboost as xgb

            xgb_model = xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=50,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
            )
            xgb_model.fit(X_train, y_train, verbose=False)
            oof_preds[pos_idx, 1] = xgb_model.predict_proba(X_test)[:, 1]
        except ImportError:
            logger.warning("XGBoost not available — using LR predictions for slot 1")
            oof_preds[pos_idx, 1] = oof_preds[pos_idx, 0]

        # 3. LightGBM — fewer leaves + min_data_in_leaf to reduce overfitting
        try:
            import lightgbm as lgb

            lgbm_model = lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                n_estimators=50,
                num_leaves=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_data_in_leaf=15,
                verbose=-1,
            )
            lgbm_model.fit(X_train, y_train)
            oof_preds[pos_idx, 2] = lgbm_model.predict_proba(X_test)[:, 1]
        except ImportError:
            logger.warning("LightGBM not available — using XGB predictions for slot 2")
            oof_preds[pos_idx, 2] = oof_preds[pos_idx, 1]

    logger.info("OOF predictions generated: shape %s", oof_preds.shape)
    return oof_preds


class StackingEnsemble:
    """Stacking ensemble: base model OOF → LogReg meta-learner.

    This wraps the full ensemble so it can be serialized and loaded
    as a single object for prediction.
    """

    def __init__(self) -> None:
        self.meta_learner = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=0.5)
        self.scaler_base = StandardScaler()  # for base models
        self.meta_calibrated: CalibratedClassifierCV | None = None
        self.feature_cols: list[str] = []
        self._base_models: dict = {}

    def fit(self, series_df: pd.DataFrame) -> StackingEnsemble:
        self.feature_cols = get_feature_cols(series_df)
        X = series_df[self.feature_cols].fillna(0)
        y = series_df["higher_seed_wins"].values

        # Fit base models on full data
        self.scaler_base.fit(X)
        X_sc = self.scaler_base.transform(X)

        # Stronger regularization on LR to match OOF training configuration
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=0.05)
        lr.fit(X_sc, y)
        self._base_models["lr"] = ("scaled", lr)

        try:
            import xgboost as xgb

            xgb_model = xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=50,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
            )
            xgb_model.fit(X, y, verbose=False)
            self._base_models["xgb"] = ("raw", xgb_model)
        except ImportError:
            pass

        try:
            import lightgbm as lgb

            lgbm_model = lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                n_estimators=50,
                num_leaves=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_data_in_leaf=15,
                verbose=-1,
            )
            lgbm_model.fit(X, y)
            self._base_models["lgbm"] = ("raw", lgbm_model)
        except ImportError:
            pass

        # Generate OOF predictions for meta-training
        oof_preds = generate_oof_predictions(series_df, self.feature_cols)
        oof_preds = np.nan_to_num(oof_preds, nan=0.5)

        # Sigmoid calibration (Platt scaling) — extrapolates better than isotonic
        # for inputs that fall outside the OOF probability range
        self.meta_calibrated = CalibratedClassifierCV(self.meta_learner, cv=3, method="sigmoid")
        self.meta_calibrated.fit(oof_preds, y)
        logger.info("Stacking ensemble fitted.")
        return self

    def _get_base_probs(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Get base model probabilities for new data."""
        X_raw_vals = X_raw.values
        X_sc = self.scaler_base.transform(X_raw_vals)
        probs = np.zeros((len(X_raw), 3))

        model_keys = list(self._base_models.keys())
        for i, key in enumerate(model_keys[:3]):
            mode, model = self._base_models[key]
            inp = X_sc if mode == "scaled" else X_raw_vals
            probs[:, i] = model.predict_proba(inp)[:, 1]

        return probs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Always restrict to the feature set used at training time
        X_aligned = X[self.feature_cols].fillna(0)
        base_probs = self._get_base_probs(X_aligned)
        return self.meta_calibrated.predict_proba(base_probs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run_ensemble_cv(series_df: pd.DataFrame) -> dict[str, list[float]]:
    """Evaluate the stacking ensemble with walk-forward CV."""
    feature_cols = get_feature_cols(series_df)
    metrics_history: dict[str, list[float]] = {
        "accuracy": [],
        "log_loss": [],
        "brier_score": [],
        "upset_recall": [],
        "ece": [],
    }

    for fold_i, (train_idx, test_idx) in enumerate(playoff_season_cv_splits(series_df)):
        train = series_df.loc[train_idx]
        test = series_df.loc[test_idx]

        ensemble = StackingEnsemble()
        ensemble.fit(train)

        X_test = test[feature_cols].fillna(0)
        y_test = test["higher_seed_wins"].values
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:, 1]

        fold_metrics = compute_winner_metrics(y_test, y_pred, y_prob)
        for k in metrics_history:
            if k in fold_metrics:
                metrics_history[k].append(fold_metrics[k])

        logger.info(
            "Ensemble fold %d: acc=%.3f, logloss=%.3f",
            fold_i,
            fold_metrics["accuracy"],
            fold_metrics["log_loss"],
        )

    return metrics_history


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    series_path = cfg.path("processed", "series_dataset")
    if not series_path.exists():
        raise SystemExit(f"Series dataset not found: {series_path}\nRun 'make process' first.")

    series_df = pd.read_parquet(series_path)
    feature_cols = get_feature_cols(series_df)
    logger.info("Training ensemble: %d series, %d features", len(series_df), len(feature_cols))

    cv_metrics = run_ensemble_cv(series_df)
    logger.info(
        "Ensemble CV: acc=%.3f±%.3f, logloss=%.3f±%.3f",
        np.mean(cv_metrics["accuracy"]),
        np.std(cv_metrics["accuracy"]),
        np.mean(cv_metrics["log_loss"]),
        np.std(cv_metrics["log_loss"]),
    )

    # Train final ensemble on all data and save
    final_ensemble = StackingEnsemble()
    final_ensemble.fit(series_df)

    out_dir = cfg.project_root / cfg.paths["models"]["trained"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ensemble_winner.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_ensemble, f)
    logger.info("Ensemble saved: %s", model_path)

    # Log to MLflow and register as champion
    setup_mlflow(EXPERIMENT_NAME)
    params = {
        "model": "StackingEnsemble",
        "base_models": "LR + XGBoost + LightGBM",
        "meta_learner": "LogisticRegression",
        "calibration": "isotonic",
        "target": "series_winner",
        "n_training_series": len(series_df),
    }
    log_training_run(
        model=final_ensemble,
        params=params,
        cv_metrics=cv_metrics,
        feature_names=feature_cols,
        run_name="stacking_ensemble",
        register_as=cfg.mlflow["model_registry"]["winner_model_name"],
    )
    logger.info("Ensemble training complete.")


if __name__ == "__main__":
    main()
