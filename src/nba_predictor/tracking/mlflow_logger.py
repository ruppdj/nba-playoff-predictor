"""MLflow experiment tracking helpers.

Thin wrapper around mlflow that:
  - Sets consistent tagging conventions (git_commit, model_type, target, data_range)
  - Logs per-fold CV metrics as both individual steps and mean/std
  - Logs artifacts (SHAP plots, calibration curves, feature importance)
  - Handles model registration to the MLflow Model Registry
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np

from nba_predictor.config import cfg, get_git_hash

logger = logging.getLogger(__name__)

TRACKING_URI = str(cfg.project_root / cfg.mlflow["tracking_uri"])


def setup_mlflow(experiment_name: str) -> None:
    """Configure MLflow tracking URI and set the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    logger.debug("MLflow experiment: '%s' at %s", experiment_name, TRACKING_URI)


def log_training_run(
    model: Any,
    params: dict[str, Any],
    cv_metrics: dict[str, list[float]],
    feature_names: list[str],
    run_name: str,
    artifact_paths: list[Path] | None = None,
    register_as: str | None = None,
) -> str:
    """Log a complete training run to MLflow.

    Args:
        model: Fitted sklearn-compatible model.
        params: Hyperparameter dict to log.
        cv_metrics: Dict of metric_name → [fold_value, ...] lists.
        feature_names: List of feature names used.
        run_name: Display name for this MLflow run.
        artifact_paths: Optional list of plot/file paths to log as artifacts.
        register_as: If set, register the model under this name in the Registry.

    Returns:
        MLflow run ID.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Tags
        mlflow.set_tag("git_commit", get_git_hash())
        mlflow.set_tag("model_type", model.__class__.__name__)
        mlflow.set_tag("target", params.get("target", "series_winner"))
        mlflow.set_tag(
            "data_range",
            f"{cfg.seasons['start']}-{cfg.seasons['end']}"
        )

        # Parameters
        mlflow.log_params({k: v for k, v in params.items() if not isinstance(v, (list, dict))})
        mlflow.log_param("n_features", len(feature_names))

        # CV metrics — mean/std and per-fold steps
        for metric_name, values in cv_metrics.items():
            arr = np.array([v for v in values if v is not None and not np.isnan(v)])
            if len(arr) == 0:
                continue
            mlflow.log_metric(f"{metric_name}_mean", float(arr.mean()))
            mlflow.log_metric(f"{metric_name}_std", float(arr.std()))
            for fold_i, val in enumerate(values):
                if val is not None and not np.isnan(float(val)):
                    mlflow.log_metric(f"{metric_name}_fold", float(val), step=fold_i)

        # Feature names as JSON artifact
        mlflow.log_dict({"features": feature_names}, "feature_names.json")

        # Model artifact
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as exc:
            logger.warning("Failed to log model as sklearn artifact: %s", exc)
            # Try generic pickle fallback
            import pickle
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                pickle.dump(model, tmp)
                mlflow.log_artifact(tmp.name, "model")

        # Additional artifacts (plots, etc.)
        if artifact_paths:
            for path in artifact_paths:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
                else:
                    logger.warning("Artifact not found, skipping: %s", path)

        # Register in Model Registry if requested
        if register_as:
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mlflow.register_model(model_uri, register_as)
                logger.info("Model registered as '%s'", register_as)
            except Exception as exc:
                logger.warning("Model registration failed: %s", exc)

        run_id = run.info.run_id
        logger.info(
            "MLflow run logged: %s (run_id=%s)", run_name, run_id
        )
        return run_id


def load_registered_model(model_name: str, stage: str = "Production") -> Any:
    """Load a model from the MLflow Model Registry.

    Args:
        model_name: Registered model name (e.g. 'series_winner_champion').
        stage: Model stage ('Production', 'Staging', 'None').

    Returns:
        Loaded sklearn model.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info("Loading model from registry: %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def get_best_run(experiment_name: str, metric: str = "accuracy_mean",
                 ascending: bool = False) -> dict[str, Any]:
    """Return the parameters and metrics of the best run in an experiment.

    Args:
        experiment_name: MLflow experiment name.
        metric: Metric to rank by.
        ascending: If True, lower is better (e.g. log_loss). Default False.

    Returns:
        Dictionary with run_id, params, and metrics.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")
    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
    }
