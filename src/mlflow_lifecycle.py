"""Shared helpers for Labs 5 and 6 using the existing local MLflow store."""

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.preprocess import RANDOM_SEED
from src.train import get_candidate_models
from src.train_mlflow import EXPERIMENT_NAME, MLRUNS_DIR, TRACKING_DB_PATH

TRACKING_URI = f"sqlite:///{TRACKING_DB_PATH.as_posix()}"


def configure_tracking() -> tuple[MlflowClient, object]:
    """Connect to the existing SQLite experiment and local registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found. Run Lab 4 first."
        )
    return client, experiment


def get_latest_best_training_run(client: MlflowClient, experiment_id: str) -> dict:
    """Find the best latest completed run for each existing baseline model."""
    runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")
    run_name_column = "tags.mlflow.runName"
    score_column = "metrics.cv_macro_f1"
    required_columns = {
        "run_id",
        "status",
        "start_time",
        run_name_column,
        score_column,
        "metrics.test_accuracy",
        "metrics.test_macro_f1",
    }
    missing_columns = required_columns - set(runs.columns)
    if missing_columns:
        raise RuntimeError(
            "Existing MLflow runs are missing required training metrics: "
            f"{sorted(missing_columns)}"
        )

    model_names = set(get_candidate_models())
    completed = runs.loc[
        (runs["status"] == "FINISHED")
        & runs[run_name_column].isin(model_names)
    ].dropna(subset=[score_column])
    if completed.empty:
        raise RuntimeError(
            "No completed Lab 3/4 baseline runs with CV metrics were found. "
            "Run Lab 4 first."
        )

    latest_per_model = (
        completed.sort_values("start_time")
        .drop_duplicates(subset=[run_name_column], keep="last")
    )
    best = latest_per_model.loc[latest_per_model[score_column].idxmax()]
    return {
        "run_id": str(best["run_id"]),
        "model_name": str(best[run_name_column]),
        "cv_macro_f1": float(best[score_column]),
        "test_accuracy": float(best["metrics.test_accuracy"]),
        "test_macro_f1": float(best["metrics.test_macro_f1"]),
        "random_seed": RANDOM_SEED,
        "mlruns_dir": MLRUNS_DIR,
    }
