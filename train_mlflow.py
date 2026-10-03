"""Train the shared baselines and log results to local MLflow."""

import mlflow

from src.evaluate import save_comparison_plot, save_evaluation_outputs
from src.preprocess import PROJECT_ROOT, RANDOM_SEED
from src.train import (
    cross_validate_model,
    fit_and_evaluate,
    get_candidate_models,
    prepare_data,
    save_model,
)

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
TRACKING_DB_PATH = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "fetal-health-baselines"


def run_mlflow() -> dict:
    """Track model parameters and scores in a local MLflow experiment."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB_PATH.as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=MLRUNS_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_features, test_features, train_target, test_target = prepare_data()
    candidates = get_candidate_models()
    cv_scores = {}
    fitted_results = {}

    for name, classifier in candidates.items():
        cv_score = cross_validate_model(classifier, train_features, train_target)
        model, metrics, predictions = fit_and_evaluate(
            classifier, train_features, test_features, train_target, test_target
        )
        cv_scores[name] = cv_score
        fitted_results[name] = (model, metrics, predictions)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metric("cv_macro_f1", cv_score)
            mlflow.log_metric("test_accuracy", metrics["accuracy"])
            mlflow.log_metric("test_macro_f1", metrics["macro_f1"])

    best_name = max(cv_scores, key=cv_scores.get)
    model, metrics, predictions = fitted_results[best_name]
    save_model(model)
    save_comparison_plot(cv_scores)
    save_evaluation_outputs(test_target, predictions, metrics, cv_scores)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Local tracking data: {MLRUNS_DIR}")
    print(f"Selected by training-only CV: {best_name}")
    print(f"Held-out test metrics: {metrics}")
    print("View runs with: mlflow ui --backend-store-uri .\\mlruns")
    return {
        "best_model_name": best_name,
        "cv_scores": cv_scores,
        "metrics": metrics,
    }
