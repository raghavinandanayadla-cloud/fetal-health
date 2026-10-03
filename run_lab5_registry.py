"""Lab 5: register the existing best model in the local MLflow registry."""

import math
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from src.evaluate import calculate_metrics
from src.mlflow_lifecycle import configure_tracking, get_latest_best_training_run
from src.preprocess import PROJECT_ROOT, RANDOM_SEED
from src.train import MODEL_PATH, prepare_data

REGISTRY_MODEL_NAME = "FetalHealthPredictionBestModel"
REPORT_PATH = PROJECT_ROOT / "reports" / "lab5_registry_report.txt"
EVALUATION_REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation_report.txt"
OUTPUT_FILES = (
    PROJECT_ROOT / "outputs" / "classification_report.txt",
    PROJECT_ROOT / "outputs" / "confusion_matrix.png",
    PROJECT_ROOT / "outputs" / "accuracy_comparison.png",
)


def _algorithm_name(model) -> str:
    classifier = model.named_steps.get("classifier")
    names = {
        "LogisticRegression": "Logistic Regression",
        "DecisionTreeClassifier": "Decision Tree",
        "RandomForestClassifier": "Random Forest",
    }
    algorithm = names.get(type(classifier).__name__)
    if algorithm is None:
        raise TypeError(
            f"Unsupported saved classifier: {type(classifier).__name__}"
        )
    return algorithm


def run() -> dict:
    """Register the persisted pipeline and write a local registry report."""
    print("Lab 5: checking the local MLflow Model Registry...")
    client, experiment = configure_tracking()
    if not callable(getattr(client, "search_registered_models", None)):
        raise RuntimeError(
            "This installed MLflow client does not support the Model Registry."
        )
    client.search_registered_models(max_results=1)

    best_run = get_latest_best_training_run(client, experiment.experiment_id)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_PATH}. Run Lab 3 or Lab 4 first."
        )
    if not EVALUATION_REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation report not found at {EVALUATION_REPORT_PATH}. "
            "Run Lab 3 or Lab 4 first."
        )

    model = joblib.load(MODEL_PATH)
    algorithm_name = _algorithm_name(model)
    if algorithm_name != best_run["model_name"]:
        raise RuntimeError(
            "Saved model does not match the best completed MLflow run: "
            f"saved={algorithm_name}, tracked={best_run['model_name']}"
        )

    train_features, test_features, train_target, test_target = prepare_data()
    actual_metrics = calculate_metrics(test_target, model.predict(test_features))
    for metric_name in ("accuracy", "macro_f1"):
        tracked_value = best_run[f"test_{metric_name}"]
        if not math.isclose(
            actual_metrics[metric_name], tracked_value, rel_tol=0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"Saved model {metric_name} does not match the selected MLflow run."
            )

    classifier = model.named_steps["classifier"]
    model_parameters = {
        f"classifier.{name}": str(value)
        for name, value in classifier.get_params(deep=False).items()
    }
    sample = train_features.head(5)
    signature = mlflow.models.infer_signature(sample, model.predict(sample))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Selected existing model: {algorithm_name}")
    print(f"Registering as: {REGISTRY_MODEL_NAME}")
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="lab5-model-registry",
    ) as active_run:
        mlflow.log_param("registered_model_name", REGISTRY_MODEL_NAME)
        mlflow.log_param("selected_model", algorithm_name)
        mlflow.log_param("source_training_run_id", best_run["run_id"])
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_params(model_parameters)
        mlflow.log_metrics(
            {
                "cv_macro_f1": best_run["cv_macro_f1"],
                "test_accuracy": actual_metrics["accuracy"],
                "test_macro_f1": actual_metrics["macro_f1"],
            }
        )
        mlflow.set_tag("lab", "5-model-registry")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="source_model")
        mlflow.log_artifact(str(EVALUATION_REPORT_PATH), artifact_path="reports")
        for artifact_path in OUTPUT_FILES:
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Expected evaluation artifact not found: {artifact_path}")
            mlflow.log_artifact(str(artifact_path), artifact_path="outputs")

        mlflow_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="registered_model",
            registered_model_name=REGISTRY_MODEL_NAME,
            signature=signature,
            input_example=sample,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        model_versions = [
            version
            for version in client.search_model_versions(
                f"name='{REGISTRY_MODEL_NAME}'"
            )
            if version.run_id == active_run.info.run_id
        ]
        if not model_versions:
            raise RuntimeError(
                "MLflow logged the model but returned no registered model version."
            )
        registered_version = max(model_versions, key=lambda item: int(item.version))
        if registered_version.status != "READY":
            raise RuntimeError(
                "Registered model version is not ready: "
                f"{registered_version.status}"
            )

        report_lines = [
            "Lab 5: Model Registry",
            "=====================",
            "Status: READY",
            f"Registry model name: {REGISTRY_MODEL_NAME}",
            f"Registry model version: {registered_version.version}",
            f"Selected algorithm: {algorithm_name}",
            f"Source training run ID: {best_run['run_id']}",
            f"Registry run ID: {active_run.info.run_id}",
            f"MLflow model URI: {mlflow_model.model_uri}",
            f"Training CV macro F1: {best_run['cv_macro_f1']:.6f}",
            f"Test accuracy: {actual_metrics['accuracy']:.6f}",
            f"Test macro F1: {actual_metrics['macro_f1']:.6f}",
            f"Local source model: {MODEL_PATH}",
            "Classifier parameters:",
        ]
        report_lines.extend(
            f"  {name.removeprefix('classifier.')}: {value}"
            for name, value in model_parameters.items()
        )
        report_lines.extend(
            [
                "Logged artifacts:",
                "  source_model/best_fetal_model.pkl",
                "  reports/evaluation_report.txt",
                "  outputs/classification_report.txt",
                "  outputs/confusion_matrix.png",
                "  outputs/accuracy_comparison.png",
            ]
        )
        REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="reports")

    client.set_model_version_tag(
        REGISTRY_MODEL_NAME,
        registered_version.version,
        "selected_algorithm",
        algorithm_name,
    )
    print(f"Registered version: {registered_version.version}")
    print(f"Test metrics: {actual_metrics}")
    print(f"Registry report: {REPORT_PATH}")
    return {
        "model_name": REGISTRY_MODEL_NAME,
        "version": registered_version.version,
        "algorithm": algorithm_name,
        "metrics": actual_metrics,
        "report_path": REPORT_PATH,
    }


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        raise RuntimeError(f"Lab 5 registry failed: {error}") from error
