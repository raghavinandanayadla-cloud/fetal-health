"""Lab 6: rerun training and verify reproducibility and model lifecycle."""

import math

import joblib

from src.evaluate import calculate_metrics
from src.mlflow_lifecycle import configure_tracking, get_latest_best_training_run
from src.preprocess import PROJECT_ROOT, RANDOM_SEED, TARGET_CLASSES
from src.train import MODEL_PATH, prepare_data
from src.train_mlflow import run_mlflow

REPORT_PATH = PROJECT_ROOT / "reports" / "lab6_reproducibility_report.txt"
METRIC_NAMES = ("accuracy", "macro_f1")


def _metrics_match(first: dict, second: dict) -> bool:
    return all(
        math.isclose(first[name], second[name], rel_tol=0, abs_tol=1e-12)
        for name in METRIC_NAMES
    )


def run() -> dict:
    """Compare the existing run with a seeded rerun and test the saved model."""
    print("Lab 6: capturing the latest completed baseline run...")
    client, experiment = configure_tracking()
    previous = get_latest_best_training_run(client, experiment.experiment_id)
    previous_metrics = {
        "accuracy": previous["test_accuracy"],
        "macro_f1": previous["test_macro_f1"],
    }
    print(
        f"Previous best: {previous['model_name']} "
        f"(accuracy={previous_metrics['accuracy']:.6f}, "
        f"macro_f1={previous_metrics['macro_f1']:.6f})"
    )

    print(f"Rerunning the existing training workflow with seed {RANDOM_SEED}...")
    current_results = run_mlflow()
    current_run = get_latest_best_training_run(client, experiment.experiment_id)
    current_metrics = current_results["metrics"]
    reproducible = (
        previous["model_name"] == current_results["best_model_name"]
        and _metrics_match(previous_metrics, current_metrics)
    )

    print(f"Loading saved model from: {MODEL_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Expected saved model was not created: {MODEL_PATH}")
    saved_model = joblib.load(MODEL_PATH)
    _, test_features, _, test_target = prepare_data()
    predictions = saved_model.predict(test_features)
    saved_model_metrics = calculate_metrics(test_target, predictions)
    prediction_check = (
        len(predictions) == len(test_target)
        and set(predictions).issubset(set(TARGET_CLASSES))
        and _metrics_match(saved_model_metrics, current_metrics)
    )

    metric_lines = []
    for name in METRIC_NAMES:
        delta = current_metrics[name] - previous_metrics[name]
        metric_lines.extend(
            [
                f"Previous {name}: {previous_metrics[name]:.12f}",
                f"Rerun {name}: {current_metrics[name]:.12f}",
                f"Delta {name}: {delta:+.12f}",
            ]
        )

    overall_pass = reproducible and prediction_check
    report_lines = [
        "Lab 6: Reproducibility and Model Lifecycle",
        "===========================================",
        f"Overall status: {'PASS' if overall_pass else 'FAIL'}",
        f"Random seed: {RANDOM_SEED}",
        f"Previous selected model: {previous['model_name']}",
        f"Rerun selected model: {current_results['best_model_name']}",
        f"Previous MLflow run ID: {previous['run_id']}",
        f"Rerun MLflow run ID: {current_run['run_id']}",
        f"Selected model reproducible: {'PASS' if reproducible else 'FAIL'}",
        *metric_lines,
        f"Saved model loaded: PASS ({MODEL_PATH})",
        f"Saved model prediction count: {len(predictions)}",
        f"Saved model predictions use expected classes: "
        f"{'PASS' if set(predictions).issubset(set(TARGET_CLASSES)) else 'FAIL'}",
        f"Saved model metrics match rerun: "
        f"{'PASS' if _metrics_match(saved_model_metrics, current_metrics) else 'FAIL'}",
        f"Loaded model accuracy: {saved_model_metrics['accuracy']:.12f}",
        f"Loaded model macro F1: {saved_model_metrics['macro_f1']:.12f}",
        f"Report path: {REPORT_PATH}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Rerun metrics: {current_metrics}")
    print(f"Saved model predictions verified: {prediction_check}")
    print(f"Reproducibility report: {REPORT_PATH}")
    if not overall_pass:
        raise RuntimeError(f"Lab 6 reproducibility check failed; see {REPORT_PATH}")
    return {
        "best_model_name": current_results["best_model_name"],
        "previous_metrics": previous_metrics,
        "rerun_metrics": current_metrics,
        "saved_model_metrics": saved_model_metrics,
        "reproducible": reproducible,
        "prediction_check": prediction_check,
        "report_path": REPORT_PATH,
    }


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        raise RuntimeError(f"Lab 6 reproducibility check failed: {error}") from error
