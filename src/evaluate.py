"""Metrics and output files for held-out model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from src.preprocess import PROJECT_ROOT, TARGET_CLASSES

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLASS_NAMES = ("Normal (1)", "Suspect (2)", "Pathological (3)")


def calculate_metrics(actual, predicted) -> dict:
    """Calculate simple summary metrics for the test partition."""
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
    }


def save_evaluation_outputs(actual, predicted, metrics: dict, cv_scores: dict) -> None:
    """Save the confusion matrix and text evaluation reports."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=list(TARGET_CLASSES),
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Held-out test set")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    report_text = classification_report(
        actual,
        predicted,
        labels=list(TARGET_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    (OUTPUTS_DIR / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    lines = [
        "Fetal health model evaluation",
        "=============================",
        "The test set was held out from cross-validation and model selection.",
        f"Test accuracy: {metrics['accuracy']:.4f}",
        f"Test macro F1: {metrics['macro_f1']:.4f}",
        "Training-only cross-validation macro F1:",
    ]
    lines.extend(f"  {name}: {score:.4f}" for name, score in cv_scores.items())
    lines.extend(["", report_text])
    (REPORTS_DIR / "evaluation_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def save_comparison_plot(cv_scores: dict) -> None:
    """Compare models using training-only cross-validation macro F1."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    names = list(cv_scores)
    scores = [cv_scores[name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, scores, color="#327a76")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean cross-validation macro F1")
    axis.set_title("Baseline model comparison (training data only)")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(figure)
