"""Lab 1: inspect the raw dataset and save two simple plots."""

import matplotlib.pyplot as plt

from src.preprocess import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    TARGET_CLASSES,
    load_raw_data,
)
from src.validate_data import validate_dataframe, validate_or_raise

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLASS_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}


def run() -> None:
    data = load_raw_data()
    report = validate_dataframe(data)
    validate_or_raise(data)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = data[TARGET_COLUMN].value_counts().reindex(TARGET_CLASSES, fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(
        [CLASS_LABELS[label] for label in TARGET_CLASSES],
        class_counts.values,
        color=["#327a76", "#df9b35", "#bd5c55"],
    )
    axis.set_ylabel("Number of records")
    axis.set_title("Fetal health class distribution")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=150)
    plt.close(figure)

    correlation = data.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
    axis.set_yticklabels(correlation.columns, fontsize=7)
    axis.set_title("Feature correlation overview")
    figure.colorbar(image, ax=axis, label="Correlation")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(figure)

    print(f"Rows: {report['rows']}; columns: {report['columns']}")
    print("Target counts:")
    print(class_counts.rename(index=CLASS_LABELS).to_string())
    print("\nNumeric summary:")
    print(data.describe().round(2).to_string())
    print(f"\nPlots saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    run()
