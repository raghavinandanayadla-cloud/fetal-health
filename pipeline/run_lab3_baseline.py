"""Lab 3: compare baselines and evaluate the selected model."""

from src.train import run_baselines


def run() -> dict:
    results = run_baselines()
    print("Training-only cross-validation macro F1:")
    for name, score in results["cv_scores"].items():
        print(f"  {name}: {score:.4f}")
    print(f"\nSelected by cross-validation: {results['best_model_name']}")
    print(f"Held-out test metrics: {results['metrics']}")
    print(f"Saved model pipeline: {results['model_path']}")
    print("See reports/evaluation_report.txt and outputs/ for detailed results.")
    return results


if __name__ == "__main__":
    run()
