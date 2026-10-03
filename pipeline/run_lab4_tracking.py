"""Lab 4: run the shared baselines with local MLflow tracking."""

from src.train_mlflow import run_mlflow


def run() -> dict:
    return run_mlflow()


if __name__ == "__main__":
    run()
