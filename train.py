"""Shared model selection and training code for Labs 3 and 4."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import (
    calculate_metrics,
    save_comparison_plot,
    save_evaluation_outputs,
)
from src.preprocess import (
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
    clean_data,
    create_preprocessor,
    get_features_and_target,
    load_raw_data,
)
from src.validate_data import validate_or_raise

MODEL_PATH = PROJECT_ROOT / "models" / "best_fetal_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"


def get_candidate_models() -> dict:
    """Return a few familiar, lightweight classification baselines."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def make_model_pipeline(classifier) -> Pipeline:
    """Keep imputation and scaling inside the model to avoid data leakage."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier),
        ]
    )


def prepare_data():
    """Validate, deduplicate, and make one reproducible stratified split."""
    raw_data = load_raw_data()
    validate_or_raise(raw_data)
    data = clean_data(raw_data)
    features, target = get_features_and_target(data)
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )


def cross_validate_model(classifier, features, target) -> float:
    """Score a model with stratified folds on training data only."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        make_model_pipeline(classifier),
        features,
        target,
        cv=folds,
        scoring="f1_macro",
        n_jobs=1,
    )
    return float(scores.mean())


def fit_and_evaluate(classifier, train_features, test_features, train_target, test_target):
    """Fit preprocessing and a classifier on training data, then score test data."""
    model = make_model_pipeline(classifier)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    return model, calculate_metrics(test_target, predictions), predictions


def save_model(model) -> None:
    """Save the fitted pipeline and its fitted preprocessing step."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model.named_steps["preprocessor"], PREPROCESSOR_PATH)


def run_baselines() -> dict:
    """Select with training-only CV, evaluate once on the held-out test set."""
    train_features, test_features, train_target, test_target = prepare_data()
    candidates = get_candidate_models()
    cv_scores = {
        name: cross_validate_model(classifier, train_features, train_target)
        for name, classifier in candidates.items()
    }
    best_name = max(cv_scores, key=cv_scores.get)
    model, metrics, predictions = fit_and_evaluate(
        candidates[best_name], train_features, test_features, train_target, test_target
    )

    save_model(model)
    save_comparison_plot(cv_scores)
    save_evaluation_outputs(test_target, predictions, metrics, cv_scores)
    return {
        "best_model_name": best_name,
        "cv_scores": cv_scores,
        "metrics": metrics,
        "model_path": Path(MODEL_PATH),
    }


if __name__ == "__main__":
    results = run_baselines()
    print(f"Selected model: {results['best_model_name']}")
    print(f"Training-only CV macro F1: {results['cv_scores'][results['best_model_name']]:.4f}")
    print(f"Held-out test metrics: {results['metrics']}")
    print(f"Saved model: {results['model_path']}")
