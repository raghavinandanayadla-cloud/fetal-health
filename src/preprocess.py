"""Load, validate, and prepare fetal health data for modeling."""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fetal_health.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fetal_health_cleaned.csv"
TARGET_COLUMN = "fetal_health"
TARGET_CLASSES = (1, 2, 3)
RANDOM_SEED = 42
TEST_SIZE = 0.2
EXPECTED_FEATURES = (
    "baseline value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the original CSV file."""
    return pd.read_csv(path)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numbers and remove exact duplicate records."""
    cleaned = data.copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    return cleaned.drop_duplicates().reset_index(drop=True)


def get_features_and_target(data: pd.DataFrame):
    """Separate predictors from the label column."""
    features = data.loc[:, list(EXPECTED_FEATURES)]
    target = data[TARGET_COLUMN]
    return features, target


def create_preprocessor() -> Pipeline:
    """Create preprocessing that is fitted only on each training fold."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
