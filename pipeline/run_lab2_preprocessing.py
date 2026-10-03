"""Lab 2: validate, deduplicate, and describe the modeling dataset."""

import json

from src.preprocess import (
    EXPECTED_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
    clean_data,
    load_raw_data,
)
from src.validate_data import validate_dataframe, validate_or_raise, write_quality_report

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run() -> dict:
    raw_data = load_raw_data()
    quality_report = validate_dataframe(raw_data)
    write_quality_report(quality_report)
    validate_or_raise(raw_data)

    cleaned_data = clean_data(raw_data)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = PROCESSED_DIR / "fetal_health_cleaned.csv"
    cleaned_data.to_csv(cleaned_path, index=False)

    class_counts = {
        str(label): int(count)
        for label, count in cleaned_data[TARGET_COLUMN].value_counts().sort_index().items()
    }
    metadata = {
        "source_file": "data/raw/fetal_health.csv",
        "rows_before_cleaning": int(len(raw_data)),
        "rows_after_cleaning": int(len(cleaned_data)),
        "duplicate_rows_removed": int(len(raw_data) - len(cleaned_data)),
        "feature_count": len(EXPECTED_FEATURES),
        "target_column": TARGET_COLUMN,
        "target_class_counts": class_counts,
        "random_seed": 42,
    }
    metadata_path = PROCESSED_DIR / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Validation report: {PROJECT_ROOT / 'reports' / 'data_quality_report.txt'}")
    print(f"Cleaned CSV: {cleaned_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Rows removed as exact duplicates: {metadata['duplicate_rows_removed']}")
    return metadata


if __name__ == "__main__":
    run()
