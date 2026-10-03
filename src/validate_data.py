"""Basic schema and data-quality checks for the raw dataset."""

from pathlib import Path

import pandas as pd

from src.preprocess import (
    EXPECTED_FEATURES,
    PROJECT_ROOT,
    TARGET_CLASSES,
    TARGET_COLUMN,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_report.txt"
REQUIRED_COLUMNS = set(EXPECTED_FEATURES) | {TARGET_COLUMN}


def validate_dataframe(data: pd.DataFrame) -> dict:
    """Return checks and counts without modifying the input data."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    non_numeric_columns = [
        column
        for column in data.columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if TARGET_COLUMN in data.columns:
        invalid_target_values = sorted(
            set(data[TARGET_COLUMN].dropna()) - set(TARGET_CLASSES)
        )
    else:
        invalid_target_values = ["target column is missing"]

    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_columns": missing_columns,
        "non_numeric_columns": non_numeric_columns,
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "invalid_target_values": invalid_target_values,
    }


def write_quality_report(report: dict, path: Path = REPORT_PATH) -> None:
    """Write a readable summary of the validation checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    issues_found = bool(
        report["missing_columns"]
        or report["non_numeric_columns"]
        or report["missing_cells"]
        or report["invalid_target_values"]
    )
    lines = [
        "Fetal health data quality report",
        "=================================",
        f"Status: {'ISSUES FOUND' if issues_found else 'PASS'}",
        f"Rows: {report['rows']}",
        f"Columns: {report['columns']}",
        f"Missing cells: {report['missing_cells']}",
        f"Exact duplicate rows: {report['duplicate_rows']}",
        f"Missing required columns: {report['missing_columns'] or 'None'}",
        f"Non-numeric columns: {report['non_numeric_columns'] or 'None'}",
        f"Invalid target values: {report['invalid_target_values'] or 'None'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_or_raise(data: pd.DataFrame) -> dict:
    """Validate the dataset and stop the pipeline when required checks fail."""
    report = validate_dataframe(data)
    problems = []
    if report["missing_columns"]:
        problems.append(f"Missing columns: {report['missing_columns']}")
    if report["non_numeric_columns"]:
        problems.append(f"Non-numeric columns: {report['non_numeric_columns']}")
    if report["invalid_target_values"]:
        problems.append(f"Invalid target values: {report['invalid_target_values']}")
    if problems:
        raise ValueError("Dataset validation failed. " + " ".join(problems))
    return report
