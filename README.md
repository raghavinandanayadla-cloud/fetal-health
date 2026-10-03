# Fetal Health Prediction

A small, beginner-friendly, six-lab project using the fetal health CSV. It uses pandas, scikit-learn, matplotlib, joblib, and local MLflow tracking. No deep learning, Docker, or cloud services are used.

This is an educational machine-learning example, not a clinical decision tool.

## Project Layout

```text
Fetal_Health_Prediction/
├── data/
│   ├── raw/fetal_health.csv
│   └── processed/                 # created by Lab 2
├── notebooks/Fetal_Health_Prediction.ipynb
├── src/                           # shared data, model, and evaluation code
├── pipelines/                     # one runnable script per lab
├── models/                        # fitted model files created by Lab 3 or 4
├── outputs/                       # plots and classification report
├── reports/                       # data quality and evaluation reports
├── artifacts/                     # reserved for generated artifacts
├── mlruns/                        # local MLflow experiment data
├── mlflow.db                      # local SQLite tracking database (created by Lab 4)
├── requirements.txt
└── README.md
```

## Setup (Windows PowerShell)

Run these commands from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, allow it for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The raw input file must be at `data\raw\fetal_health.csv`.

## Run Each Lab

Run from the project folder. These commands use the project virtual environment directly, so activation is optional:

```powershell
.\.venv\Scripts\python.exe -m pipelines.run_lab1_eda
.\.venv\Scripts\python.exe -m pipelines.run_lab2_preprocessing
.\.venv\Scripts\python.exe -m pipelines.run_lab3_baseline
.\.venv\Scripts\python.exe -m pipelines.run_lab4_tracking
.\.venv\Scripts\python.exe -m pipelines.run_lab5_registry
.\.venv\Scripts\python.exe -m pipelines.run_lab6_reproducibility
```

Lab 1 prints basic summaries and saves the class-distribution and correlation plots. Lab 2 validates the schema, removes exact duplicate records, and writes the cleaned CSV, metadata, and data-quality report. Labs 3 and 4 can also run on their own; both load and clean the raw CSV through the shared code.

Lab 3 selects a baseline by 5-fold stratified cross-validation on the training partition. The test partition is stratified, held out during selection, and used once for final evaluation. Imputation and scaling are part of each model pipeline, so they are fitted within each training fold. The random seed is fixed at 42. Class weighting helps account for the unequal class counts.

Lab 3 and Lab 4 create:

- `models\best_fetal_model.pkl`: fitted preprocessing and classifier pipeline
- `models\preprocessor.pkl`: fitted imputer and scaler
- `outputs\accuracy_comparison.png`: cross-validation macro-F1 comparison
- `outputs\confusion_matrix.png` and `outputs\classification_report.txt`
- `reports\evaluation_report.txt`

Lab 4 logs parameters and metrics to local MLflow. To view the runs, start the UI from the project folder:

```powershell
mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

Open the local URL printed by MLflow (usually `http://127.0.0.1:5000`). The UI is local; no cloud account is needed.

Lab 5 registers the existing `models\best_fetal_model.pkl` pipeline in the local MLflow Model Registry as `FetalHealthPredictionBestModel`. It records a model version, training parameters and metrics, and relevant evaluation artifacts. Run Lab 4 first if the saved model or baseline runs do not exist. Its report is written to `reports\lab5_registry_report.txt`.

Lab 6 snapshots the latest completed baseline metrics, reruns the Lab 4 tracked workflow with the fixed seed, and compares the selected model and test metrics. It then reloads the saved pipeline and verifies predictions on the same reproducible test split. Its report is written to `reports\lab6_reproducibility_report.txt`. Labs 5 and 6 add new MLflow records and do not delete earlier runs.

## Notebook

Open `notebooks\Fetal_Health_Prediction.ipynb` in VS Code or Jupyter and run cells from top to bottom. It calls the same modules as the lab scripts to avoid duplicating logic.

## Dataset Target

The target column is `fetal_health`: 1 = Normal, 2 = Suspect, and 3 = Pathological. The pipeline checks the expected feature names and target values before training.
