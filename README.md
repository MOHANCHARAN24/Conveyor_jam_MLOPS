# Universal MLOps Pipeline (Colab + GitHub)

A reusable exam-oriented MLOps starter for **tabular classification and regression**. It includes synthetic demo data, dataset snapshots, generic preprocessing, feature engineering/storage, two-model comparison, MLflow tracking/registry, FastAPI serving, pytest tests and GitHub Actions CI.

> Important: this is a reusable template, not a magical problem-statement interpreter. On exam day, identify the target column, choose classification/regression, set the CSV path, and add domain-specific feature engineering if the scenario requires it.

## 1. Project contents

- `src/data_generator.py`: creates synthetic classification and regression CSVs.
- `src/data_versioning.py`: snapshots datasets with SHA-256 metadata.
- `src/preprocessing.py`: median/mode imputation, one-hot encoding and numeric scaling.
- `src/feature_engineering.py`: example generic row aggregates; replace/extend for your domain.
- `src/train.py`: trains two candidate models, evaluates, logs to MLflow and registers the selected model.
- `features/`: transformed train/test feature files (feature store demonstration).
- `api/main.py`: loads the saved pipeline and exposes `/predict`.
- `tests/`: basic pipeline and API tests.
- `.github/workflows/ci.yml`: runs tests on push and pull request.
- `colab/Colab_Quickstart.ipynb`: notebook guide for Colab.

## 2. Run in Google Colab

1. Upload this ZIP to Google Drive and extract it, or push the folder to GitHub and clone it in Colab.
2. In Colab, change directory into the repository.
3. Install dependencies: `!pip install -r requirements.txt`
4. Generate both demo datasets: `!python -m src.data_generator`
5. Snapshot the datasets: `!python -m src.data_versioning`
6. Configure `config/config.json` (see below).
7. Train: `!python -m src.train --config config/config.json`
8. Test: `!pytest -q`
9. Start API: `!uvicorn api.main:app --host 0.0.0.0 --port 8000`
10. For a local test, POST to `http://127.0.0.1:8000/predict`.

Colab runtimes are temporary. Download/copy generated artifacts and datasets to Drive or commit the required files to GitHub before the runtime disconnects.

## 3. Configure tomorrow's problem statement

Edit `config/config.json`:

```json
{
  "dataset_path": "data/raw/your_dataset.csv",
  "target": "your_target_column",
  "task": "classification",
  "test_size": 0.2,
  "random_state": 42,
  "primary_metric": "f1"
}
```

For regression, set `"task": "regression"` and `"primary_metric": "rmse"` (or `mae` / `r2`).

- Classification examples: pass/fail, churn, fraud flag, disease class.
- Regression examples: price, sales amount, marks, demand quantity.

Do not infer the task from the target dtype alone: integer targets can represent either class labels or a numeric quantity. Use the problem statement.

## 4. Dataset version management

The versioning script creates folders such as `data/versions/v001/` containing a CSV copy and `metadata.json` with a SHA-256 hash, timestamp and notes. Run it again after changing the input data to create the next version. The demo generator makes two separate sample datasets.

## 5. Feature store demonstration

After training, `features/train_features.csv` and `features/test_features.csv` contain the transformed features used by the models. The fitted preprocessing pipeline is stored with the model, so the same transformations are applied at inference time. These CSVs are a lightweight file-based feature store demonstration, not an online enterprise feature store.

## 6. MLflow

The training script uses a local SQLite tracking backend (`mlflow.db`) and logs each candidate run's parameters, metrics and model. It then registers the best run under `UniversalMLOpsModel` (or `registered_model_name` in config). Run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Open the forwarded/tunneled port in Colab or run locally. Registry availability can depend on the installed MLflow version and backend configuration. Keep `mlflow.db` and `mlruns/` as local runtime artifacts; do not commit large temporary tracking directories unless requested.

## 7. FastAPI

Start after training:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Request format:

```json
{
  "features": {
    "numeric_1": 0.2,
    "numeric_2": 0.5,
    "category": "A"
  }
}
```

The keys must match the original input feature columns (exclude the target). Response includes prediction, task and model name. In a new scenario, use the actual feature names and valid example values.

## 8. GitHub and CI

Create a GitHub repository, then from the project directory:

```bash
git init
git add .
git commit -m "Add reusable MLOps pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Universal-MLOps-Pipeline.git
git push -u origin main
```

GitHub Actions runs `pytest` automatically on pushes and pull requests. Check the **Actions** tab and capture a successful run screenshot for your report.

Never commit passwords, access tokens, `.env` files, or private data.

## 9. Exam-day checklist

- [ ] Copy the allotted CSV into `data/raw/`.
- [ ] Set `dataset_path`, `target`, `task` and metric in `config/config.json`.
- [ ] Check column names and remove identifiers/leakage columns if appropriate.
- [ ] Add meaningful domain-specific features if required.
- [ ] Run dataset snapshot and training.
- [ ] Inspect `reports/model_comparison.csv`.
- [ ] Show MLflow experiment runs and registered model.
- [ ] Run `pytest -q` and show GitHub Actions.
- [ ] Start FastAPI and demonstrate `/predict`.
- [ ] Capture screenshots and complete the PDF report.

## 10. Limitations and good practice

This template targets **tabular supervised learning**. It is not directly suitable without adaptation for images, audio, text/NLP, time-series forecasting, clustering, recommendation, or unsupervised anomaly detection. Those scenarios need task-specific data loading, feature extraction, models and metrics. For classification, the starter assumes a single-label target. For highly imbalanced data, consider stratified splitting, class-specific metrics and threshold analysis. Review feature leakage, duplicate rows, ID columns and target definition before training. Compare models on held-out data and do not report illustrative demo metrics as real results.
