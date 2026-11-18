# Conveyor Belt Jam Prediction MLOps

This project predicts whether an automated parcel-sorting conveyor is likely to
jam during its **next operating shift**. The included data is synthetic and
intended for demonstration, not production maintenance decisions.

Repository: [MOHANCHARAN24/Conveyor_jam_MLOPS](https://github.com/MOHANCHARAN24/Conveyor_jam_MLOPS)

## What is included

- Two versioned datasets, each with 500 synthetic conveyor shifts:
  - `data/versions/conveyor/v001/`: baseline operating period.
  - `data/versions/conveyor/v002/`: festival period with 30% more parcels per
    minute and elevated motor, vibration, dust, and humidity stress.
- SHA-256 checksums, row counts, and notes in each version's `metadata.json`.
- A validation and feature pipeline that creates `motor_load_ratio` and
  `belt_wear_pressure_index`, and persists reusable rows to
  `features/conveyor_feature_store.csv`.
- Logistic Regression and Random Forest evaluation, positive-class metrics,
  confusion matrices, and MLflow tracking/model registration.
- FastAPI `/health` and `/predict` endpoints. Predictions are returned as
  `Jam Risk` or `Normal Operation`.
- GitHub Actions tests for sensor validation and API predictions.
- An implementation report at `reports/conveyor_jam_mlop_report.pdf`.

## Reproduce the pipeline

Use Python 3.11+:

```powershell
python -m pip install -r requirements.txt
python -m src.data_versioning
python -m src.train --config config/config.json
pytest -q
python scripts/build_report.py
```

`src.data_versioning` generates and snapshots both versions. The committed
versioned inputs make training reproducible; generated live model files and
MLflow's local database remain local runtime artifacts.

## API

Start the API after training:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`GET /health` reports whether a model is loaded. `POST /predict` accepts:

```json
{
  "features": {
    "motor_current_a": 18.0,
    "rated_motor_current_a": 20.0,
    "vibration_mm_s": 7.5,
    "belt_age_years": 8.0,
    "parcels_per_minute": 110.0,
    "dust_accumulation_index": 72.0,
    "days_since_maintenance": 95.0,
    "humidity_pct": 68.0
  }
}
```

The response contains the decision, estimated jam probability, and selected
model name. Required values and sensor ranges are validated; malformed requests
receive HTTP 422.

## MLflow and CI

Training logs both candidates to the `conveyor-jam-prediction` experiment and
registers the highest-F1 model as `ConveyorJamRiskModel`. Use
`mlflow ui --backend-store-uri sqlite:///mlflow.db` to inspect local runs.
GitHub Actions runs `pytest -q` for pushes and pull requests to `main` or
`master`. A successful local test run does not imply a GitHub Actions run; a
remote repository must be configured and the workflow run completed to provide
that evidence.

## Limitations

The labels and sensor values are synthetic. Metrics in the report measure
performance on a held-out split of this synthetic data only. Real deployment
requires validated sensor calibration, representative labeled shifts,
operational review of alert thresholds, and monitoring for drift.
