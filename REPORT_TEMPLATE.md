# Conveyor Jam Prediction Report

The evidence-backed implementation report is generated at
[`reports/conveyor_jam_mlop_report.pdf`](reports/conveyor_jam_mlop_report.pdf).
It includes measured dataset-version statistics, the held-out model comparison
and confusion matrices, local test output, registry metadata, and a prediction
from the saved FastAPI model.

Regenerate after dataset/model changes:

```powershell
python -m src.data_versioning
python -m src.train --config config/config.json
python scripts/build_report.py
```

Hosted workflow status is included only when a CI evidence record exists. Local
test success is reported separately from hosted GitHub Actions status.
