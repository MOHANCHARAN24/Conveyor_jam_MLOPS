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

The report labels hosted GitHub Actions evidence as pending until this checkout
has a GitHub remote and the workflow completes successfully.
