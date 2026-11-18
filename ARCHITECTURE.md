# Conveyor Jam Prediction Architecture

```mermaid
flowchart LR
  A[Sensor + maintenance signals] --> B[Generate / snapshot V1]
  A --> C[Festival stress transform / snapshot V2]
  B --> D[Validate and combine]
  C --> D
  D --> E[Engineer load ratio + wear pressure]
  E --> F[Reusable CSV feature store]
  E --> G[Stratified train/test split]
  G --> H[Logistic Regression]
  G --> I[Random Forest]
  H --> J[Compare accuracy, jam recall, F1]
  I --> J
  J --> K[MLflow tracking + selected model registry]
  K --> L[FastAPI /predict + /health]
  M[GitHub push / PR] --> N[GitHub Actions]
  N --> O[Sensor validation + API prediction tests]
```

The target, `next_shift_jam`, is a binary label. The prediction is intended for
the next operating shift, not the current shift. Version 2 retains the same
shift-level equipment record while applying a 1.30 throughput multiplier plus
increased wear-related stress. The train/test split is stratified and metrics
are calculated on the positive jam class.
