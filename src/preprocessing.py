import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SENSOR_BOUNDS = {
    "motor_current_a": (0.0, 100.0),
    "rated_motor_current_a": (0.01, 100.0),
    "vibration_mm_s": (0.0, 50.0),
    "belt_age_years": (0.0, 50.0),
    "parcels_per_minute": (0.0, 1000.0),
    "dust_accumulation_index": (0.0, 100.0),
    "days_since_maintenance": (0.0, 3650.0),
    "humidity_pct": (0.0, 100.0),
}
SENSOR_COLUMNS = list(SENSOR_BOUNDS)
TARGET_COLUMN = "next_shift_jam"
FEATURE_COLUMNS = SENSOR_COLUMNS + [
    "motor_load_ratio",
    "belt_wear_pressure_index",
]


def validate_sensor_frame(df: pd.DataFrame, require_target: bool = False) -> None:
    required = SENSOR_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")
    for column, (minimum, maximum) in SENSOR_BOUNDS.items():
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        if ((values < minimum) | (values > maximum)).any():
            raise ValueError(f"{column} must be between {minimum} and {maximum}.")
    if require_target:
        labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{TARGET_COLUMN} must contain only 0 or 1.")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found.")
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipe, numeric_cols)],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except (AttributeError, ValueError):
        return []
