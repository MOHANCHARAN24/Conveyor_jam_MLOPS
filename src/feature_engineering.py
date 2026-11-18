import pandas as pd


def add_conveyor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reusable, shift-level features from raw conveyor sensor readings."""
    out = df.copy()
    if "rated_motor_current_a" not in out.columns:
        raise ValueError("rated_motor_current_a is required to calculate motor_load_ratio.")
    if (out["rated_motor_current_a"] <= 0).any():
        raise ValueError("rated_motor_current_a must be greater than zero.")

    required = {
        "motor_current_a",
        "vibration_mm_s",
        "belt_age_years",
        "days_since_maintenance",
    }
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Missing columns required for conveyor features: {missing}")

    out["motor_load_ratio"] = out["motor_current_a"] / out["rated_motor_current_a"]
    out["belt_wear_pressure_index"] = (
        (out["belt_age_years"] / 10.0)
        * (out["vibration_mm_s"] / 10.0)
        * (out["days_since_maintenance"] / 180.0)
    )
    return out
