import pandas as pd

def add_generic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative, domain-neutral feature engineering.
    Adds row-level aggregates from numeric input columns. The target must
    already have been removed before calling this function.
    """
    out = df.copy()
    numeric = out.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric) >= 2:
        out["__numeric_mean"] = out[numeric].mean(axis=1)
        out["__numeric_sum"] = out[numeric].sum(axis=1)
    return out
