from pathlib import Path
import json
import pandas as pd
from src.data_generator import generate_demo_datasets
from src.data_versioning import snapshot_dataset
from src.preprocessing import build_preprocessor
from src.feature_engineering import add_generic_features

def test_demo_data_and_versioning(tmp_path):
    c, r = generate_demo_datasets(tmp_path, n_samples=120)
    assert pd.read_csv(c).shape[0] == 120
    saved, meta = snapshot_dataset(c, tmp_path / "versions")
    assert saved.exists()
    assert len(meta["sha256"]) == 64

def test_preprocessing_handles_mixed_columns():
    x = pd.DataFrame({"n": [1.0, None, 3.0], "c": ["a", None, "b"]})
    transformed = build_preprocessor(x).fit_transform(x)
    assert transformed.shape[0] == 3

def test_generic_feature_engineering():
    x = add_generic_features(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert "__numeric_mean" in x.columns
