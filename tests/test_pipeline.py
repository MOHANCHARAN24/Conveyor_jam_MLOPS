import json

import pandas as pd
import pytest

from src.data_generator import generate_conveyor_datasets
from src.data_versioning import create_versioned_conveyor_datasets
from src.feature_engineering import add_conveyor_features
from src.preprocessing import validate_sensor_frame


def test_generates_baseline_and_festival_versions(tmp_path):
    baseline_path, festival_path = generate_conveyor_datasets(
        tmp_path, n_samples=250
    )
    baseline = pd.read_csv(baseline_path)
    festival = pd.read_csv(festival_path)

    assert len(baseline) == len(festival) == 250
    assert baseline["next_shift_jam"].isin([0, 1]).all()
    assert festival["parcels_per_minute"].mean() == pytest.approx(
        baseline["parcels_per_minute"].mean() * 1.3
    )
    assert (
        festival["motor_current_a"].div(festival["rated_motor_current_a"]).mean()
        > baseline["motor_current_a"].div(baseline["rated_motor_current_a"]).mean()
    )


def test_versioning_writes_checksums_and_row_counts(tmp_path):
    versioned = create_versioned_conveyor_datasets(
        raw_dir=tmp_path / "raw",
        versions_dir=tmp_path / "versions",
        n_samples=250,
    )
    assert [metadata["version"] for _, metadata in versioned] == ["v001", "v002"]
    assert all(path.exists() and metadata["rows"] == 250 for path, metadata in versioned)
    saved_metadata = json.loads((tmp_path / "versions" / "v002" / "metadata.json").read_text())
    assert len(saved_metadata["sha256"]) == 64


def test_sensor_validation_rejects_out_of_range_values():
    row = {
        "motor_current_a": 10,
        "rated_motor_current_a": 20,
        "vibration_mm_s": 3,
        "belt_age_years": 2,
        "parcels_per_minute": 80,
        "dust_accumulation_index": 20,
        "days_since_maintenance": 30,
        "humidity_pct": 45,
        "next_shift_jam": 0,
    }
    validate_sensor_frame(pd.DataFrame([row]), require_target=True)
    row["humidity_pct"] = 101
    with pytest.raises(ValueError, match="humidity_pct"):
        validate_sensor_frame(pd.DataFrame([row]), require_target=True)


def test_conveyor_feature_engineering():
    row = pd.DataFrame(
        {
            "motor_current_a": [12.0],
            "rated_motor_current_a": [20.0],
            "vibration_mm_s": [5.0],
            "belt_age_years": [4.0],
            "days_since_maintenance": [90.0],
        }
    )
    result = add_conveyor_features(row)
    assert result.loc[0, "motor_load_ratio"] == pytest.approx(0.6)
    assert result.loc[0, "belt_wear_pressure_index"] == pytest.approx(0.1)
