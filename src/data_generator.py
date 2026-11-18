from pathlib import Path

import numpy as np
import pandas as pd


def _jam_probability(frame: pd.DataFrame) -> np.ndarray:
    load_ratio = frame["motor_current_a"] / frame["rated_motor_current_a"]
    score = (
        -5.8
        + 4.0 * load_ratio
        + 0.25 * frame["vibration_mm_s"]
        + 0.018 * frame["dust_accumulation_index"]
        + 0.005 * frame["days_since_maintenance"]
        + 0.07 * frame["belt_age_years"]
        + 0.012 * (frame["humidity_pct"] - 50)
    )
    return 1.0 / (1.0 + np.exp(-score))


def generate_conveyor_datasets(
    output_dir: str | Path = "data/raw",
    n_samples: int = 500,
    random_state: int = 42,
) -> tuple[Path, Path]:
    if n_samples < 250:
        raise ValueError("n_samples must be at least 250.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    rated_current = rng.choice([15.0, 20.0, 25.0, 30.0], size=n_samples)
    baseline = pd.DataFrame(
        {
            "rated_motor_current_a": rated_current,
            "vibration_mm_s": np.clip(rng.normal(4.0, 1.4, n_samples), 0.5, 9.0),
            "belt_age_years": rng.uniform(0.5, 12.0, n_samples),
            "parcels_per_minute": rng.uniform(50.0, 90.0, n_samples),
            "dust_accumulation_index": rng.uniform(5.0, 65.0, n_samples),
            "days_since_maintenance": rng.integers(0, 121, n_samples),
            "humidity_pct": rng.uniform(30.0, 80.0, n_samples),
        }
    )
    baseline_load = np.clip(
        0.62
        + 0.003 * (baseline["parcels_per_minute"] - 70)
        + 0.0015 * (baseline["dust_accumulation_index"] - 35)
        + 0.025 * (baseline["vibration_mm_s"] - 4)
        + 0.018 * (baseline["belt_age_years"] - 5)
        + 0.0007 * (baseline["days_since_maintenance"] - 60)
        + rng.normal(0.0, 0.055, n_samples),
        0.40,
        1.30,
    )
    baseline["motor_current_a"] = baseline["rated_motor_current_a"] * baseline_load

    festival = baseline.copy()
    festival["parcels_per_minute"] *= 1.30
    festival["vibration_mm_s"] = np.clip(
        festival["vibration_mm_s"] * 1.12 + 0.25, 0.5, 12.0
    )
    festival["dust_accumulation_index"] = np.clip(
        festival["dust_accumulation_index"] + 10.0, 0.0, 100.0
    )
    festival["days_since_maintenance"] = np.minimum(
        festival["days_since_maintenance"] + 3, 365
    )
    festival["humidity_pct"] = np.clip(festival["humidity_pct"] + 4.0, 0.0, 100.0)
    festival_load = np.clip(
        baseline_load
        + 0.04
        + 0.0015 * (festival["parcels_per_minute"] - baseline["parcels_per_minute"])
        + rng.normal(0.0, 0.025, n_samples),
        0.40,
        1.40,
    )
    festival["motor_current_a"] = festival["rated_motor_current_a"] * festival_load

    datasets = []
    for frame, version in ((baseline, "v1"), (festival, "v2")):
        frame = frame.copy()
        frame["dataset_version"] = version
        frame["next_shift_jam"] = rng.binomial(1, _jam_probability(frame))
        datasets.append(frame)

    paths = (
        output / "conveyor_shifts_v1.csv",
        output / "conveyor_shifts_v2.csv",
    )
    for frame, path in zip(datasets, paths):
        frame.to_csv(path, index=False)
    return paths


if __name__ == "__main__":
    version_1, version_2 = generate_conveyor_datasets()
    print(f"Created {version_1} and {version_2}")
