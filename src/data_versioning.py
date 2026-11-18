from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

from src.data_generator import generate_conveyor_datasets


def snapshot_dataset(
    source_path: str | Path,
    versions_dir: str | Path = "data/versions",
    version: str | None = None,
    notes: str = "",
) -> tuple[Path, dict]:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    destination_dir = Path(versions_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(destination_dir.glob("v[0-9][0-9][0-9]"))
    number = int(existing[-1].name[1:]) + 1 if existing else 1
    version = version or f"v{number:03d}"
    version_dir = destination_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    target = version_dir / source.name
    shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "version": version,
        "file": target.name,
        "rows": _count_rows(target),
        "sha256": digest,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    (version_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return target, metadata


def _count_rows(path: Path) -> int:
    import pandas as pd

    return len(pd.read_csv(path))


def create_versioned_conveyor_datasets(
    raw_dir: str | Path = "data/raw",
    versions_dir: str | Path = "data/versions/conveyor",
    n_samples: int = 500,
    random_state: int = 42,
) -> list[tuple[Path, dict]]:
    source_paths = generate_conveyor_datasets(
        raw_dir, n_samples=n_samples, random_state=random_state
    )
    notes = [
        "Baseline operating shifts; 500 synthetic records.",
        "Festival shopping shifts; approximately 30% higher parcel throughput and elevated operating stress.",
    ]
    return [
        snapshot_dataset(source, versions_dir, version=f"v{index:03d}", notes=note)
        for index, (source, note) in enumerate(zip(source_paths, notes), start=1)
    ]


if __name__ == "__main__":
    for path, metadata in create_versioned_conveyor_datasets():
        print(path, json.dumps(metadata, indent=2))
