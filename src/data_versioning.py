from pathlib import Path
import hashlib, json, shutil
from datetime import datetime, timezone

def snapshot_dataset(source_path, versions_dir="data/versions", version=None, notes=""):
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    dest_dir = Path(versions_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("v[0-9][0-9][0-9]"))
    number = int(existing[-1].name[1:]) + 1 if existing else 1
    version = version or f"v{number:03d}"
    version_dir = dest_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    target = version_dir / source.name
    shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "version": version, "file": target.name, "sha256": digest,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return target, metadata

if __name__ == "__main__":
    from src.data_generator import generate_demo_datasets
    paths = generate_demo_datasets()
    for p in paths:
        saved, meta = snapshot_dataset(p, notes="Initial synthetic demo dataset")
        print(saved, meta)
