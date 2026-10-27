"""Small dependency-free helpers shared by preparation, inference and reporting."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "kuanhuggingface/audiocaps_hallucination"
DATASET_REVISION = "89ecbf42744b923421e14dff2ae484d015d88391"
SEED = 20261027


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def claim_config(path, value):
    """Resume only the same experiment; never overwrite a different protocol."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Configuration changed: use a new output directory ({path})")
    else:
        write_json(path, value)


def load_manifest(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or not manifest.get("samples"):
        raise ValueError("Expected a nonempty version-1 manifest")
    identifiers = [row["id"] for row in manifest["samples"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate sample IDs")
    for row in manifest["samples"]:
        audio = (path.parent / row["audio_path"]).resolve()
        if not audio.is_relative_to(path.parent) or not audio.is_file():
            raise ValueError(f"Invalid local audio path for {row['id']}")
        if digest(audio) != row["audio_sha256"]:
            raise ValueError(f"Audio changed for {row['id']}")
        if not row.get("reference_captions") or not isinstance(row["labels"], list):
            raise ValueError(f"Missing evaluation annotations for {row['id']}")
    return manifest
