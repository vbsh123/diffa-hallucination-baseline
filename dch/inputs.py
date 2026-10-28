"""Load immutable saved captions; references never enter verification prompts."""
import json
from pathlib import Path
import re

from baseline.common import digest,load_manifest,read_json


def load_inputs(baseline_run=None,manifest_path=None,caption_log=None,samples=None):
    provenance = {}
    if caption_log is not None:
        if baseline_run is not None or manifest_path is None:
            raise ValueError("Caption-log input needs --manifest and cannot also use --baseline-run")
        config = dict(instruction="Describe the audio.",decode=dict(gen_length=128))
        predictions = {}
        for line in Path(caption_log).read_text().splitlines():
            match = re.fullmatch(r"(\d+)/(\d+) remaining clips: ([A-Za-z0-9_-]+): (.+)",line)
            if not match:
                raise ValueError("Caption log must contain only complete caption lines")
            _,_,clip_id,text = match.groups()
            if clip_id in predictions:
                raise ValueError("Duplicate caption-log ID")
            predictions[clip_id] = dict(id=clip_id,text=text)
        provenance = dict(kind="provided_caption_log",caption_log_sha256=digest(caption_log),
                          warning="Original generation config and token diagnostics unavailable; assumes original default instruction and 128 answer positions")
    else:
        baseline_run = Path(baseline_run or "runs/baseline50").resolve()
        config_path = baseline_run/"config.json"
        config = read_json(config_path)
        config_hash = digest(config_path)
        manifest_path = Path(manifest_path or config["manifest"])
        if digest(manifest_path) != config["manifest_sha256"]:
            raise ValueError("Source manifest changed since caption generation")
        predictions = {}
        for path in sorted((baseline_run/"predictions").glob("*.json")):
            row = read_json(path)
            if row["id"] != path.stem or row.get("run_config_sha256") != config_hash:
                raise ValueError(f"Mixed or invalid baseline prediction: {path}")
            if not isinstance(row.get("text"),str):
                raise ValueError(f"Invalid caption text: {path}")
            predictions[row["id"]] = row
        provenance = dict(kind="saved_baseline_run",baseline_config_sha256=config_hash,
                          prediction_hashes={p.stem:digest(p) for p in sorted((baseline_run/"predictions").glob("*.json"))})
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    ids = [r["id"] for r in manifest["samples"]]
    if set(predictions)-set(ids):
        raise ValueError("Captions contain IDs outside this cohort")
    count = len(ids) if samples is None else samples
    if not 1<=count<=len(ids):
        raise ValueError("Sample count must be between one and cohort size")
    selected = manifest["samples"][:count]
    missing = [r["id"] for r in selected if r["id"] not in predictions]
    if missing:
        raise ValueError(f"Missing saved captions: {missing}; generation is never rerun by DCH")
    for row in selected:
        if (row["sample_rate"]!=16000 or row["num_samples"]<=0
                or row["num_samples"]>30*16000):
            raise ValueError(f"Invalid or >30-second input clip: {row['id']}")
    provenance.update(manifest=str(manifest_path),manifest_sha256=digest(manifest_path),
                      selected_ids=[r["id"] for r in selected])
    return manifest_path,selected,predictions,config,provenance
