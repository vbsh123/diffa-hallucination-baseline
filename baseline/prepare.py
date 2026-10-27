"""Pin a seeded AudioCaps/ECHO cohort and materialize complete 16-kHz WAVs."""
import argparse
import io
import math
import random
import re
from pathlib import Path

from .common import DATASET, DATASET_REVISION, SEED, digest, load_manifest, write_json


def audio_array(audio):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not isinstance(audio, dict):
        raise ValueError("Expected a datasets.Audio(decode=False) dictionary")
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    if not len(samples) or not np.isfinite(samples).all() or rate <= 0:
        raise ValueError("Empty or non-finite audio")
    samples = samples.mean(axis=1)
    duration = len(samples) / rate
    if duration > 30:
        raise ValueError("Clip exceeds the native 30-second encoder window; no cropping")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype("float32")
    if not np.isfinite(samples).all():
        raise ValueError("Resampling produced non-finite values")
    return samples, duration


def prepare(out, count=50, seed=SEED, dataset_revision=DATASET_REVISION):
    out = Path(out).resolve()
    config = dict(dataset=DATASET, revision=dataset_revision, split="test", samples=count,
                  seed=seed, audio_format="mono-float32-wav-16000-v1")
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest["selection"] != config:
            raise ValueError("Cohort settings changed: use a new data directory")
        print(f"Verified existing cohort: {manifest_path}", flush=True)
        return manifest_path
    if count < 1:
        raise ValueError("Sample count must be positive")
    from datasets import Audio, load_dataset
    import soundfile as sf

    dataset = load_dataset(DATASET, revision=dataset_revision, split="test")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if count > len(dataset):
        raise ValueError(f"Requested {count} clips from {len(dataset)} rows")
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    rows, excluded, seen = [], [], set()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    for index in order:
        row = dataset[index]
        clip_id = row["youtube_id"]
        if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
            raise ValueError("Invalid dataset clip identifier")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip identifier: {clip_id}")
        seen.add(clip_id)
        if (not isinstance(row["caption"], list) or not row["caption"] or
                not all(isinstance(x, str) and x.strip() for x in row["caption"]) or
                not isinstance(row["label"], list) or
                not all(isinstance(x, str) for x in row["label"])):
            raise ValueError(f"Unexpected annotation schema for {clip_id}")
        try:
            samples, original_duration = audio_array(row["audio"])
        except (ValueError, RuntimeError) as error:
            excluded.append(dict(id=clip_id, dataset_index=index, reason=str(error)))
            continue
        destination = out / "audio" / f"{clip_id}.wav"
        sf.write(destination, samples, 16000, subtype="FLOAT")
        rows.append(dict(id=clip_id, dataset_index=index,
                         audio_path=f"audio/{clip_id}.wav", audio_sha256=digest(destination),
                         sample_rate=16000, num_samples=len(samples),
                         duration_seconds=len(samples) / 16000,
                         original_duration_seconds=original_duration,
                         reference_captions=row["caption"], labels=row["label"]))
        print(f"Prepared {len(rows)}/{count}: {clip_id}", flush=True)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"Only {len(rows)} eligible complete clips; exclusions={excluded}")
    write_json(manifest_path, dict(schema_version=1, selection=config, source_rows=len(dataset),
                                   excluded_before_selection=excluded, samples=rows))
    print(f"Saved {manifest_path}", flush=True)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/audiocaps50"))
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    prepare(args.out, args.samples, args.seed)


if __name__ == "__main__":
    main()
