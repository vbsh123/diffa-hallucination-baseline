"""Generate untouched DIFFA-2 captions with the official FPD implementation."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import random
import time

from .checkout import COMMIT, verify_checkout
from .common import ROOT, SEED, claim_config, digest, load_manifest, read_json, write_json

PROMPT = "Describe the audio."
DECODE = dict(steps=128, gen_length=128, block_length=32, temperature=0.,
              remasking="low_confidence", threshold=None, factor=1.0)


def source_fingerprint():
    paths = sorted((ROOT / "baseline").glob("*.py")) + [ROOT / "requirements.txt"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def output_record(ids, eos_id, mask_id, tokenizer):
    has_eos = eos_id in ids
    visible = ids[:ids.index(eos_id)] if has_eos else ids
    return dict(canvas_ids=ids, visible_ids=visible,
                text=tokenizer.decode(visible, skip_special_tokens=True).strip(),
                eos_found=has_eos, capacity_exhausted=not has_eos,
                unresolved_visible_masks=visible.count(mask_id), visible_tokens=len(visible))


def run(manifest_path, out, diffa_root, device="cuda:0", seed=SEED, instruction=PROMPT):
    import numpy as np
    import soundfile as sf
    import torch
    from .model import (DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION,
                        WHISPER_REPO, WHISPER_REVISION, load_model, prepare)

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Inference needs a CUDA GPU; data preparation and scoring work on CPU")
    torch.cuda.set_device(torch.device(device))
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This pinned DIFFA loader requires a GPU with bfloat16 support")
    if not instruction.strip():
        raise ValueError("Caption instruction must be nonempty")
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    manifest = load_manifest(manifest_path)
    verify_checkout(diffa_root)
    packages = ["torch", "torchaudio", "transformers", "huggingface-hub", "peft",
                "numpy", "soundfile", "accelerate", "einops", "sentencepiece"]
    config = dict(schema_version=1, manifest=str(manifest_path),
                  manifest_sha256=digest(manifest_path), upstream_commit=COMMIT,
                  implementation=source_fingerprint(),
                  models=dict(diffa=[DIFFA_REPO, DIFFA_REVISION], llm=[LLM_REPO, LLM_REVISION],
                              whisper=[WHISPER_REPO, WHISPER_REVISION]),
                  instruction=instruction, decode=DECODE, seed=seed,
                  python=platform.python_version(), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),
                  packages={p: importlib.metadata.version(p) for p in packages})
    claim_config(out / "config.json", config)
    config_hash = digest(out / "config.json")
    pending = []
    for sample in manifest["samples"]:
        path = out / "predictions" / f"{sample['id']}.json"
        if path.exists():
            saved = read_json(path)
            if (saved.get("id") != sample["id"] or
                    saved.get("run_config_sha256") != config_hash or
                    not isinstance(saved.get("text"), str)):
                raise ValueError(f"Invalid or mixed-run prediction: {path}")
        else:
            pending.append(sample)
    if pending:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Temperature is zero; these settings also avoid TF32 precision changes.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, tokenizer, extractor, load_report = load_model(
            diffa_root, DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION,
            WHISPER_REPO, WHISPER_REVISION, device=device)
        write_json(out / "model_load.json", load_report)
        from src.fast_dllm import diffa_generate as upstream

        with torch.inference_mode():
            for index, sample in enumerate(pending):
                audio, rate = sf.read(manifest_path.parent / sample["audio_path"], dtype="float32")
                if (rate != 16000 or audio.ndim != 1 or len(audio) != sample["num_samples"]
                        or not np.isfinite(audio).all() or len(audio) > 30 * rate):
                    raise ValueError(f"Invalid prepared audio: {sample['id']}")
                torch.cuda.synchronize()
                start = time.perf_counter()
                prepared = prepare(model, tokenizer, extractor, audio, instruction, DECODE["gen_length"])
                output, nfe = upstream.generate(
                    model, prepared.prompt,
                    audio_info_lengths=torch.tensor([prepared.insertion], device=prepared.prompt.device),
                    speech_embeds=prepared.speech, mask_id=prepared.mask_id, **DECODE)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                ids = output[0, prepared.prompt.shape[1]:].cpu().tolist()
                if len(ids) != DECODE["gen_length"]:
                    raise ValueError(f"Unexpected answer canvas length: {len(ids)}")
                record = output_record(ids, tokenizer.eos_token_id, prepared.mask_id, tokenizer)
                record.update(id=sample["id"], run_config_sha256=config_hash,
                              actual_nfe=int(nfe), elapsed_seconds=elapsed,
                              audio_tokens=prepared.speech.shape[1])
                write_json(out / "predictions" / f"{sample['id']}.json", record)
                print(f"{index+1}/{len(pending)} remaining clips: {sample['id']}: {record['text']}", flush=True)
    from .score import score_run
    score_run(out, manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/audiocaps50/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/baseline50"))
    parser.add_argument("--diffa-root", type=Path, default=Path("external/DIFFA"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--instruction", default=PROMPT)
    args = parser.parse_args()
    run(args.manifest, args.out, args.diffa_root, args.device, args.seed, args.instruction)


if __name__ == "__main__":
    main()
