"""Apply paper-based Detect–Cut–Heal to saved DIFFA captions without regeneration."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import random
import time

from baseline.common import ROOT,claim_config,digest,read_json,write_json
from baseline.checkout import COMMIT,verify_checkout
from .inputs import load_inputs
from .nlp import PaperNLP
from .cut import cut_caption


def fingerprint():
    paths = sorted((ROOT/"dch").rglob("*.py"))+sorted((ROOT/"baseline").glob("*.py"))
    paths += [ROOT/"requirements.txt",ROOT/"requirements-dch.txt"]
    paths += [ROOT/"dch/reference/SOURCE.json"]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    from baseline.model import (DIFFA_REPO,DIFFA_REVISION,LLM_REPO,LLM_REVISION,
                                WHISPER_REPO,WHISPER_REVISION,load_model,prepare)
    from .backend import DiffaBackend,QUESTION
    from .heal import heal_caption
    from .report import report

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("DCH inference needs a CUDA GPU; CPU tests use tiny/synthetic models only")
    torch.cuda.set_device(torch.device(args.device))
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Use a GPU supporting bfloat16")
    manifest_path,samples,predictions,source,origin = load_inputs(
        args.baseline_run,args.manifest,args.caption_log,args.samples,args.brace_cohort)
    models = dict(diffa=[DIFFA_REPO,DIFFA_REVISION],llm=[LLM_REPO,LLM_REVISION],
                  whisper=[WHISPER_REPO,WHISPER_REVISION])
    if source.get("models",models) != models:
        raise ValueError("Saved captions use different model revisions; this first adaptation requires the pinned DIFFA-2")
    verify_checkout(args.diffa_root)
    nlp = PaperNLP(audio_meta=args.audio_meta_stoplist)
    out = args.out.resolve()
    config = dict(schema_version=1,method="released_DCH_DIFFA_audio_adapter_v1",
                  inputs=origin,models=models,upstream_commit=COMMIT,
                  source=fingerprint(),nlp=nlp.provenance(),question=QUESTION,
                  caption_instruction=source["instruction"],threshold=0.,answer_masks=2,
                  repair_window=2,commit_tokens_per_forward=1,nonword_retries=2,
                  new_object_retry=1,detect_only=args.detect_only,seed=20261029,
                  failure_policy="released retry limits; no additional rewrite or rollback",
                  candidate_unit="word-level noun lemma, not compound/event claims",
                  python=platform.python_version(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                  packages={p:importlib.metadata.version(p) for p in ["torch","transformers","peft","nltk","numpy","spacy","huggingface-hub"]})
    claim_config(out/"config.json",config)
    config_hash = digest(out/"config.json")
    pending = []
    for sample in samples:
        path = out/"clips"/f"{sample['id']}.json"
        if path.exists():
            saved = read_json(path)
            if saved.get("config_sha256")!=config_hash or saved.get("original")!=predictions[sample["id"]]["text"]:
                raise ValueError(f"Mixed or changed DCH record: {path}")
        else:
            pending.append(sample)
    if pending:
        random.seed(20261029)
        np.random.seed(20261029)
        torch.manual_seed(20261029)
        torch.cuda.manual_seed_all(20261029)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model,tokenizer,extractor,load_report = load_model(
            args.diffa_root,DIFFA_REPO,DIFFA_REVISION,LLM_REPO,LLM_REVISION,
            WHISPER_REPO,WHISPER_REVISION,device=args.device)
        write_json(out/"model_load.json",load_report)
        nlp.tokenizer = tokenizer
        with torch.inference_mode():
            for sample in pending:
                text = predictions[sample["id"]]["text"]
                candidates = nlp.candidates(text)
                audio,rate = sf.read(manifest_path.parent/sample["audio_path"],dtype="float32")
                if (rate!=16000 or audio.ndim!=1 or not len(audio) or len(audio)>30*rate
                        or len(audio)!=sample["num_samples"] or not np.isfinite(audio).all()):
                    raise ValueError(f"Invalid complete input audio: {sample['id']}")
                torch.cuda.synchronize()
                start = time.perf_counter()
                prepared = prepare(model,tokenizer,extractor,audio,source["instruction"],2)
                backend = DiffaBackend(model,tokenizer,prepared.speech,source["instruction"])
                local_prompt,insertion = backend.prompt(source["instruction"])
                if insertion!=prepared.insertion or not torch.equal(local_prompt,prepared.prompt):
                    raise AssertionError("DCH prompt does not match original DIFFA embedding insertion")
                detections = [dict(backend.verify(g["lemma"])) for g in sorted(candidates,key=lambda g:g["lemma"])]
                detector_nfe = backend.nfe
                if detector_nfe!=len(candidates):
                    raise AssertionError("Detector did not use exactly one forward per unique noun")
                record = dict(id=sample["id"],config_sha256=config_hash,original=text,
                              candidates=candidates,detections=detections,detector_nfe=detector_nfe,
                              answer_variant_ids=backend.answer_ids,
                              source_generation_diagnostics={k:predictions[sample["id"]].get(k)
                                  for k in ["eos_found","capacity_exhausted","unresolved_visible_masks"]})
                if not args.detect_only:
                    flagged = {x["lemma"] for x in detections if x["flagged"]}
                    accepted = {x["lemma"] for x in detections if not x["flagged"]}
                    cutting = cut_caption(text,candidates,flagged,nlp)
                    healing = heal_caption(text,cutting["text"],backend,nlp,flagged,accepted,
                                           capacity=source.get("decode",{}).get("gen_length",128))
                    record.update(cut=cutting["text"],healed=healing["text"],cutting=cutting,healing=healing)
                torch.cuda.synchronize()
                record.update(total_added_nfe=backend.nfe,elapsed_seconds=time.perf_counter()-start,
                              audio_tokens=prepared.speech.shape[1])
                write_json(out/"clips"/f"{sample['id']}.json",record)
                flags = [x["lemma"] for x in detections if x["flagged"]]
                print(f"{sample['id']}: candidates={len(candidates)} flagged={flags} nfe={backend.nfe}",flush=True)
                if not args.detect_only:
                    print(f"  CUT: {record['cut']}\n  HEAL: {record['healed']}",flush=True)
    report(out,samples,manifest_path,config_hash,args.detect_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run",type=Path)
    parser.add_argument("--brace-cohort",type=Path,help="Prepared paired BRACE manifest; --samples counts audio clips")
    parser.add_argument("--caption-log",type=Path,help="Fallback for pasted logs; original config unavailable")
    parser.add_argument("--manifest",type=Path,help="Optional relocated manifest; original hash must match")
    parser.add_argument("--out",type=Path,default=Path("runs/dch50"))
    parser.add_argument("--diffa-root",type=Path,default=Path("external/DIFFA"))
    parser.add_argument("--samples",type=int)
    parser.add_argument("--detect-only",action="store_true")
    parser.add_argument("--audio-meta-stoplist",action="store_true",
                        help="Optional ablation: additionally exclude audio/recording/clip; default uses literal supplied stoplists")
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    run(args)


if __name__=="__main__":
    main()
