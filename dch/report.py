"""Paired proxy scores and blinded inputs, not self-verification as ground truth."""
import csv
import hashlib
import json
from pathlib import Path
import random

from baseline.common import read_json,write_json


def report(out,samples,manifest_path,config_hash,detect_only=False):
    if samples and "brace" in samples[0]:
        from .brace import report as brace_report
        return brace_report(out,samples,manifest_path,config_hash,detect_only)
    import spacy
    from baseline.score import NLP_MODEL,NLP_VERSION,aggregate,compare,extract
    out = Path(out)
    nlp = spacy.load(NLP_MODEL)
    if nlp.meta["version"] != NLP_VERSION:
        raise ValueError("Scoring NLP version differs from the pinned baseline")
    records = []
    for sample in samples:
        path = out/"clips"/f"{sample['id']}.json"
        if path.exists():
            record = read_json(path)
            if record["id"] != sample["id"] or record["config_sha256"] != config_hash:
                raise ValueError("Mixed DCH output records")
            records.append((sample,record))
    versions = ["original"] if detect_only else ["original","cut","healed"]
    scores = {version:[] for version in versions}
    for sample,row in records:
        references = set().union(*(set(extract(t,nlp)["terms"])
                                    for t in sample["reference_captions"]+sample["labels"]))
        for version in versions:
            score = compare(extract(row[version],nlp)["terms"],references)
            score["id"] = row["id"]
            scores[version].append(score)
    summary = dict(expected_clips=len(samples),completed_clips=len(records),
                   complete=len(records)==len(samples),detect_only=detect_only,
                   candidate_objects=sum(len(r["detections"]) for _,r in records),
                   flagged_objects=sum(x["flagged"] for _,r in records for x in r["detections"]),
                   detector_nfe=sum(r["detector_nfe"] for _,r in records),
                   total_added_nfe=sum(r["total_added_nfe"] for _,r in records),
                   warning="Noun overlap is the unchanged legacy lexical proxy, not a factual hallucination metric. Detector rejections are intervention decisions, not ground-truth hallucinations.",
                   paired_lexical_proxy={v:aggregate(rows,len(samples)) for v,rows in scores.items()})
    if not detect_only:
        summary.update(captions_cut=sum(r["original"]!=r["cut"] for _,r in records),
                       captions_healed=sum(r["cut"]!=r["healed"] for _,r in records),
                       repair_fallbacks=sum(r["healing"]["status"].startswith("fallback") for _,r in records),
                       remaining_flagged_lemmas=sum(len(r["cutting"]["remaining_flagged_lemmas"]) for _,r in records),
                       all_fixed_tokens_preserved=all(r["healing"]["fixed_tokens_preserved"] for _,r in records))
    write_json(out/"summary.json",summary)
    write_json(out/"paired_proxy_scores.json",scores)
    fields = ["id","original","cut","healed","flagged","remaining_flagged","repair_status",
              "reference_captions","labels","audio_file","review_notes"]
    notes = {}
    if (out/"review.csv").exists():
        with (out/"review.csv").open(newline="") as stream:
            notes = {r["id"]:r.get("review_notes","") for r in csv.DictReader(stream)}
    with (out/"review.csv.tmp").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        for sample,row in records:
            writer.writerow(dict(id=row["id"],original=row["original"],cut=row.get("cut",""),
                                 healed=row.get("healed",""),
                                 flagged=" | ".join(x["lemma"] for x in row["detections"] if x["flagged"]),
                                 remaining_flagged=" | ".join(row.get("cutting",{}).get("remaining_flagged_lemmas",[])),
                                 repair_status=row.get("healing",{}).get("status","detect_only"),
                                 reference_captions=" | ".join(sample["reference_captions"]),
                                 labels=" | ".join(sample["labels"]),
                                 audio_file=str(manifest_path.parent/sample["audio_path"]),
                                 review_notes=notes.get(row["id"],"")))
    (out/"review.csv.tmp").replace(out/"review.csv")
    # Three balanced sessions. Each session has one version per clip, and each
    # clip appears in every version across sessions. Method key is outside inputs.
    key = {}
    for session in range(len(versions)):
        items = []
        for index,(sample,row) in enumerate(records):
            version = versions[(index+session)%len(versions)]
            item_id = hashlib.sha256(f"{config_hash}:{row['id']}:{version}".encode()).hexdigest()[:20]
            items.append(dict(item_id=item_id,audio_file=str(manifest_path.parent/sample["audio_path"]),
                              caption=row[version],reference_captions=sample["reference_captions"],
                              labels=sample["labels"]))
            key[item_id] = dict(clip_id=row["id"],version=version)
        random.Random(20261029+session).shuffle(items)
        path = out/"judge_inputs"/f"session_{session}.jsonl"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
    write_json(out/"private_judge_key.json",key)
    print(json.dumps({k:v for k,v in summary.items() if k!="paired_lexical_proxy"},indent=2),flush=True)
    return summary
