"""Prepare and score a small paired BRACE-Hallucination intervention pilot.

The released human/corrupted captions are inputs, not new DIFFA generations.
Ground-truth sides and swapped nouns are used only by the offline report.
"""
import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

from baseline.common import claim_config, digest, load_manifest, read_json, write_json

META_COMMIT = "709b44d24ec79d3296b45790cefb238292cbc3d5"
AUDIO_REPO = "gtysssp/audio_benchmarks"
AUDIO_REVISION = "60729676c42a9430b5134cbf2ff81289b35730d6"
META_SHA256 = "969f0f6506b68980eec572cfd57fefa5c3a0d5c5e2a3c9035c6c6cffcaa7add1"
META_URL = ("https://raw.githubusercontent.com/HychTus/BRACE_Evaluation/"
            + META_COMMIT + "/BRACE/meta/AudioCaps_Hallu.json")
SEED = 20261029


def words(text):
    return re.findall(r"[A-Za-z]+", text.lower())


def mentions(text, nlp):
    # POS-independent presence check: deletion often leaves ungrammatical text.
    return {nlp.lemmatizer.lemmatize(w) for w in words(text)}


def noun_inventory(text, nlp):
    return {nlp.lemmatizer.lemmatize(w.lower())
            for w, tag in nlp.nltk.pos_tag(nlp.nltk.word_tokenize(text))
            if tag.startswith("NN") and w.isalpha()}


def parse_pair(row, key, nlp):
    values = row[key]
    if (len(values) != 5 or set(values[2:4]) != {"human", "hallucination"}
            or not all(isinstance(t, str) and t.strip() for t in values[:2])):
        raise ValueError(f"Unexpected released pair schema: {row['file_name']}:{key}")
    human_index = values[2:4].index("human")
    human, corrupted = values[human_index], values[1-human_index]
    references = values[4]["references"]
    if not isinstance(references, list) or not all(isinstance(t, str) for t in references):
        raise ValueError("Invalid BRACE reference captions")
    before, after = words(human), words(corrupted)
    edits = [dict(before=before[i:j], after=after[k:l])
             for op, i, j, k, l in difflib.SequenceMatcher(
                 None, before, after, autojunk=False).get_opcodes() if op != "equal"]
    # Include only new nouns in changed spans, never punctuation-only edits.
    changed = {nlp.lemmatizer.lemmatize(w) for e in edits for w in e["after"]}
    # Do not let the detector's POS mistakes hide a known injected noun from
    # evaluation. E.g. NLTK tags "elephant" as JJ in "an elephant ticks".
    noun_words = noun_inventory(corrupted, nlp) | {
        w for w in changed if len(w) >= 3 and nlp.wn.synsets(w, pos=nlp.wn.NOUN)}
    injected = sorted((noun_words & changed) - mentions(human, nlp))
    return dict(pair_key=key, human=human, corrupted=corrupted,
                human_side=human_index, references=references, token_edits=edits,
                injected_noun_lemmas=injected,
                automatic_target_status="scorable" if injected else "no_distinct_changed_noun")


def prepare(out, clips=5, seed=SEED):
    from huggingface_hub import hf_hub_download
    import soundfile as sf
    from baseline.prepare import audio_array
    from .nlp import PaperNLP

    if clips < 1:
        raise ValueError("--clips must be positive")
    out = Path(out).resolve()
    nlp = PaperNLP()
    selection = dict(benchmark="BRACE-Hallucination", subset="AudioCaps", clips=clips,
                     pairs_per_clip=1, seed=seed, metadata_url=META_URL,
                     metadata_sha256=META_SHA256, audio_repo=AUDIO_REPO,
                     audio_revision=AUDIO_REVISION, audio_format="mono-float32-wav-16000-v1",
                     target_rule="changed-word noun sense or NN tag; absent from original caption v1",
                     nlp=nlp.provenance())
    manifest_path = out/"manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest["selection"] != selection:
            raise ValueError("BRACE cohort settings changed; use a new data directory")
        print(f"Verified existing BRACE cohort: {manifest_path}", flush=True)
        return manifest_path
    claim_config(out/"selection.json", selection)
    metadata_path = out/"source_metadata.json"
    if not metadata_path.exists():
        with urllib.request.urlopen(META_URL, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != META_SHA256:
            raise ValueError("Pinned BRACE metadata hash mismatch")
        metadata_path.write_bytes(raw)
    if digest(metadata_path) != META_SHA256:
        raise ValueError("Local BRACE metadata changed")
    metadata = read_json(metadata_path)
    filenames = [r["file_name"] for r in metadata]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate BRACE audio filenames")
    if clips > len(metadata):
        raise ValueError(f"Requested {clips} distinct clips from {len(metadata)}")
    order = list(range(len(metadata)))
    random.Random(seed).shuffle(order)
    pairs, samples, excluded = [], [], []
    for index in order:
        row = metadata[index]
        filename = row["file_name"]
        if not re.fullmatch(r"[0-9]+\.wav", filename):
            raise ValueError(f"Unexpected AudioCaps filename: {filename}")
        keys = sorted(k for k in row if re.fullmatch(r"caption_[0-9]+", k))
        if not keys:
            raise ValueError(f"Missing captions for {filename}")
        # Fixed per-audio random selection, before seeing model scores or targets.
        key = random.Random(f"{seed}:{filename}").choice(keys)
        pair = parse_pair(row, key, nlp)
        remote = f"benchmark_hallucination/AudioCaps/audio/{filename}"
        source = Path(hf_hub_download(AUDIO_REPO, remote, repo_type="dataset",
                                    revision=AUDIO_REVISION))
        try:
            audio, original_duration = audio_array(dict(path=str(source)))
        except (ValueError, RuntimeError) as error:
            excluded.append(dict(file_name=filename, reason=str(error)))
            continue
        pair_id = f"brace_{Path(filename).stem}_{key}"
        relative = f"audio/{Path(filename).stem}.wav"
        destination = out/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, 16000, subtype="FLOAT")
        pair.update(pair_id=pair_id, file_name=filename, metadata_index=index,
                    source_audio_path=remote, source_audio_sha256=digest(source),
                    control_id=pair_id+"_human", corrupted_id=pair_id+"_corrupted")
        pairs.append(pair)
        for role, text in [("human", pair["human"]), ("corrupted", pair["corrupted"])]:
            samples.append(dict(id=pair_id+"_"+role, audio_path=relative,
                                audio_sha256=digest(destination), sample_rate=16000,
                                num_samples=len(audio), duration_seconds=len(audio)/16000,
                                original_duration_seconds=original_duration,
                                reference_captions=[pair["human"]]+pair["references"], labels=[],
                                brace=dict(pair_id=pair_id, role=role, text=text)))
        print(f"Prepared {len(pairs)}/{clips}: {filename} {key}; "
              f"injected nouns={pair['injected_noun_lemmas']}", flush=True)
        if len(pairs) == clips:
            break
    if len(pairs) != clips:
        raise ValueError(f"Only {len(pairs)} eligible clips; exclusions={excluded}")
    write_json(out/"pairs.json", dict(schema_version=1, pairs=pairs))
    write_json(manifest_path, dict(schema_version=1, selection=selection,
                                  source_rows=len(metadata), excluded_before_selection=excluded,
                                  pairs_sha256=digest(out/"pairs.json"), samples=samples))
    print(f"Saved {clips} clips / {len(samples)} caption cases: {manifest_path}", flush=True)
    return manifest_path


def load_inputs(manifest_path, clips=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    pairs_path = manifest_path.parent/"pairs.json"
    if digest(pairs_path) != manifest["pairs_sha256"]:
        raise ValueError("BRACE pair annotations changed")
    pairs = read_json(pairs_path)["pairs"]
    count = len(pairs) if clips is None else clips
    if not 1 <= count <= len(pairs):
        raise ValueError("BRACE sample count must be between one and the prepared clip count")
    selected_pairs = pairs[:count]
    case_ids = [p[k] for p in selected_pairs for k in ("control_id", "corrupted_id")]
    by_id = {s["id"]: s for s in manifest["samples"]}
    samples = [by_id[i] for i in case_ids]
    for sample in samples:
        if (sample["sample_rate"] != 16000 or not 0 < sample["num_samples"] <= 480000):
            raise ValueError("Invalid or >30-second BRACE input")
    predictions = {}
    for pair in selected_pairs:
        for role, id_key in [("human", "control_id"), ("corrupted", "corrupted_id")]:
            case_id = pair[id_key]
            if by_id[case_id]["brace"] != dict(pair_id=pair["pair_id"], role=role, text=pair[role]):
                raise ValueError("Manifest caption differs from released BRACE pair")
            predictions[case_id] = dict(id=case_id, text=pair[role])
    config = dict(instruction="Describe the audio.", decode=dict(gen_length=128))
    origin = dict(kind="released_brace_caption_pairs", manifest=str(manifest_path),
                  manifest_sha256=digest(manifest_path), pairs_sha256=digest(pairs_path),
                  selected_ids=case_ids, selected_pair_ids=[p["pair_id"] for p in selected_pairs],
                  selection=manifest["selection"],
                  note="Released captions are supplied inputs; no DIFFA caption generation is claimed")
    return manifest_path, samples, predictions, config, origin


def fraction(numerator, denominator):
    return numerator/denominator if denominator else None


def score_pair(pair, control, corrupted, nlp, detect_only=False):
    targets = set(pair["injected_noun_lemmas"])
    candidates = {x["lemma"] for x in corrupted["detections"]}
    flagged = {x["lemma"] for x in corrupted["detections"] if x["flagged"]}
    result = dict(pair_id=pair["pair_id"], injected_noun_lemmas=sorted(targets),
                  target_count=len(targets), candidate_target_count=len(targets & candidates),
                  flagged_target_count=len(targets & flagged),
                  control_candidate_count=len(control["detections"]),
                  control_flagged_count=sum(x["flagged"] for x in control["detections"]))
    if not detect_only:
        preserved = {g["lemma"] for g in nlp.candidates(control["original"])}
        shared = preserved & {g["lemma"] for g in nlp.candidates(corrupted["original"])} - targets
        result["control_original_noun_count"] = len(preserved)
        result["corrupted_shared_noun_count"] = len(shared)
        for version in ("cut", "healed"):
            result[version] = dict(
                target_mentions_removed=len(targets-mentions(corrupted[version], nlp)),
                control_nouns_retained=len(preserved & mentions(control[version], nlp)),
                corrupted_shared_nouns_retained=len(shared & mentions(corrupted[version], nlp)),
                control_unchanged=control[version] == control["original"],
                corrupted_empty=not bool(corrupted[version].strip()),
                control_empty=not bool(control[version].strip()))
    return result


def report(out, samples, manifest_path, config_hash, detect_only=False):
    from .nlp import PaperNLP
    out = Path(out)
    nlp = PaperNLP()
    pairs = read_json(Path(manifest_path).parent/"pairs.json")["pairs"]
    selected_ids = {s["id"] for s in samples}
    pairs = [p for p in pairs if p["control_id"] in selected_ids and p["corrupted_id"] in selected_ids]
    rows, records = [], {}
    for sample in samples:
        path = out/"clips"/f"{sample['id']}.json"
        if path.exists():
            record = read_json(path)
            if (record["config_sha256"] != config_hash or record["id"] != sample["id"]
                    or record["original"] != sample["brace"]["text"]):
                raise ValueError("Mixed BRACE DCH output records")
            records[sample["id"]] = record
    for pair in pairs:
        if pair["control_id"] in records and pair["corrupted_id"] in records:
            rows.append(score_pair(pair, records[pair["control_id"]], records[pair["corrupted_id"]],
                                   nlp, detect_only))
    total = lambda key: sum(r[key] for r in rows)
    target_count = total("target_count")
    summary = dict(benchmark="BRACE-Hallucination controlled caption intervention pilot",
                   expected_pairs=len(pairs), completed_pairs=len(rows),
                   completed_caption_cases=len(records), complete=len(rows) == len(pairs),
                   pairs_without_automatic_noun_target=sum(r["target_count"] == 0 for r in rows),
                   injected_noun_targets=target_count,
                   target_candidate_coverage=fraction(total("candidate_target_count"), target_count),
                   target_detection_recall=fraction(total("flagged_target_count"), target_count),
                   control_noun_flag_rate=fraction(total("control_flagged_count"), total("control_candidate_count")),
                   total_added_nfe=sum(r["total_added_nfe"] for r in records.values()),
                   elapsed_seconds=sum(r["elapsed_seconds"] for r in records.values()),
                   warning="Targets follow released synthetic pair labels, not independent listening. Removal is noun-mention disappearance, not complete factuality. New claims, negation, synonyms and fluency require separate evaluation. Controls are released human captions. This is not the official BRACE caption-ranking task.")
    if not detect_only:
        noun_count = total("control_original_noun_count")
        summary["stages"] = {}
        for version in ("cut", "healed"):
            stage_total = lambda key: sum(r[version][key] for r in rows)
            summary["stages"][version] = dict(
                target_mention_removal_rate=fraction(stage_total("target_mentions_removed"), target_count),
                control_noun_retention=fraction(stage_total("control_nouns_retained"), noun_count),
                corrupted_shared_noun_retention=fraction(stage_total("corrupted_shared_nouns_retained"), total("corrupted_shared_noun_count")),
                control_unchanged_rate=fraction(stage_total("control_unchanged"), len(rows)),
                corrupted_empty_captions=stage_total("corrupted_empty"),
                control_empty_captions=stage_total("control_empty"))
        summary["all_fixed_tokens_preserved"] = all(r["healing"]["fixed_tokens_preserved"] for r in records.values())
    write_json(out/"summary.json", summary)
    write_json(out/"pair_scores.json", rows)
    fields = ["pair_id", "role", "injected_nouns", "original", "flagged", "cut", "healed", "audio_file"]
    by_pair = {p["pair_id"]: p for p in pairs}
    with (out/"review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            if sample["id"] not in records:
                continue
            r, info = records[sample["id"]], sample["brace"]
            writer.writerow(dict(pair_id=info["pair_id"], role=info["role"],
                                 injected_nouns=" | ".join(by_pair[info["pair_id"]]["injected_noun_lemmas"]),
                                 original=r["original"], flagged=" | ".join(x["lemma"] for x in r["detections"] if x["flagged"]),
                                 cut=r.get("cut", ""), healed=r.get("healed", ""),
                                 audio_file=str(Path(manifest_path).parent/sample["audio_path"])))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download one released pair per distinct AudioCaps clip")
    prep.add_argument("--out", type=Path, default=Path("data/brace5"))
    prep.add_argument("--clips", type=int, default=5)
    prep.add_argument("--seed", type=int, default=SEED)
    scoring = sub.add_parser("report", help="Rebuild targeted scores from existing GPU outputs")
    scoring.add_argument("--out", type=Path, default=Path("runs/brace5"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.clips, args.seed)
    else:
        config = read_json(args.out/"config.json")
        path, samples, _, _, _ = load_inputs(config["inputs"]["manifest"])
        samples = [s for s in samples if s["id"] in config["inputs"]["selected_ids"]]
        report(args.out, samples, path, digest(args.out/"config.json"), config["detect_only"])


if __name__ == "__main__":
    main()
