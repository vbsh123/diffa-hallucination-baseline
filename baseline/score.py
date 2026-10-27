"""An auditable noun-overlap proxy, not a factual hallucination oracle."""
import argparse
import csv
import importlib.metadata
from pathlib import Path
import statistics

from .common import digest, load_manifest, read_json, write_json

NLP_MODEL = "en_core_web_sm"
NLP_VERSION = "3.8.0"
# Meta-description words are excluded on BOTH sides. No sound ontology/synonym map.
META_NOUNS = frozenset({"audio", "clip", "recording", "sound", "background", "foreground"})
NEGATION_WORDS = frozenset({"no", "not", "never", "without", "neither", "nor"})


def extract(text, nlp):
    doc = nlp(text)
    mentions = [dict(surface=t.text, lemma=t.lemma_.lower(), start=t.idx, end=t.idx+len(t.text))
                for t in doc if t.pos_ in {"NOUN", "PROPN"}
                and t.lemma_.lower() not in META_NOUNS and t.is_alpha]
    return dict(terms=sorted({x["lemma"] for x in mentions}), mentions=mentions,
                has_negation=any(t.lower_ in NEGATION_WORDS or t.dep_ == "neg" for t in doc))


def compare(predicted, reference):
    p, r = set(predicted), set(reference)
    unsupported, supported = p-r, p&r
    return dict(predicted_terms=sorted(p), reference_terms=sorted(r),
                supported_terms=sorted(supported), unsupported_terms=sorted(unsupported),
                unsupported_fraction=len(unsupported)/len(p) if p else None,
                reference_coverage=len(supported)/len(r) if r else None)


def mean_or_none(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def aggregate(rows, expected):
    predicted = sum(len(r["predicted_terms"]) for r in rows)
    reference = sum(len(r["reference_terms"]) for r in rows)
    unsupported = sum(len(r["unsupported_terms"]) for r in rows)
    supported = sum(len(r["supported_terms"]) for r in rows)
    return dict(expected_clips=expected, completed_clips=len(rows),
                complete=len(rows) == expected,
                noun_bearing_captions=sum(bool(r["predicted_terms"]) for r in rows),
                empty_noun_captions=sum(not r["predicted_terms"] for r in rows),
                empty_reference_noun_sets=sum(not r["reference_terms"] for r in rows),
                predicted_noun_count=predicted, unsupported_noun_count=unsupported,
                reference_noun_count=reference, supported_noun_count=supported,
                unsupported_noun_fraction_micro=unsupported/predicted if predicted else None,
                unsupported_noun_fraction_macro=mean_or_none(r["unsupported_fraction"] for r in rows),
                caption_flag_rate=mean_or_none(bool(r["unsupported_terms"]) for r in rows),
                reference_noun_coverage_micro=supported/reference if reference else None,
                reference_noun_coverage_macro=mean_or_none(r["reference_coverage"] for r in rows))


def score_run(run_dir, manifest_path=None, nlp=None):
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    config = read_json(config_path)
    config_hash = digest(config_path)
    manifest_path = Path(manifest_path or config["manifest"]).resolve()
    if digest(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Manifest changed since inference")
    manifest = load_manifest(manifest_path)
    if nlp is None:
        import spacy
        nlp = spacy.load(NLP_MODEL)
        if nlp.meta["version"] != NLP_VERSION:
            raise ValueError(f"Expected {NLP_MODEL} {NLP_VERSION}; run setup.sh")
    rows = []
    expected_ids = {s["id"] for s in manifest["samples"]}
    prediction_files = list((run_dir / "predictions").glob("*.json"))
    if any(p.stem not in expected_ids for p in prediction_files):
        raise ValueError("Prediction directory contains clips outside this cohort")
    for sample in manifest["samples"]:
        path = run_dir / "predictions" / f"{sample['id']}.json"
        if not path.exists():
            continue
        prediction = read_json(path)
        if prediction["id"] != sample["id"] or prediction["run_config_sha256"] != config_hash:
            raise ValueError(f"Invalid or mixed-run prediction: {path}")
        pred = extract(prediction["text"], nlp)
        references = [extract(text, nlp) for text in sample["reference_captions"] + sample["labels"]]
        ref_terms = set().union(*(set(ref["terms"]) for ref in references))
        row = compare(pred["terms"], ref_terms)
        row.update(id=sample["id"], caption=prediction["text"], audio_path=sample["audio_path"],
                   reference_captions=sample["reference_captions"], labels=sample["labels"],
                   predicted_mentions=pred["mentions"], prediction_has_negation=pred["has_negation"],
                   reference_has_negation=any(r["has_negation"] for r in references),
                   eos_found=prediction["eos_found"], capacity_exhausted=prediction["capacity_exhausted"],
                   unresolved_visible_masks=prediction["unresolved_visible_masks"],
                   actual_nfe=prediction["actual_nfe"], elapsed_seconds=prediction["elapsed_seconds"])
        rows.append(row)
    summary = aggregate(rows, len(manifest["samples"]))
    summary.update(metric="echo_style_lexical_noun_proxy_v1", unit="fractions in [0,1]",
                   warning="Unmatched nouns are audit candidates, not verified hallucinations.",
                   manifest_sha256=config["manifest_sha256"], run_config_sha256=config_hash,
                   spacy=importlib.metadata.version("spacy"),
                   nlp_model=NLP_MODEL, nlp_model_version=nlp.meta["version"],
                   scorer_sha256=digest(Path(__file__)), excluded_meta_nouns=sorted(META_NOUNS),
                   captions_with_negation=sum(r["prediction_has_negation"] for r in rows),
                   captions_without_eos=sum(not r["eos_found"] for r in rows),
                   captions_with_unresolved_masks=sum(r["unresolved_visible_masks"] > 0 for r in rows),
                   mean_actual_nfe=mean_or_none(r["actual_nfe"] for r in rows),
                   mean_elapsed_seconds=mean_or_none(r["elapsed_seconds"] for r in rows))
    write_json(run_dir / "scores.json", dict(summary=summary, clips=rows))
    fields = ["id", "audio_file", "caption", "references", "labels", "unsupported_terms",
              "supported_terms", "unsupported_fraction", "reference_coverage", "has_negation",
              "scene_or_source_hallucination", "audible_event_hallucination", "review_notes"]
    # Preserve listening judgments on re-score/resume, keyed by the exact caption.
    reviews = {}
    if (run_dir / "review.csv").exists():
        with (run_dir / "review.csv").open(newline="") as stream:
            reviews = {(r["id"], r["caption"]): r for r in csv.DictReader(stream)}
    manual_fields = ("scene_or_source_hallucination", "audible_event_hallucination", "review_notes")
    review_temporary = run_dir / "review.csv.tmp"
    with review_temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(id=r["id"], audio_file=str(manifest_path.parent / r["audio_path"]),
                                 caption=r["caption"], references=" | ".join(r["reference_captions"]),
                                 labels=" | ".join(r["labels"]),
                                 unsupported_terms=" | ".join(r["unsupported_terms"]),
                                 supported_terms=" | ".join(r["supported_terms"]),
                                 unsupported_fraction=r["unsupported_fraction"],
                                 reference_coverage=r["reference_coverage"],
                                 has_negation=r["prediction_has_negation"],
                                 **{key: reviews.get((r["id"], r["caption"]), {}).get(key, "")
                                    for key in manual_fields}))
    review_temporary.replace(run_dir / "review.csv")
    print(f"Scored {len(rows)}/{len(manifest['samples'])}: {run_dir / 'scores.json'}", flush=True)
    for key in ("unsupported_noun_fraction_micro", "unsupported_noun_fraction_macro",
                "caption_flag_rate", "reference_noun_coverage_macro"):
        print(f"  {key}: {summary[key]}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/baseline50"))
    parser.add_argument("--manifest", type=Path, help="Relocated manifest; its hash must match")
    args = parser.parse_args()
    score_run(args.run, args.manifest)


if __name__ == "__main__":
    main()
