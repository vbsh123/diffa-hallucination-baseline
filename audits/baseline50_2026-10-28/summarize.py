"""Summarize fixed assistant reference judgments; optionally replay the noun scorer.

This script does NOT judge audio or infer new semantic labels. It counts the
explicit decisions in decisions.json and checks their cohort/caption provenance.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_captions():
    captions = {}
    for line in (HERE / "captions.txt").read_text().splitlines():
        match = re.fullmatch(r"(\d+)/50 remaining clips: ([A-Za-z0-9_-]+): (.+)", line)
        if not match:
            raise ValueError(f"Malformed caption line: {line}")
        number, clip_id, caption = match.groups()
        if int(number) != len(captions)+1 or clip_id in captions:
            raise ValueError("Caption numbering or IDs are inconsistent")
        captions[clip_id] = caption
    if len(captions) != 50:
        raise ValueError("Expected all 50 user-provided captions")
    return captions


def outcome(row):
    if row["unreferenced"]:
        return "unreferenced_main_claim"
    if row["unclear"]:
        return "unclear_only"
    if row["supported"]:
        return "supported_main_claims_only"
    raise ValueError(f"Clip has no reviewed claims: {row['id']}")


def replay_lexical(captions, manifest, audit):
    import spacy
    sys.path.insert(0, str(ROOT))
    from baseline.score import aggregate, compare, extract

    nlp = spacy.load("en_core_web_sm")
    scores = []
    for row in manifest["samples"]:
        predicted = extract(captions[row["id"]], nlp)["terms"]
        reference = set().union(*(set(extract(t, nlp)["terms"])
                                  for t in row["reference_captions"] + row["labels"]))
        result = compare(predicted, reference)
        result["id"] = row["id"]
        scores.append(result)
    summary = aggregate(scores, 50)
    differences = {key: summary[key]-expected
                   for key, expected in audit["user_reported_original_metrics"].items()}
    report = dict(method="replay_of_existing_lexical_noun_proxy_no_model_inference",
                  note="Same outputs, same scorer. Not a new generation or intervention result.",
                  spacy=spacy.__version__, nlp_model_version=nlp.meta["version"],
                  captions_sha256=sha256(HERE / "captions.txt"), summary=summary,
                  differences_from_user_report=differences,
                  all_reported_metrics_match=all(abs(x) < 1e-12 for x in differences.values()),
                  clips=scores)
    write_json(HERE / "lexical_replay.json", report)
    return report["all_reported_metrics_match"]


def summarize(manifest_path=None, replay=False):
    audit = json.loads((HERE / "decisions.json").read_text())
    captions = read_captions()
    rows = audit["clips"]
    if len(rows) != 50 or [r["id"] for r in rows] != list(captions):
        raise ValueError("Audit and complete caption cohort do not match")
    if [r["n"] for r in rows] != list(range(1,51)):
        raise ValueError("Audit rows are not numbered 1 through 50")
    for row in rows:
        for key in ("supported", "unreferenced", "unclear"):
            if not isinstance(row[key], list) or not all(isinstance(x, str) and x for x in row[key]):
                raise ValueError(f"Invalid judgments: {row['id']} {key}")
        if row["source_mismatch_candidate"] and not row["unreferenced"]:
            raise ValueError("A source-mismatch candidate must have an unreferenced claim")

    manifest = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if sha256(manifest_path) != audit["cohort_manifest_sha256"]:
            raise ValueError("This audit belongs to a different manifest")
        manifest = json.loads(manifest_path.read_text())
        if [r["id"] for r in manifest["samples"]] != list(captions):
            raise ValueError("Manifest/caption IDs or order differ")
        selection = manifest["selection"]
        if selection["dataset"] != audit["dataset"] or selection["revision"] != audit["dataset_revision"]:
            raise ValueError("Dataset provenance differs")
    if replay and manifest is None:
        raise ValueError("Lexical replay needs --manifest for all reference annotations")

    counts = Counter(outcome(row) for row in rows)
    summary = dict(
        method=audit["method"], basis="Reference/metadata audit; no waveform listening",
        rater=audit["rater"], blind=False, held_out=False, clips=50,
        counts=dict(counts), caption_fractions={key: count/50 for key,count in counts.items()},
        captions_with_any_supported_main_claim=sum(bool(r["supported"]) for r in rows),
        source_mismatch_candidate_clips=[r["n"] for r in rows if r["source_mismatch_candidate"]],
        source_mismatch_candidate_count=sum(r["source_mismatch_candidate"] for r in rows),
        source_mismatch_candidate_fraction=sum(r["source_mismatch_candidate"] for r in rows)/50,
        annotation_conflict_clips=[r["n"] for r in rows if r.get("annotation_conflict")],
        unreferenced_claim_clips=[r["n"] for r in rows if r["unreferenced"]],
        unclear_only_clips=[r["n"] for r in rows if outcome(r) == "unclear_only"],
        supported_main_claims_only_clips=[r["n"] for r in rows if outcome(r) == "supported_main_claims_only"],
        cohort_manifest_sha256=audit["cohort_manifest_sha256"],
        reference_manifest_verified=manifest is not None,
        captions_sha256=sha256(HERE / "captions.txt"),
        decisions_sha256=sha256(HERE / "decisions.json"),
        warning="These are exploratory annotation-support judgments, not measured acoustic hallucination rates. Same outputs re-evaluated; no intervention improvement is demonstrated.",
    )
    if replay:
        summary["lexical_replay_matches_all_user_reported_metrics"] = replay_lexical(captions, manifest, audit)
    write_json(HERE / "summary.json", summary)

    fields = ["n", "id", "outcome", "caption", "supported", "unreferenced", "unclear",
              "source_mismatch_candidate", "reference_evidence", "annotation_conflict",
              "qualitative_omissions", "excluded_explanations", "other_issues"]
    with (HERE / "review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, "") for key in fields}
            output.update(outcome=outcome(row), caption=captions[row["id"]])
            for key, value in output.items():
                if isinstance(value, list):
                    output[key] = " | ".join(value)
            writer.writerow(output)

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    report = [
        "# Audit of the user's 50 baseline captions\n",
        "This is an assistant semantic review against the five saved reference captions, sound labels and metadata. **No waveform was listened to.** Judgments are non-blind and exploratory. They do not establish physical absence of unannotated sounds.\n",
        "## Caption-level results\n",
        "| Outcome | Clips | Fraction |",
        "| --- | ---: | ---: |",
    ]
    labels = {"supported_main_claims_only": "All reviewed main claims reference-supported",
              "unreferenced_main_claim": "At least one unreferenced main sound/source claim",
              "unclear_only": "No unreferenced claim, but at least one unclear claim"}
    for key, label in labels.items():
        report.append(f"| {label} | {counts[key]}/50 | {counts[key]/50:.0%} |")
    report.extend([
        "\nThese three rows partition all 50 captions. 'Supported' is limited to the reviewed main claims; it does not mean complete, fluent, correctly ordered or hallucination-free. Neither unreferenced nor unclear means verified false.\n",
        f"**Source-identity mismatch candidates:** {summary['source_mismatch_candidate_count']}/50 ({summary['source_mismatch_candidate_fraction']:.0%}), a subset of the unreferenced group. Clip numbers: " + ", ".join(map(str, summary["source_mismatch_candidate_clips"])) + ".\n",
        "The source candidates include sheep/goat, power tool/steam, water/bicycle-gravel, train/ocean waves, car/basketball-whoosh, vacuum/power tool, sewing machine/typewriter, door/telephone, machinery/insects, horse/insects, dog/horse and train/bus. Some distinctions are much weaker acoustically than others; these are listening priorities, not a claim that all twelve are wrong.\n",
        "## Why this differs from the noun proxy\n",
        "The original reported micro noun fraction was 71.96%, macro 66.73%, caption flag rate 100%, and macro reference noun coverage 22.61%. Generic nouns, explanations of typical sources, hypothetical alternatives and negated mentions inflate that proxy. This audit scores semantic source/event assertions instead; the difference is a change of evaluation, not a model improvement.\n",
        "Examples: clip 3's man and birds have reference support even though the noun scorer flags atmosphere/environment. Clip 27 explains common siren sources without asserting every emergency vehicle is present. Clip 33's sewing-machine attribution conflicts with typewriter references and is a source-error candidate, not just verbose style.\n",
        "The main claim review does not exhaustively score acoustic attributes, timing/order, counts, negation or all possible scene inferences. The no-additional-sounds assertion in clip 33 conflicts with the referenced bell; clip 34's four-second recording claim disagrees with the 9.50-second file duration. These are separate observations rather than additions to the headline sound-presence tally. Hypothetical scene/source explanations are recorded, not blanket-approved.\n",
        "## Annotation/coverage concerns\n",
        "Clip 1 has sheep labels but firecracker/crowd captions. Clip 45 mixes a thunder label with divergent engine/static/hum/rain captions; its engine attribution is unclear, not a definite failure. Some supported captions omit important references: clip 13 omits the passing vehicle, clip 16 the goat, and clip 46 speech. No exhaustive semantic coverage percentage is claimed.\n",
        "## Every clip\n",
        "Full source captions and evidence notes are in review.csv and decisions.json; captions.txt preserves the user's quoted outputs. U = at least one unreferenced main claim; ? = unclear-only; S = all reviewed main claims supported.\n",
        "| # | ID | Outcome | Unreferenced / unclear main claims |",
        "| ---: | --- | :---: | --- |",
    ])
    markers = {"unreferenced_main_claim": "U", "unclear_only": "?", "supported_main_claims_only": "S"}
    for row in rows:
        claims = "; ".join(row["unreferenced"] + ["unclear: " + s for s in row["unclear"]]) or "—"
        report.append(f"| {row['n']} | {row['id']} | {markers[outcome(row)]} | {cell(claims)} |")
    report.extend([
        "\n## Reproduce the summaries\n",
        "From the repository root, with the original cohort prepared:\n",
        "```bash\npython3 audits/baseline50_2026-10-28/summarize.py --manifest data/audiocaps50/manifest.json\n# Optional: replay all original noun metrics without DIFFA inference:\n.venv/bin/python audits/baseline50_2026-10-28/summarize.py --manifest data/audiocaps50/manifest.json --replay-lexical\n```\n",
        "The script validates all 50 IDs and the source-manifest hash, then counts the explicit judgments; it does not generate new semantic judgments or assess audio. The optional lexical replay re-extracts nouns from the quoted captions using the existing pinned scorer. Reference captions remain in the cohort manifest and are not redistributed here; evidence notes paraphrase them.\n",
    ])
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({key: summary[key] for key in ("clips", "counts", "source_mismatch_candidate_count",
                                                   "source_mismatch_candidate_fraction", "annotation_conflict_clips")}, indent=2))
    print(f"Saved report and per-clip CSV under {HERE}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--replay-lexical", action="store_true")
    args = parser.parse_args()
    summarize(args.manifest, args.replay_lexical)


if __name__ == "__main__":
    main()
