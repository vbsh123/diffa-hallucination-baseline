# Five-clip BRACE-Hallucination smoke test

Run on the existing Vast machine, in the same repository/environment as the baseline:

```bash
cd /workspace/diffa-hallucination-baseline
git pull
export HF_HOME=/workspace/hf-cache
bash setup_dch.sh
set -o pipefail
bash run_brace.sh 2>&1 | tee brace5.log
```

This downloads **five individual AudioCaps WAV files** from the official BRACE audio release, selects one released caption pair per distinct clip, and applies the unchanged DCH helpers to **both sides: ten caption cases total**. Full Detect–Cut–Heal is the default. It supplies the released captions directly; it does not generate new DIFFA baseline captions or need `runs/baseline50/`. It reuses the model cache. The default repair instruction remains `Describe the audio.`.

Selection is seeded before inspecting model predictions. Every selected pair is retained even if the candidate extractor misses the injected noun. Complete clips are mixed to mono and resampled to 16 kHz; clips longer than 30 seconds are excluded, never cropped. The default AudioCaps clips are about ten seconds. Runtime downloads are pinned to metadata commit `709b44d24ec79d3296b45790cefb238292cbc3d5` and audio dataset revision `60729676c42a9430b5134cbf2ff81289b35730d6`. Metadata and all materialized WAVs are hash-checked.

The cohort is saved under `data/brace5/`; outputs under `runs/brace5/`. Repeating the command resumes an unchanged experiment. Changed code, inputs or configuration requires a new output directory. Raw per-caption detector scores and cut/healed texts are saved after each case.

Other small runs:

```bash
CLIPS=2 bash run_brace.sh                 # four caption cases
CLIPS=5 MODE=detect-only bash run_brace.sh
CLIPS=10 bash run_brace.sh                # twenty caption cases
PREPARE_ONLY=1 bash run_brace.sh          # CPU data preparation only
```

Options: `CLIPS` (5), `BRACE_DATA`, `RUN_NAME`, `MODE`, `AUDIO_VENV`, `DIFFA_ROOT`, `DEVICE`, `NLTK_DATA`. Keep `HF_HOME` consistent with the earlier model run. Run `setup.sh` first only if the original baseline environment/source checkout has not been installed.

## Reading the results

`runs/brace5/summary.json` reports:

- `target_candidate_coverage`: fraction of injected noun targets found by the original DCH candidate extractor. Missed/filtered targets remain in the denominator.
- `target_detection_recall`: fraction of all injected noun targets rejected by the detector.
- `control_noun_flag_rate`: fraction of candidate nouns rejected in released original human captions.
- `stages.cut` / `stages.healed`: injected target mention removal, original-control noun retention, retention of supported nouns shared by both input sides, unchanged-control rate, and empty-caption counts. Deleting everything must not look like an unqualified success.

`pair_scores.json` gives per-pair counters. `review.csv` shows both input sides, flags, Cut/Heal results, target nouns and local audio paths. `clips/*.json` retains the complete native-forward diagnostics. Scores can be rebuilt without a GPU:

```bash
source .venv/bin/activate
export NLTK_DATA="$PWD/.cache/nltk_data"
python -m dch.brace report --out runs/brace5
```

The scoring code derives changed-word noun targets from the supplied human/corrupted pair, using WordNet noun senses or full-caption noun tags. This evaluation target extraction is separate from the unchanged DCH candidate filter: an NLTK tagger mistake must not erase a known injected noun from the denominator. Punctuation-only changes, replacements already present in the original caption and changes without a distinct identifiable noun are explicitly unscorable; their count is reported, not silently filtered out. Full token edits remain in `data/brace5/pairs.json`.

These are **targeted synthetic-error diagnostics**, not an overall acoustic hallucination rate or the official BRACE caption-ranking protocol. Disappearance of a noun is a mention-removal proxy; it does not resolve synonyms, negation, unrelated new claims, grammar or factuality of an arbitrary healed caption. Pair labels are taken from BRACE and are not independent listening judgments. The original benchmark’s hallucination pairs use LLM noun substitutions, while its Main track uses human preference annotations; do not describe every injected absence as human-verified.

This smoke test validates the mechanism on supplied errors. Natural DIFFA outputs and a separate held-out cohort remain necessary for later effectiveness claims. References, target nouns and human/corrupted labels are never inserted into detector or repair prompts. The official [paper](https://arxiv.org/html/2512.10403v1), [metadata/code](https://github.com/HychTus/BRACE_Evaluation) and [audio release](https://huggingface.co/datasets/gtysssp/audio_benchmarks) describe the source data.

Local validation: five BRACE-specific CPU tests and nine existing DCH core regression tests pass. Real five-clip download, mono/resampling preparation, input validation, shell preparation mode and unchanged-cohort resume were exercised. All five selected source clips are ten seconds. Full-checkpoint inference for this pilot still needs the Vast GPU; no model-effectiveness result is claimed from CPU fixtures.
