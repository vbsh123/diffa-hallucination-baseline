# Run the supplied DCH implementation on DIFFA audio

The core method files are copied **byte for byte from your `DCH-main.zip`** into [`dch/reference/`](dch/reference/). The ZIP identifies commit `db099487f853ba2947fad6ffd032ba55690b9774`; [`SOURCE.json`](dch/reference/SOURCE.json) records archive/file hashes and the original license is preserved. We use the supplied code rather than reconstructing its grammar from the manuscript.

The surrounding adapter replaces LLaDA-V's image-conditioned model interface with DIFFA's audio-conditioned interface. No extra LLM, API, training, prompted full rewrite, or fresh caption generation is required. Saved baseline captions are the immutable inputs.

## Run on your existing Vast instance

Run directly in your terminal, using the existing baseline environment and `runs/baseline50/`:

```bash
cd /workspace/diffa-hallucination-baseline
git pull
export HF_HOME=/workspace/hf-cache
bash setup_dch.sh
set -o pipefail

# Two saved captions through Detect, Cut and Heal:
SAMPLES=2 bash run_dch.sh 2>&1 | tee dch2.log

# If that succeeds, process all 50 originals:
bash run_dch.sh 2>&1 | tee dch50.log
```

Results: `runs/dch2/` and `runs/dch50/`. Repeating the same command resumes identical unfinished runs. Changed code, settings, NLTK resources or input captions require a new output directory. Baseline Python files and original requirements are unchanged, so your completed captions are preserved and are not regenerated.

`setup_dch.sh` adds NLTK and its standard resources in the existing environment. If the baseline environment does not exist, run `setup.sh` first. No model weights are downloaded by DCH setup; the run reuses the pinned DIFFA/LLaDA/Whisper cache.

Detector-only mode is also available:

```bash
SAMPLES=2 MODE=detect-only bash run_dch.sh
# Output: runs/dch_detect2/
```

Options: `BASELINE_RUN` (default `runs/baseline50`), `SAMPLES` (50), `RUN_NAME` (output directory), `MODE` (`full` or `detect-only`), `AUDIO_VENV`, `DIFFA_ROOT`, `DEVICE`, `NLTK_DATA`, and `MANIFEST` for a relocated identical-hash cohort. The underlying CLI is `python -m dch.run --help`. References and labels are never supplied to detection or repair.

## What is reused unchanged

| Source component | Use in the audio adaptation |
| --- | --- |
| `candidates.py` | Original NLTK tokenization/POS tags, minimum noun length, WordNet lemmas and physical-sense test, stoplists and token-span matching. No LLM/event-claim extractor. |
| `detect.py` | Actual `yes_no()` function: two masked positions, one forward per candidate, first-position Yes/No log-odds, original case/space variants and zero threshold. |
| `cut.py` | Actual regular-expression deletion grammar and surface-form processing. Candidate/flag ordering matches the supplied pipeline. |
| `heal.py` | Actual space-delimited word alignment, seam windows of two tokens per side, punctuation exclusions, banned word forms, invalid-word retries and new-object checks. |
| `model.py::infill` | Actual native fixed-canvas sampler, committing one masked position per forward. Its confidence is taken from the **pre-ban** distribution; we do not renormalize after banning tokens. |

The detector question is translated from `Is there a {o} in the image?` to `Is there a sound of {o} in the audio?`, preserving the supplied plural-question rule and newline instruction. Exact token variants are validated against the pinned LLaDA tokenizer. We record probabilities and total Yes/No mass for diagnostics, but flag using the supplied function's score, without an extra threshold or verbal-answer parsing.

The model adapter uses DIFFA's original voice-assistant chat template and audio insertion offset; runtime checks compare its prompt to baseline preparation. Audio features are computed once per clip and reused for isolated queries. Repair receives the same captioning instruction, audio features and both text sides. **Only seam positions may change**; a runtime assertion checks every other token.

## Explicit model/modality changes and inherited limitations

- DIFFA uses bfloat16 and its native EOS token `126081`, whereas the image wrapper uses its model's end-of-turn token. The same supplied canvas/infill functions operate through the adapter using DIFFA's EOS. No tail regeneration is substituted for local infill.
- Default candidates use the **literal supplied stoplists**. This can select input-meta words such as `audio` or `clip`, because the visual lists remove image/photo/picture but do not include their audio counterparts. An explicitly optional ablation adds just `audio`, `recording`, `clip`: `AUDIO_META_STOPLIST=1 RUN_NAME=runs/dch_audio_meta50 bash run_dch.sh`. It is recorded and off by default.
- Word-level physical nouns remain the scope. It may ask about `machine` and separately `sewing`, rather than the compound `sewing machine`. NLTK/POS ambiguities and WordNet's first-three-sense behavior are inherited. There is no silent phrase-level, negation, action, temporal or semantic matching extension.
- The deletion grammar protects clauses/sentences as the supplied regex code does; it does not guarantee retention of every correct fact. Residual rejected words are recorded. No detachable deletion means no repair.
- The supplied healer retries invalid words up to twice and retries rejected new objects once. It does **not** guarantee the final retry is clean, and we do not add an undisclosed rollback/full-rewrite fallback. Final nonword spans and residual original flags are recorded for review.
- The supplied token-span helper assumes concatenated individual-token decodes reconstruct the text. We fail explicitly on unalignable input instead of guessing coordinates. All 50 supplied English captions were checked with the pinned tokenizer.

The supplied executable code resolves details that were ambiguous in the PDF: its adjective-series test is part of the deletion grammar, its exact ban forms and function-word list are fixed, and its confidence ranking uses the original distribution. The code in the ZIP is the reference for these details. This is the supplied algorithm on a different audio model, not a reproduction of the visual benchmark's numerical results.

## Outputs and evaluation

- `config.json`: input hashes, supplied DCH source hashes, actual NLTK resource hashes, model revisions, question, threshold and runtime versions.
- `clips/<id>.json`: original/cut/healed text, candidate spans, all detector scores, flags, residuals, cut/repair token IDs, mutable positions, frozen-token checks, new-object verification scores, retry details and actual added forward counts.
- `summary.json`, `paired_proxy_scores.json`: completion/edit statistics and the **unchanged legacy noun-overlap proxy** for all three versions. These are not waveform-verified hallucination rates.
- `review.csv`: all three versions with references, audio paths and a review-notes column preserved on re-reporting.
- `judge_inputs/session_*.jsonl`: three balanced sessions, each showing one anonymous version per clip; across sessions every version appears once. Give each judge one session. `private_judge_key.json` is outside that directory and must not be supplied to the judge.

Use `prompts/dch_judge.txt` for independent acoustic/reference assessment, retained correct information and readability. A reference-only judge must label its evidence basis; a detector rejection is not itself ground truth. The raw baseline/Cut-only/Heal outputs are retained even if the proxy improves or the repair is awkward.

This runs the core original/Cut-only/DCH comparison. It does not implement every research control in the paper (random/oracle flags, length-matched truncation, external rewrites and attribute extensions). The existing 50 clips are development examples; final claims need a frozen evaluation set.

## Validation and remaining execution check

All 23 CPU tests pass, including tests with the actual pinned tokenizer and a tiny randomly initialized upstream LLaDA model. Checks include source-file hash identity, raw logits versus upstream audio embedding insertion, exact single-forward detection, pooled Yes/No scores, frozen-token preservation, and output/NFE parity with the unmodified supplied healer. Native parity tests use synthetic inputs and do not demonstrate improved acoustic accuracy.

The new full-checkpoint DCH path still needs its first GPU smoke run on Vast because local GPU access is unavailable. This is the remaining execution check, not a missing model interface or a rewrite workaround.

```bash
source .venv/bin/activate
export NLTK_DATA="${NLTK_DATA:-$PWD/.cache/nltk_data}"
python -m unittest discover -s tests -v
# Optional native CPU checks; requires torch and the cached pinned tokenizer:
DCH_NATIVE_TEST=1 python -m unittest discover -s tests -p test_dch_native.py -v
```
