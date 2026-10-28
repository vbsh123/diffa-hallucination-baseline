# DIFFA hallucination baseline

Measure untouched **DIFFA-2 audio captioning** on a fixed AudioCaps/ECHO sample before implementing Detect–Cut–Heal. The first experiment asks: **which source or event nouns in the generated description lack support in the available captions and sound labels?** Listen to the flagged examples before calling them hallucinations.

## Run

Use Linux, Python 3.10–3.12 and one CUDA GPU with bfloat16 support. A 48-GB GPU is a reasonable starting point; peak VRAM has not yet been measured. Expect multi-GB model downloads and a roughly 0.6-GB dataset download. Setup installs CUDA 12.4 wheels; the host needs a compatible NVIDIA driver. No training is performed.

```bash
git clone https://github.com/vbsh123/diffa-hallucination-baseline.git
cd diffa-hallucination-baseline
bash setup.sh

# First check loading, generation and reporting on two clips:
SAMPLES=2 bash run.sh

# Then run the fixed 50-clip pilot:
bash run.sh
```

`setup.sh` installs packages and fetches source. `run.sh` downloads the dataset and model weights as needed, materializes complete audio clips, generates captions, and scores them. Hugging Face's cache is reused. Raw captions are written after every clip, so a failed run can resume with the same command. The first model load can take a while. Dependencies are pinned; deterministic sampling and greedy decoding do not guarantee identical GPU numerics across machines.

If reusing an environment, set `AUDIO_VENV=/absolute/path/to/.venv` for **both** scripts. Setup installs the pinned versions into that environment. `SKIP_INSTALL=1 bash setup.sh` skips installation if those versions are already present. `DIFFA_ROOT=/path/to/DIFFA` accepts an existing clean checkout at the pinned commit; other revisions or modified checkouts are rejected. A separate default checkout is created under `external/DIFFA`.

Options for `run.sh`: `SAMPLES`, `SEED`, `DATA_DIR`, `RUN_NAME` (output directory), `DEVICE`, `AUDIO_VENV`, `DIFFA_ROOT`, `INSTRUCTION` (optional caption prompt). Use new data/run directories when changing the seed or protocol. Do not mix runs. On an unchanged run, existing predictions are reused; changed code, dependencies, model settings or input manifest require a new output directory.

### Concise sound-description comparison

An initial user-provided 50-clip Vast log with the default prompt includes long explanations of source categories, possible settings and atmosphere. Those words inflate the noun-overlap proxy. For example, a caption explaining which emergency vehicles commonly use sirens does not necessarily assert that all those vehicles are present, but the current scorer counts their nouns anyway. The same log also contains source-confusion candidates: a clip whose reference captions describe a typewriter is described as a sewing machine. These cases need listening; the overall unmatched-noun fraction cannot separate them.

Re-scoring the quoted caption for `YZ_smJ66Tb3c` against the selected references matches `bird` and `man` but flags `atmosphere` and `environment`: a 0.5 unsupported fraction despite reference support for the two main sources. This is a lexical diagnostic of that quoted caption, not a verified acoustic hallucination judgment. Likewise, the scorer counts `speech` in the sewing-machine caption's negated phrase “contains no speech.” Do not interpret the reported 0.7196 overall unmatched-noun fraction as a verified hallucination rate.

To request short descriptions of audible sources and events, without conditioning the answer on the scorer's choice of nouns:

```bash
INSTRUCTION="$(cat prompts/concise.txt)" \
RUN_NAME=runs/concise50 bash run.sh
```

The prompt in `prompts/concise.txt` requests one or two short sentences, includes clearly audible background sounds, and discourages general explanations, hypothetical settings and duration estimates. It is an optional prompt baseline; the original default and saved predictions are preserved. The instruction is recorded in the new run's config. Prompting for concision and avoiding guesses can itself change error rates, so use a fixed prompt when subsequently comparing any intervention. Compare coverage and manually verified errors as well as the noun proxy: a lower score from saying less is not sufficient evidence of better grounding. The current scorer still does not validate verbs, temporal claims or whether a source noun is asserted versus mentioned as an example.

### Describing events in order

To test whether chronological descriptions help inspection, run the same cohort with a separate instruction and output directory:

```bash
INSTRUCTION='Describe the audio, reporting audible events in the order they occur. If sounds overlap or continue throughout, say so.' \
RUN_NAME=runs/ordered50 bash run.sh
```

The default run continues to use `Describe the audio.`. The exact instruction is saved in each run's `config.json`. The ordered run uses the same clips and decoding settings; simultaneous sounds should not be forced into a false sequence. Check missing-EOS flags, since a more detailed answer can exhaust the 128-token answer capacity.

This prompt does **not** add temporal evaluation. The current scorer compares sets of noun lemmas against the union of the reference captions and labels. A matched noun validates only lexical support: “a dog barks” and “a dog is silent” can both match “dog.” It ignores verbs, argument roles and event order; “a bell then a dog” and “a dog then a bell” can score identically. No sentence embeddings, semantic entailment model or audio judge is used. Human reference captions also lack a complete verified event timeline. Review chronological claims by listening; a future temporal metric needs verified event intervals or before/after/overlap annotations. Keep source/scene and audible-event judgments separate.

## Run on Vast.ai

Use an existing suitable instance or start a PyTorch instance with Python 3.10–3.12 and CUDA 12.4 support. For a first attempt, budget one GPU with 48 GB VRAM, 64 GB system RAM and 100 GB disk. These are conservative starting allocations, not measured minimum requirements. The GPU must support bfloat16. Connect through the instance's SSH command or open a Jupyter terminal; see [Vast's CUDA setup guide](https://github.com/vast-ai/docs/blob/main/cuda.mdx).

In the instance terminal (the standard root container):

```bash
apt-get update
apt-get install -y git python3-venv libsndfile1 tmux
mkdir -p /workspace
cd /workspace
git clone https://github.com/vbsh123/diffa-hallucination-baseline.git
cd diffa-hallucination-baseline
tmux new -s diffa-baseline
```

Inside the new tmux session, run:

```bash
set -euo pipefail
cd /workspace/diffa-hallucination-baseline
export HF_HOME=/workspace/hf-cache
python3 --version  # Must be 3.10, 3.11 or 3.12.
nvidia-smi
bash setup.sh 2>&1 | tee setup.log

.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable in this container'
assert torch.cuda.is_bf16_supported(), 'Choose a GPU with bfloat16 support'
print(torch.cuda.get_device_name(0))
PY

SAMPLES=2 bash run.sh 2>&1 | tee smoke2.log
cat runs/baseline2/scores.json
bash run.sh 2>&1 | tee baseline50.log
```

`set -o pipefail` stops the sequence if setup or the smoke run fails, even with `tee`. Inspect `runs/baseline2/predictions/` before treating any quality result as meaningful. Detach with **Ctrl-b, then d**; reconnect with `tmux attach -t diffa-baseline`. In a new tmux session, re-export `HF_HOME=/workspace/hf-cache` before resuming. Repeat `bash run.sh` to resume the 50-clip run, keeping code/settings unchanged. All model/dataset cache downloads use this `HF_HOME`; no Hugging Face token is normally needed for these public repositories.

Results are in `runs/baseline50/scores.json`, `review.csv`, and `predictions/`. Download those plus `data/audiocaps50/` for listening and re-scoring. For example, create an archive and download it through Jupyter's file browser:

```bash
tar -czf /workspace/diffa-baseline50-results.tar.gz runs/baseline50 data/audiocaps50 baseline50.log
```

`tmux` survives terminal disconnects while the instance keeps running. A `/workspace` directory alone does not make files survive destruction of the instance; download your results or use a [Vast volume](https://github.com/vast-ai/docs/blob/main/guides/instances/storage/volumes.mdx) for longer-lived storage. Full DIFFA inference still needs its first GPU smoke test; the CPU validation below does not establish GPU compatibility or peak memory use.

## What is measured

The dataset is [`kuanhuggingface/audiocaps_hallucination`](https://huggingface.co/datasets/kuanhuggingface/audiocaps_hallucination), the AudioCaps sound-caption split used for [ECHO](https://arxiv.org/abs/2406.08402). It contains 957 test clips, sound labels and five human reference captions per clip. We select 50 using a seeded shuffle, without selecting on model output or negative labels. Complete clips are converted to mono 16-kHz float WAVs. Unreadable clips and clips longer than the encoder's native 30-second window are excluded and logged; clips are never cropped. The manifest saves selected IDs, source row indices, audio hashes, captions and labels. Reference annotations are used **only after generation**, never in the model prompt.

The prompt is `Describe the audio.` with the upstream chat template. Generation calls the **unchanged official factor-based parallel decoder**: 128 answer slots, 32-token blocks, 128 nominal steps, factor 1, temperature 0. The actual number of forward evaluations is saved per clip. Text ends at the first EOS; no forced caption length, attention modification, pruning, or correction is applied. We retain all answer IDs and flag missing EOS or unresolved masks.

The first scorer is an **ECHO-style lexical noun proxy**, not a bit-for-bit reproduction of ECHO or a factual hallucination detector. spaCy extracts lemmatized nouns/proper nouns from the generated caption. The reference set is the union of nouns from *all* reference captions and labels. The generic meta-nouns `audio`, `clip`, `recording`, `sound`, `background`, and `foreground` are excluded on both sides. Sets are per clip; repeated words count once. There is no synonym matching or sound ontology normalization.

Let `P` be predicted nouns, `R` reference nouns, `U = P − R`, and `S = P ∩ R`:

| Output | Definition | Interpretation |
| --- | --- | --- |
| `unsupported_noun_fraction_micro` | sum of \|U\| / sum of \|P\| | Proportion of predicted noun types flagged across clips |
| `unsupported_noun_fraction_macro` | mean of \|U\|/\|P\| over nonempty P | Average per-caption unsupported fraction |
| `caption_flag_rate` | fraction of completed captions with nonempty U | Captions requiring a closer look |
| `reference_noun_coverage_macro` | mean of \|S\|/\|R\| over nonempty R | Whether fewer flags merely reflect saying less |
| `reference_noun_coverage_micro` | sum of \|S\| / sum of \|R\| | Coverage weighted by reference noun counts |

Undefined ratios are JSON `null`, never negative sentinels or NaN. Empty noun sets are counted explicitly. A caption with no nouns is unflagged for the caption flag rate, so always read that rate alongside empty captions and coverage. Missing predictions are excluded from metric denominators and reported as incomplete; they are never counted as successes.

**Limits:** an unmatched noun can be a valid synonym, an unannotated audible event, a negated mention, or an unsupported claim. Negation is flagged but not semantically resolved. Verbs such as “barking,” adjectives, event relationships, counts, and timing are not reliably assessed by this noun scorer. Single-word matching also loses compound meaning. Reference coverage measures annotation overlap, not complete acoustic coverage. AudioCaps is named in DIFFA's training recipe; this is a measurement pilot, not proof of unseen-data generalization.

## Inspect results

Each run creates:

- `config.json`: dataset/cohort hash, source hashes, all model revisions, decoder settings, package versions, seed and GPU.
- `predictions/<id>.json`: raw caption, token canvas, EOS/mask flags, actual forward count and elapsed time including audio feature preparation. Model loading is excluded from timing.
- `model_load.json`: checkpoint-loading diagnostics.
- `scores.json`: per-clip noun spans, support flags, references, aggregate scores and scorer provenance.
- `review.csv`: audio paths, captions, reference annotations, flagged nouns, and **separate manual columns for scene/source claims and audible-event claims**.

For the first 50 clips, listen to **every** clip and review the whole caption, including unflagged claims. Fill each manual column with `yes`, `no`, or `uncertain`; record the specific unsupported claim in `review_notes`. A scene/source error is an unsupported assertion about an object, entity or setting. An audible-event error is a claim that a sound occurred when it was not heard. An inferred source may be uncertain even when its sound description is reasonable. Keep these judgments separate; the automatic noun score does not distinguish them. Do not infer absence from missing reference words alone. Re-scoring preserves the three manual columns for matching clip IDs and captions; keep a separate backup of your listening judgments as well.

Re-score saved captions on CPU without loading DIFFA:

```bash
source .venv/bin/activate
python -m baseline.score --run runs/baseline50
```

If moving results between machines, also copy the cohort directory and supply `--manifest /new/path/manifest.json`; the manifest and audio hashes must still match.

Individual steps:

```bash
python -m baseline.prepare --samples 50 --out data/audiocaps50
python -m baseline.run --manifest data/audiocaps50/manifest.json --out runs/baseline50
python -m unittest discover -s tests -v
```

CPU-only preparation/scoring needs `numpy`, `soundfile`, `scipy`, `datasets`, `spacy`, and `en_core_web_sm==3.8.0`; model inference additionally needs the full pinned environment. `tests/` uses synthetic audio and captions and never downloads model weights or the full dataset.

Validation on 2026-10-27: all eight CPU tests passed; the real two-clip and 50-clip cohorts loaded and passed audio-hash checks (50 clips, zero exclusions, five references each). A tiny randomly initialized upstream LLaDA model exercised the official FPD API on CPU, including audio embedding insertion and answer extraction. **Full DIFFA checkpoint loading, GPU generation and quality measurements have not been run here**, because local GPU access is unavailable. No generated-corpus hallucination result is claimed.

## Pinned upstreams

- [DIFFA source](https://github.com/NKU-HLT/DIFFA): `0a87594e110984fbaf00ebf5bfe5840c212e4438`.
- [DIFFA-2 weights](https://huggingface.co/zhoujiaming777/DIFFA-2): `8aea6122bafb25578cb49d142c6ac1f8e83e3dea`.
- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct): `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3): `06f233fe06e710322aca913c1bc4249a0d71fce1`.
- AudioCaps/ECHO dataset: `89ecbf42744b923421e14dff2ae484d015d88391`.

The loader adapts the official code's hard-coded local checkpoint paths, restores the checkpoint's connector settings, loads the public Whisper encoder and pinned LLaDA base, and merges DIFFA's LoRA weights when present (alpha 16, as in the upstream training configuration). Required connector/LoRA mismatches abort loading. Upstream code is fetched, not vendored; its licenses and the models'/dataset's terms apply separately.

After the audit confirms real failures, the next experiment can compare the same clips and prompt before/after Detect–Cut–Heal, reporting both unsupported claims and lost correct content. No intervention is implemented in this baseline.
