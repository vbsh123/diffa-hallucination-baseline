#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source "${AUDIO_VENV:-$PWD/.venv}/bin/activate"
export NLTK_DATA="${NLTK_DATA:-$PWD/.cache/nltk_data}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
dch_samples="${SAMPLES:-50}"
dch_args=()
dch_name="dch$dch_samples"
case "${MODE:-full}" in
  full) ;;
  detect-only) dch_args+=(--detect-only); dch_name="dch_detect$dch_samples" ;;
  *) echo 'MODE must be full or detect-only' >&2; exit 2 ;;
esac
if [[ -n "${MANIFEST:-}" ]]; then dch_args+=(--manifest "$MANIFEST"); fi
if [[ "${AUDIO_META_STOPLIST:-0}" == 1 ]]; then dch_args+=(--audio-meta-stoplist); fi
python -u -m dch.run \
  --baseline-run "${BASELINE_RUN:-runs/baseline50}" \
  --out "${RUN_NAME:-runs/$dch_name}" \
  --samples "$dch_samples" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}" \
  --device "${DEVICE:-cuda:0}" \
  "${dch_args[@]}"
