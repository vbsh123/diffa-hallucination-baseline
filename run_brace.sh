#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source "${AUDIO_VENV:-$PWD/.venv}/bin/activate"
export NLTK_DATA="${NLTK_DATA:-$PWD/.cache/nltk_data}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
brace_clips="${CLIPS:-5}"
brace_data="${BRACE_DATA:-data/brace$brace_clips}"
brace_args=()
brace_name="brace$brace_clips"
case "${MODE:-full}" in
  full) ;;
  detect-only) brace_args+=(--detect-only); brace_name="brace_detect$brace_clips" ;;
  *) echo 'MODE must be full or detect-only' >&2; exit 2 ;;
esac
python -u -m dch.brace prepare --clips "$brace_clips" --out "$brace_data"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then exit 0; fi
python -u -m dch.run \
  --brace-cohort "$brace_data/manifest.json" \
  --out "${RUN_NAME:-runs/$brace_name}" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}" \
  --device "${DEVICE:-cuda:0}" \
  "${brace_args[@]}"
