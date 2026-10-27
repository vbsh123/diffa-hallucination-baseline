#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source "${AUDIO_VENV:-$PWD/.venv}/bin/activate"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
baseline_samples="${SAMPLES:-50}"
baseline_data="${DATA_DIR:-$PWD/data/audiocaps$baseline_samples}"
baseline_run="${RUN_NAME:-$PWD/runs/baseline$baseline_samples}"
python -m baseline.prepare --out "$baseline_data" --samples "$baseline_samples" --seed "${SEED:-20261027}"
python -m baseline.run \
  --manifest "$baseline_data/manifest.json" \
  --out "$baseline_run" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}" \
  --device "${DEVICE:-cuda:0}" \
  --seed "${SEED:-20261027}"
