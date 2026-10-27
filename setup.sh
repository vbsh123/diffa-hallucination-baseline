#!/usr/bin/env bash
# Installs dependencies and pinned source. No model weights or dataset download.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
baseline_venv="${AUDIO_VENV:-$PWD/.venv}"
if [[ ! -x "$baseline_venv/bin/python" ]]; then
  "${PYTHON:-python3}" -m venv "$baseline_venv"
fi
source "$baseline_venv/bin/activate"
python -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10–3.12"'
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -r requirements.txt
  python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
  python -m pip check
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python -m baseline.checkout --root "${DIFFA_ROOT:-$PWD/external/DIFFA}"
echo 'Setup complete. Run: bash run.sh'
