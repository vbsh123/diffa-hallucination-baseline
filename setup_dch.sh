#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source "${AUDIO_VENV:-$PWD/.venv}/bin/activate"
export NLTK_DATA="${NLTK_DATA:-$PWD/.cache/nltk_data}"
python -m pip install -r requirements-dch.txt
python -m nltk.downloader -d "$NLTK_DATA" averaged_perceptron_tagger_eng wordnet stopwords punkt_tab
python -m pip check
python -c 'from dch.nlp import PaperNLP; p=PaperNLP().provenance(); print("DCH NLP ready:",p["nltk"],"WordNet",p["wordnet"])'
