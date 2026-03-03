# news-nlp-classifier
End-to-end news article classification system with spaCy preprocessing, TF-IDF/Word2Vec models, FastAPI API, Dockerized deployment, and a simple UI.

## Current status
Milestone 2 complete: config-driven, deterministic preprocessing with optional spaCy (lemmatization, wtopwords) + fast batch processing via `nlp.pipe()`.

## Project structure 
- `configs/base.yaml` - all settings (spaCy + preprocessing + model hyperparams)
- `src/newsclf/preprocessing/spacy_preprocess.py` - preparing (regex baseline + optional spacy)
- `tests/test_preprocess.py` - unit tests

## Setup
pip install -e .
python -m spacy download en_core_web_sm

## Run test
pytest

## Quick preprocessing demo
python - <<'PY'
import yaml
from newsclf.preprocessing.spacy_preprocess import preprocess_many

cfg = yaml.safe_load(open("configs/base.yaml"))
texts = ["Email me test@example.com and visit https://example.com"] * 3
print(preprocess_many(texts, cfg))
PY

---

## Baseline results (AG News)
- Accuracy: **0.9187**
- Macro F1: **0.9185**
- Eval output: `eval/results/eval_2026-03-04.json`
- Confusion matrix: `eval/results/confusion_matrix_2026-03-04.csv`