# news-nlp-classifier
End-to-end news article classification system with spaCy preprocessing, TF-IDF/Word2Vec models, FastAPI API, Dockerized deployment, and a simple UI.

## Features
* **Config-Driven Preprocessing:** Deterministic text cleaning with optional spaCy integration (lemmatization, stop words) and fast batch processing via `nlp.pipe()`.
* **Production-Ready API:** Built with FastAPI, featuring request tracking middleware, latency calculation, and strict text validation.
* **Containerized Deployment:** A lightweight, optimized Docker setup for instant, isolated deployment.
* **High Baseline Accuracy:** Evaluated on the AG News dataset with solid multi-class performance.

## Baseline Results (AG News)
* **Accuracy:** 0.9187
* **Macro F1:** 0.9185
* **Evaluation Output:** `eval/results/eval_2026-03-04.json`
* **Confusion Matrix:** `eval/results/confusion_matrix_2026-03-04.csv`

## Project Structure 
* `configs/base.yaml` - Configuration settings (spaCy, preprocessing, and model hyperparameters)
* `src/newsclf/preprocessing/spacy_preprocess.py` - Text preparation (regex baseline + optional spaCy)
* `src/newsclf/api/main.py` - FastAPI application, middleware, and inference endpoints
* `docker/Dockerfile` - Instructions for containerizing the API
* `tests/` - Unit tests for preprocessing and API endpoints
* `artifacts/` - Serialized model, vectorizer, and label mapping

---

## Setup
pip install -e .
python -m spacy download en_core_web_sm

## Running Tests
To run the test suite (which validates preprocessing and API endpoint logic/error handling):

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

## Local Setup (Development)

### Install dependencies in editable mode
pip install -e .

### Download spaCy model
python -m spacy download en_core_web_sm

### 1. Build the image
docker build -t newsclf -f docker/Dockerfile .

### 2. Run the container
docker run --rm --name newsclf -p 8000:8000 newsclf

---

## API Documentation
The API includes middleware that calculates latency_ms and attaches a unique x-request-id to the headers of every response.

1. Health Check
Verify the API is running and the model artifacts are loaded.

curl -X GET [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

{
  "status": "ok", 
  "model_loaded": true
}

2. Predict Classification
Classifies text into one of four categories: World, Sports, Business, Sci/Tech. Includes text validation (max 5000 characters, no empty strings).

curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
     -H "Content-Type: application/json" \
     -d '{"text": "Apple releases new iPhone with advanced AI features for developers."}'

Response:

{
  "label": "Sci/Tech",
  "confidence": 0.92,
  "request_id": "dbd60d84-b59c-42c6-bea7-571e671852d8"
}