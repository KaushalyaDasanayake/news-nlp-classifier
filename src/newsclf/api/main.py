# FastAPI implementation
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from newsclf.api.schemas import PredictRequest, PredictResponse
from newsclf.preprocessing.spacy_preprocess import preprocess_many

app = FastAPI(title="news-nlp-classifier", version="0.1.0")

# declare global variables
CFG: dict | None = None
VEC = None
CLF = None
LABELS: list[int] | None = None

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        print(f"request_id={request_id} path={request.url.path} latency_ms={latency_ms:.2f}")

        response.headers["x-request-id"] = request_id
        return response
    
app.add_middleware(RequestIdMiddleware)


@app.on_event("startup")
def load_artifacts() -> None:
    global CFG, VEC, CLF, LABELS

    with open("configs/base.yaml", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    artifacts_dir = Path(CFG["paths"]["artifacts_dir"])
    VEC = joblib.load(artifacts_dir / "vectorizer.joblib")
    CLF = joblib.load(artifacts_dir / "model.joblib")

    with open(artifacts_dir / "labels.json", encoding="utf-8") as f:
        LABELS = json.load(f)["labels"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": VEC is not None and CLF is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    assert CFG is not None
    assert VEC is not None
    assert CLF is not None
    assert LABELS is not None

    text = payload.text

    if not text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    if len(text) > 5000:
        raise HTTPException(status_code=422, detail="text is too long (max 5000 chars)")

    cleaned = preprocess_many([text], CFG)
    X = VEC.transform(cleaned)

    probs = CLF.predict_proba(X)[0]
    pred_idx = int(probs.argmax())
    pred_label_id = LABELS[pred_idx]

    label_str = str(pred_label_id)
    confidence = float(probs[pred_idx])

    request_id = request.state.request_id
    return PredictResponse(label=label_str, confidence=confidence, request_id=request_id)



