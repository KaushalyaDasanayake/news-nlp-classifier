import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from newsclf.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_api_dependencies():
    # Create a fake file object that returns valid JSON string when read
    mock_file = MagicMock()
    mock_file.read.return_value = json.dumps({"labels": [0, 1, 2, 3]})

    # Setup the mocks
    with (
        patch("newsclf.api.main.joblib.load"),
        patch("newsclf.api.main.yaml.safe_load") as mock_yaml,
        patch("newsclf.api.main.preprocess_many") as mock_prep,
        patch("builtins.open") as mock_open,
    ):
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock the Config YAML
        mock_yaml.return_value = {"paths": {"artifacts_dir": "artifacts"}}

        import newsclf.api.main as api_main

        # Create mock objects
        mock_vec = MagicMock()
        mock_vec.transform.return_value = MagicMock()
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.1, 0.1, 0.7]])

        # Manually set the globals to bypass the startup loading errors
        api_main.CFG = mock_yaml.return_value
        api_main.VEC = mock_vec
        api_main.CLF = mock_clf
        api_main.LABELS = [0, 1, 2, 3]

        mock_prep.return_value = ["cleaned text"]

        yield


# test health
def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# test valid pred
def test_predict_valid():
    res = client.post("/predict", json={"text": "Apple releases new iPhone with new AI features"})
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "Sci/Tech"


# test empty text
def test_predict_empty_text():
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 422


# test too long text
def test_predict_too_long():
    long_text = "a" * 6001
    res = client.post("/predict", json={"text": long_text})
    assert res.status_code == 422
