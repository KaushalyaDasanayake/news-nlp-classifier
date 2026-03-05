from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Local import from your project
from newsclf.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_api_dependencies():
    """
    Mock all external dependencies: Files, Joblib, and Global Variables.
    This prevents FileNotFoundError in GitHub Actions.
    """

    with (
        patch("newsclf.api.main.joblib.load"),
        patch("newsclf.api.main.yaml.safe_load") as mock_yaml,
        patch("newsclf.api.main.preprocess_many") as mock_prep,
        patch("builtins.open", MagicMock()),
    ):
        # Mock the Config YAML data
        mock_yaml.return_value = {"paths": {"artifacts_dir": "artifacts"}}

        # Mock Global Variables inside newsclf/api/main.py
        import newsclf.api.main as api_main

        # Create a mock vectorizer
        mock_vec = MagicMock()
        mock_vec.transform.return_value = MagicMock()

        # Create a mock classifier that returns probabilities
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.1, 0.1, 0.7]])

        # Inject these 'fake' objects into the API so it doesn't crash
        api_main.CFG = mock_yaml.return_value
        api_main.VEC = mock_vec
        api_main.CLF = mock_clf
        api_main.LABELS = [0, 1, 2, 3]

        # Mock preprocessing to just return a simple list
        mock_prep.return_value = ["cleaned text"]

        yield


# test health
def test_health():
    with client:
        res = client.get("/health")

    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# test valid pred
def test_predict_valid():
    res = client.post("/predict", json={"text": "Apple releases new iPhone with new AI features"})

    assert res.status_code == 200

    data = res.json()

    assert "label" in data
    assert "confidence" in data
    assert "request_id" in data


# test empty text
def test_predict_empty_text():
    res = client.post("/predict", json={"text": ""})

    assert res.status_code == 422


# test too long text
def test_predict_too_long():
    long_text = "a" * 6000

    res = client.post("/predict", json={"text": long_text})

    assert res.status_code == 422
