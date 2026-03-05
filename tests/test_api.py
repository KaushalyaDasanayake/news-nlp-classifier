from fastapi.testclient import TestClient

from newsclf.api.main import app

client = TestClient(app)


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
