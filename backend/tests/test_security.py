from fastapi.testclient import TestClient

from panoptes.main import app


def test_large_text_is_rejected() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"text": "x" * 120_001})
    assert response.status_code == 413


def test_metrics_disabled_by_default() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 404


def test_markup_is_not_executed_and_response_is_json() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"text": "<script>alert(1)</script> " * 60})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "<script>" not in response.text
