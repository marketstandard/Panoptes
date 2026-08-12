from fastapi.testclient import TestClient

from panoptes.main import app


def test_healthz() -> None:
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_fixture() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"text": "AI-generated " * 80})
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.1.0"
    assert payload["runtime"]["profile"] == "fixture"
    assert 0 <= payload["summary"]["overall"]["ai_generated"] <= 1
    assert payload["limitations"]


def test_analyze_short_input_abstains() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"text": "Too short."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["evidence_state"] == "insufficient_data"
