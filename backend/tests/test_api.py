from fastapi.testclient import TestClient
import pytest

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
    assert payload["schema_version"] == "1.2.0"
    assert payload["runtime"]["profile"] == "fixture"
    assert 0 <= payload["summary"]["overall"]["ai_generated"] <= 1
    assert payload["limitations"]


def test_analyze_reports_distinct_attribution_outputs() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"text": "AI-generated " * 80, "prior_odds": 1.0})
    payload = response.json()
    overall = payload["summary"]["overall"]
    assert payload["summary"]["ai_participation"] == pytest.approx(
        overall["ai_generated"] + overall["ai_refined_or_mixed"]
    )
    assert payload["summary"]["ai_generation"] == overall["ai_generated"]
    assert payload["posterior"]["cohort_prevalence"] == 0.5
    assert payload["provenance"]["level"] in {"P0", "P1", "P2", "P3", "P4"}


def test_analyze_short_input_abstains() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"text": "Too short."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["evidence_state"] == "insufficient_data"
