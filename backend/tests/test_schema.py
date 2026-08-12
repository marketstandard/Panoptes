import json
from pathlib import Path

import jsonschema

from panoptes.analysis.pipeline import analyze
from panoptes.schemas import AnalysisRequest
from panoptes.settings import Settings


def test_response_matches_schema() -> None:
    schema = json.loads(
        Path(__file__).resolve().parents[2].joinpath("schemas", "analysis-response.schema.json").read_text()
    )
    response = analyze(AnalysisRequest(fixture="ai-prose"), Settings())
    jsonschema.validate(response.public_dict(), schema)
