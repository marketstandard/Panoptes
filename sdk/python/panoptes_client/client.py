from __future__ import annotations

from typing import Any

import httpx


class PanoptesError(RuntimeError):
    pass


class PanoptesClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def analyze(
        self,
        text: str | None = None,
        *,
        filename: str | None = None,
        file_base64: str | None = None,
        prior_odds: float = 1.0,
    ) -> dict[str, Any]:
        response = self._client.post(
            "/api/v1/analyze",
            json={
                "text": text,
                "filename": filename,
                "file_base64": file_base64,
                "prior_odds": prior_odds,
            },
        )
        if response.status_code >= 400:
            raise PanoptesError(f"Analysis failed: {response.status_code} {response.text}")
        return response.json()

    def capabilities(self) -> dict[str, Any]:
        response = self._client.get("/api/v1/capabilities")
        response.raise_for_status()
        return response.json()

    def runtime(self) -> dict[str, Any]:
        response = self._client.get("/api/v1/runtime")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PanoptesClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
