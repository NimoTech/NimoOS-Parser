from typing import Optional

import httpx


class WikiClient:
    def __init__(self, base_url: str, *, timeout_s: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url,
                                          timeout=timeout_s)

    async def fetch_file_events(
        self, *, since_ms: int, limit: int = 200,
    ) -> list[dict]:
        r = await self._client.get(
            "/v1/wiki/_internal/file-events",
            params={"since": since_ms, "limit": limit},
        )
        r.raise_for_status()
        return r.json().get("events", [])

    async def report_index_status(
        self, *, path: str, status: str, parser_version: str,
        modalities: Optional[dict] = None, error: Optional[str] = None,
    ) -> None:
        body = {
            "path": path, "status": status,
            "parser_version": parser_version,
        }
        if modalities is not None:
            body["modalities"] = modalities
        if error is not None:
            body["error"] = error
        r = await self._client.post(
            "/v1/wiki/_internal/index-status", json=body,
        )
        r.raise_for_status()

    async def list_roots(self) -> list[dict]:
        r = await self._client.get("/v1/wiki/roots")
        r.raise_for_status()
        return r.json().get("roots", [])

    async def aclose(self) -> None:
        await self._client.aclose()
