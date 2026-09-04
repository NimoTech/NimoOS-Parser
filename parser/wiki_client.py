import logging
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("parser.wiki_client")


class WikiRootNotFound(Exception):
    """Wiki has no such root (404 from /_internal/files)."""


class WikiClient:
    def __init__(self, base_url: str, *, discovery_path: str | None = None,
                 timeout: float = 10.0) -> None:
        self._discovery_path = discovery_path
        self._timeout = timeout
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    def _read_discovery(self) -> str | None:
        if not self._discovery_path:
            return None
        try:
            return Path(self._discovery_path).read_text().strip() or None
        except OSError:
            return None

    async def _request(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            return await self._client.request(method, path, **kw)
        except httpx.RequestError:
            # Wiki restarts on a random port and rewrites wiki.url; a frozen
            # base_url strands us until our own restart (bitten 3x — the
            # "service discovery hot-read" follow-up). Re-resolve and retry
            # once; re-raise if discovery is unavailable or unchanged-and-
            # still-failing.
            fresh = self._read_discovery()
            if not fresh:
                raise
            # Concurrency invariant: this compare-and-swap of self._client is
            # safe WITHOUT a lock only because there is no await between
            # reading base_url and reassigning self._client — do not insert
            # one (racing coroutines would double-swap/double-close).
            if fresh != str(self._client.base_url).rstrip("/"):
                old = self._client
                self._client = httpx.AsyncClient(base_url=fresh,
                                                 timeout=self._timeout)
                try:
                    await old.aclose()
                except Exception:  # noqa: BLE001 — best-effort close
                    pass
                log.info("wiki base_url re-resolved to %s", fresh)
            return await self._client.request(method, path, **kw)

    async def fetch_file_events_page(
        self, *, since_ms: int, after_seq: int = 0, limit: int = 200,
    ) -> dict:
        """Whole feed page: `events` plus, on Wiki >= 2026-09, the archive
        horizon (`archive_cutoff_ms`, `has_archived`) the consumer uses to
        detect a cursor that fell behind the archive line."""
        r = await self._request(
            "GET", "/v1/wiki/_internal/file-events",
            params={"since": since_ms, "after_seq": after_seq, "limit": limit},
        )
        r.raise_for_status()
        body = r.json()
        return body if isinstance(body, dict) else {"events": body}

    async def fetch_file_events(
        self, *, since_ms: int, after_seq: int = 0, limit: int = 200,
    ) -> list[dict]:
        page = await self.fetch_file_events_page(
            since_ms=since_ms, after_seq=after_seq, limit=limit)
        return page.get("events", [])

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
        r = await self._request(
            "POST", "/v1/wiki/_internal/index-status", json=body,
        )
        r.raise_for_status()

    async def list_roots(self) -> list[dict]:
        """Normalized `[{"id","path","enabled"}]`. Wiki's GET /v1/wiki/roots
        returns a bare array with Go-cased keys; the wrapped lower-case shape
        is accepted for older fixtures."""
        r = await self._request("GET", "/v1/wiki/roots")
        r.raise_for_status()
        body = r.json()
        rows = body.get("roots", []) if isinstance(body, dict) else body
        rows = rows or []
        out = []
        dropped = 0
        for row in rows:
            rid = row.get("id") or row.get("ID")
            if not rid:
                # Wiki's repo.WikiRoot pins its json tags to the PascalCase
                # names (ID/Path/Enabled, models_wire_test.go), so a rename
                # upstream now fails Wiki's own tests before reaching the wire.
                # Keep the guard anyway: silently returning [] once told verify
                # "Wiki holds no roots" and it retired the whole ledger — count,
                # log, and refuse outright when nothing is usable.
                dropped += 1
                continue
            out.append({
                "id": rid,
                "path": row.get("path") if "path" in row else row.get("Path", ""),
                "enabled": bool(row.get("enabled") if "enabled" in row else row.get("Enabled", True)),
            })
        if dropped:
            log.warning("wiki roots: dropped %d of %d rows with no usable id",
                        dropped, len(rows))
            if not out:
                raise RuntimeError("wiki roots response has no usable ids")
        return out

    async def fetch_root_files(self, root_id: str, *, after: str = "",
                               limit: int = 1000) -> dict:
        r = await self._request(
            "GET", "/v1/wiki/_internal/files",
            params={"root_id": root_id, "after": after, "limit": limit},
        )
        if r.status_code == 404:
            raise WikiRootNotFound(root_id)
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()
