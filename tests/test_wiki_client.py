import httpx
import pytest

from parser.wiki_client import WikiClient


@pytest.mark.asyncio
async def test_fetch_file_events(httpx_mock):
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/_internal/file-events?since=10&after_seq=0&limit=100",
        json={"events": [{"id": "e1", "root_id": "r", "path": "/a.md",
                          "op": "create", "is_dir": 0,
                          "detected_at": 20}]},
    )
    c = WikiClient(base_url="http://wiki")
    out = await c.fetch_file_events(since_ms=10, limit=100)
    assert len(out) == 1
    assert out[0]["path"] == "/a.md"


@pytest.mark.asyncio
async def test_report_index_status(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://wiki/v1/wiki/_internal/index-status",
        json={"ok": True},
    )
    c = WikiClient(base_url="http://wiki")
    await c.report_index_status(
        path="/a.md", status="indexed", parser_version="parser/0.1.0",
        modalities={"text": "bge-m3/v1"},
    )


@pytest.mark.asyncio
async def test_list_roots_normalizes_wiki_shape(httpx_mock):
    # Wiki returns a bare array with Go-cased keys (route/v1/roots.go listRoots).
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/roots",
        json=[{"ID": "root1", "Path": "/DATA", "Enabled": True, "Level": "space"}],
    )
    c = WikiClient(base_url="http://wiki")
    out = await c.list_roots()
    assert out == [{"id": "root1", "path": "/DATA", "enabled": True}]


@pytest.mark.asyncio
async def test_list_roots_accepts_legacy_wrapped_shape(httpx_mock):
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/roots",
        json={"roots": [{"id": "root1", "path": "/DATA", "enabled": 1}]},
    )
    c = WikiClient(base_url="http://wiki")
    assert await c.list_roots() == [{"id": "root1", "path": "/DATA", "enabled": True}]


@pytest.mark.asyncio
async def test_fetch_file_events_page_returns_whole_body(httpx_mock):
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/_internal/file-events?since=10&after_seq=0&limit=100",
        json={"events": [], "archive_cutoff_ms": 123, "has_archived": True},
    )
    c = WikiClient(base_url="http://wiki")
    page = await c.fetch_file_events_page(since_ms=10, limit=100)
    assert page["archive_cutoff_ms"] == 123 and page["has_archived"] is True


@pytest.mark.asyncio
async def test_fetch_root_files_pages_and_404(httpx_mock):
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/_internal/files?root_id=r1&after=&limit=2",
        json={"files": [{"path": "/DATA/a.md", "mtime_ms": 1, "size": 1}], "next_after": ""},
    )
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/_internal/files?root_id=gone&after=&limit=1000",
        status_code=404, json={"message": "root not found"},
    )
    c = WikiClient(base_url="http://wiki")
    page = await c.fetch_root_files("r1", limit=2)
    assert page["files"][0]["path"] == "/DATA/a.md" and page["next_after"] == ""
    from parser.wiki_client import WikiRootNotFound
    with pytest.raises(WikiRootNotFound):
        await c.fetch_root_files("gone")


@pytest.mark.asyncio
async def test_wiki_client_rereads_discovery_on_connect_error(tmp_path, monkeypatch):
    from parser.wiki_client import WikiClient

    url_file = tmp_path / "wiki.url"
    url_file.write_text("http://127.0.0.1:59999")  # new address
    c = WikiClient("http://127.0.0.1:59998",       # old address (dead)
                   discovery_path=str(url_file))

    calls = []

    async def fake_request(self, method, path, **kw):
        calls.append(str(self.base_url))
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        # httpx internals: raise_for_status() requires a request bound to
        # the response, which a bare httpx.Response(...) doesn't have.
        resp = httpx.Response(200, json={"events": []},
                              request=httpx.Request(method, path))
        return resp

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    events = await c.fetch_file_events(since_ms=0, after_seq=0)
    assert events == []
    assert calls[0].startswith("http://127.0.0.1:59998")
    assert calls[1].startswith("http://127.0.0.1:59999")
