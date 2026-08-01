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
async def test_list_roots(httpx_mock):
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/roots",
        json={"roots": [{"id": "root1", "path": "/DATA", "enabled": 1}]},
    )
    c = WikiClient(base_url="http://wiki")
    out = await c.list_roots()
    assert out[0]["id"] == "root1"


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
