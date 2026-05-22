import pytest

from parser.wiki_client import WikiClient


@pytest.mark.asyncio
async def test_fetch_file_events(httpx_mock):
    httpx_mock.add_response(
        url="http://wiki/v1/wiki/_internal/file-events?since=10&limit=100",
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
