"""Unit tests for parser.service_files.

These tests use a real sqlite connection but no HTTP — they exercise the
business logic directly.
"""
from pathlib import Path

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job
from parser.repo_records import (
    upsert_file_record, upsert_file_path, set_tombstone, set_last_error,
)


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def _seed(conn, *, file_id, root_id="r1", path=None, mime="text/plain",
          size=10, indexed_at=100, tombstoned=False, last_error=None,
          parser_version="parser/0.2.0"):
    if path is None:
        path = f"/p/{file_id}"
    upsert_file_record(
        conn, file_id=file_id, sha256_full="sha-" + file_id, size=size,
        mime=mime, modalities_done={"text": "bge-m3/v1"},
        parser_version=parser_version, indexed_at=indexed_at,
    )
    upsert_file_path(
        conn, root_id=root_id, path=path, file_id=file_id, mtime_ms=indexed_at,
    )
    if tombstoned:
        set_tombstone(conn, file_id=file_id, at_ms=indexed_at + 1)
    if last_error:
        set_last_error(conn, root_id=root_id, path=path, error=last_error)


def test_list_files_returns_alive_by_default(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="alive1")
    _seed(conn, file_id="dead1", tombstoned=True)
    res = list_files(conn)
    ids = {f["file_id"] for f in res["files"]}
    assert ids == {"alive1"}
    assert res["total"] == 1


def test_list_files_filter_root_id(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="a", root_id="r1")
    _seed(conn, file_id="b", root_id="r2")
    res = list_files(conn, root_id="r1")
    assert {f["file_id"] for f in res["files"]} == {"a"}


def test_list_files_filter_path_prefix(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="a", root_id="r1", path="/data/x/1.md")
    _seed(conn, file_id="b", root_id="r1", path="/data/x/2.md")
    _seed(conn, file_id="c", root_id="r1", path="/other/z.md")
    res = list_files(conn, root_id="r1", path_prefix="/data/x/")
    assert {f["file_id"] for f in res["files"]} == {"a", "b"}


def test_list_files_filter_mime_prefix(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="leg", mime="application/legacy-office/doc")
    _seed(conn, file_id="md", mime="text/markdown")
    res = list_files(conn, mime_prefix="application/legacy-office/")
    assert {f["file_id"] for f in res["files"]} == {"leg"}


def test_list_files_filter_has_error(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="ok")
    _seed(conn, file_id="bad", last_error="boom")
    res = list_files(conn, has_error=True)
    assert {f["file_id"] for f in res["files"]} == {"bad"}
    assert res["files"][0]["last_error"] == "boom"


def test_list_files_tombstoned_filter_all(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="alive")
    _seed(conn, file_id="dead", tombstoned=True)
    res = list_files(conn, tombstoned="all")
    assert {f["file_id"] for f in res["files"]} == {"alive", "dead"}


def test_list_files_tombstoned_only(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="alive")
    _seed(conn, file_id="dead", tombstoned=True)
    res = list_files(conn, tombstoned="tombstoned")
    assert {f["file_id"] for f in res["files"]} == {"dead"}


def test_list_files_sort_size_asc(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="big", size=1000)
    _seed(conn, file_id="small", size=10)
    res = list_files(conn, sort="size", order="asc")
    assert [f["file_id"] for f in res["files"]] == ["small", "big"]


def test_list_files_pagination(conn):
    from parser.service_files import list_files
    for i in range(15):
        _seed(conn, file_id=f"f{i:02d}", indexed_at=100 + i)
    res = list_files(conn, limit=10, offset=0)
    assert len(res["files"]) == 10
    assert res["total"] == 15
    res2 = list_files(conn, limit=10, offset=10)
    assert len(res2["files"]) == 5


def test_list_files_status_ok(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="ok")
    res = list_files(conn)
    assert res["files"][0]["status"] == "ok"


def test_list_files_status_error(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="bad", last_error="boom")
    res = list_files(conn, has_error=True)
    assert res["files"][0]["status"] == "error"


def test_list_files_status_tombstoned(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="dead", tombstoned=True)
    res = list_files(conn, tombstoned="all")
    f = [x for x in res["files"] if x["file_id"] == "dead"][0]
    assert f["status"] == "tombstoned"


def test_list_files_status_indexing_when_open_job(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="busy", root_id="r1", path="/p/busy")
    enqueue_job(conn, root_id="r1", path="/p/busy", op="reindex",
                priority=1000, now_ms=200)
    res = list_files(conn)
    f = [x for x in res["files"] if x["file_id"] == "busy"][0]
    assert f["status"] == "indexing"


def test_list_files_includes_paths_and_modalities(conn):
    from parser.service_files import list_files
    _seed(conn, file_id="a", root_id="r1", path="/p/a")
    res = list_files(conn)
    f = res["files"][0]
    assert f["paths"] == [
        {"root_id": "r1", "path": "/p/a", "mtime_ms": 100}
    ]
    assert f["modalities_done"] == {"text": "bge-m3/v1"}
    assert f["sha256_full"] == "sha-a"


def test_select_file_ids_by_filter_matches_root(conn):
    from parser.service_files import select_file_ids_by_filter
    _seed(conn, file_id="a", root_id="r1")
    _seed(conn, file_id="b", root_id="r1")
    _seed(conn, file_id="c", root_id="r2")
    ids = select_file_ids_by_filter(conn, root_id="r1", limit=10000)
    assert set(ids) == {"a", "b"}


def test_select_file_ids_by_filter_mime_prefix(conn):
    from parser.service_files import select_file_ids_by_filter
    _seed(conn, file_id="leg1", mime="application/legacy-office/doc")
    _seed(conn, file_id="leg2", mime="application/legacy-office/ppt")
    _seed(conn, file_id="md",   mime="text/markdown")
    ids = select_file_ids_by_filter(
        conn, mime_prefix="application/legacy-office/", limit=10000,
    )
    assert set(ids) == {"leg1", "leg2"}


def test_select_file_ids_by_filter_excludes_tombstoned_by_default(conn):
    from parser.service_files import select_file_ids_by_filter
    _seed(conn, file_id="alive")
    _seed(conn, file_id="dead", tombstoned=True)
    ids = select_file_ids_by_filter(conn, limit=10000)
    assert set(ids) == {"alive"}


def test_select_file_ids_by_filter_caps_at_limit(conn):
    from parser.service_files import select_file_ids_by_filter
    for i in range(50):
        _seed(conn, file_id=f"f{i:02d}")
    # limit=10 → only 10 returned, but the COUNT path is for the caller
    ids = select_file_ids_by_filter(conn, limit=10)
    assert len(ids) == 10
