import json
import time

import pytest

from parser.db import init_db
from parser.repo_records import (
    upsert_file_record, get_file_record,
    upsert_file_path, list_paths_for_file, remove_file_path,
    list_paths_under_root,
    set_tombstone, clear_tombstone, count_paths_for_file_in_root,
)


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def test_upsert_and_get_record(conn):
    now = int(time.time() * 1000)
    upsert_file_record(
        conn, file_id="abc123", sha256_full="abc123" + "0" * 58, size=42,
        mime="text/markdown",
        modalities_done={"text": "bge-m3/v1"},
        parser_version="parser/0.1.0", indexed_at=now,
    )
    rec = get_file_record(conn, "abc123")
    assert rec["mime"] == "text/markdown"
    assert json.loads(rec["modalities_done"]) == {"text": "bge-m3/v1"}
    assert rec["tombstoned_at"] is None


def test_path_lifecycle(conn):
    upsert_file_record(conn, file_id="abc", sha256_full="abc" + "0" * 61, size=1,
                       mime="text/plain", modalities_done={"text": "bge-m3/v1"},
                       parser_version="parser/0.1.0", indexed_at=1)
    upsert_file_path(conn, root_id="root1", path="/a/b.txt",
                     file_id="abc", mtime_ms=100)
    upsert_file_path(conn, root_id="root1", path="/a/dup.txt",
                     file_id="abc", mtime_ms=100)
    upsert_file_path(conn, root_id="root2", path="/x.txt",
                     file_id="abc", mtime_ms=100)
    assert count_paths_for_file_in_root(conn, "abc", "root1") == 2
    assert count_paths_for_file_in_root(conn, "abc", "root2") == 1
    paths = list_paths_for_file(conn, "abc")
    assert len(paths) == 3
    remove_file_path(conn, root_id="root1", path="/a/b.txt")
    assert count_paths_for_file_in_root(conn, "abc", "root1") == 1


def test_list_paths_under_root(conn):
    upsert_file_record(conn, file_id="a", sha256_full="a" + "0" * 63, size=1,
                       mime="text/plain", modalities_done={},
                       parser_version="parser/0.1.0", indexed_at=1)
    upsert_file_path(conn, "root1", "/x.txt", "a", 1)
    upsert_file_path(conn, "root2", "/y.txt", "a", 1)
    rs = list_paths_under_root(conn, "root1")
    assert [r["path"] for r in rs] == ["/x.txt"]


def test_tombstone_set_clear(conn):
    upsert_file_record(conn, file_id="abc", sha256_full="abc" + "0" * 61, size=1,
                       mime="text/plain", modalities_done={},
                       parser_version="parser/0.1.0", indexed_at=1)
    set_tombstone(conn, file_id="abc", at_ms=999)
    rec = get_file_record(conn, "abc")
    assert rec["tombstoned_at"] == 999
    clear_tombstone(conn, file_id="abc")
    rec = get_file_record(conn, "abc")
    assert rec["tombstoned_at"] is None
