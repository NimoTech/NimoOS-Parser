import asyncio
import sqlite3
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from parser.db import init_db
from parser import repo_allowlist, tombstone_task


@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as d:
        c = init_db(Path(d) / "test.db")
        yield c
        c.close()


def _seed_file_record(conn, *, file_id, root_id, path, mime):
    now = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO file_records(file_id, sha256_full, size, mime, "
        "modalities_done, parser_version, indexed_at, vector_count) "
        "VALUES (?, ?, 0, ?, '{\"text\":\"v1\"}', 'test', ?, 1)",
        (file_id, file_id, mime, now),
    )
    conn.execute(
        "INSERT INTO file_paths(root_id, path, file_id, mtime_ms) "
        "VALUES (?, ?, ?, ?)",
        (root_id, path, file_id, now),
    )


@pytest.mark.asyncio
async def test_sweep_tombstones_files_no_longer_indexable(conn):
    # Seed: 2 .pdf files
    _seed_file_record(conn, file_id="f1", root_id="r1",
                       path="/Wiki/a.pdf", mime="text/markdown+docling/pdf")
    _seed_file_record(conn, file_id="f2", root_id="r1",
                       path="/Wiki/b.pdf", mime="text/markdown+docling/pdf")
    # Disable .pdf
    repo_allowlist.set_extension_enabled(conn, ".pdf", False)

    qstore = MagicMock()
    affected = await tombstone_task.sweep_once(conn, qstore=qstore,
                                                now_ms=999)
    assert affected == 2
    qstore.tombstone_file.assert_any_call(file_id="f1", tombstoned_at=999)
    qstore.tombstone_file.assert_any_call(file_id="f2", tombstoned_at=999)


@pytest.mark.asyncio
async def test_sweep_skips_already_tombstoned(conn):
    _seed_file_record(conn, file_id="f1", root_id="r1",
                       path="/Wiki/a.pdf", mime="text/markdown+docling/pdf")
    conn.execute("UPDATE file_records SET tombstoned_at = 1 WHERE file_id = ?",
                  ("f1",))
    repo_allowlist.set_extension_enabled(conn, ".pdf", False)

    qstore = MagicMock()
    affected = await tombstone_task.sweep_once(conn, qstore=qstore, now_ms=2)
    assert affected == 0
    qstore.tombstone_file.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_keeps_files_still_indexable(conn):
    _seed_file_record(conn, file_id="f1", root_id="r1",
                       path="/Wiki/keep.md", mime="text/markdown")
    repo_allowlist.set_extension_enabled(conn, ".pdf", False)

    qstore = MagicMock()
    affected = await tombstone_task.sweep_once(conn, qstore=qstore, now_ms=1)
    assert affected == 0
    qstore.tombstone_file.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_tombstones_records_under_container_dirs(conn):
    # allowed extension, but the path sits under /DATA/.system_data — the
    # startup sweep is what retires records indexed before the gate existed.
    _seed_file_record(conn, file_id="sys", root_id="r1",
                       path="/DATA/.system_data/home/nimo/.claude/cache/changelog.md",
                       mime="text/markdown")
    _seed_file_record(conn, file_id="ok", root_id="r1",
                       path="/DATA/Documents/changelog.md", mime="text/markdown")
    qstore = MagicMock()
    affected = await tombstone_task.sweep_once(conn, qstore=qstore, now_ms=999)
    assert affected == 1
    qstore.tombstone_file.assert_called_once_with(file_id="sys", tombstoned_at=999)
