import hashlib
import json
import time

import pytest

from parser.db import init_db
from parser.pipeline_base import (
    sha256_file, IdentityResolver, ResolveOutcome,
)
from parser.repo_records import (
    upsert_file_record, upsert_file_path, get_file_record,
    count_paths_for_file_in_root,
)


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def write_file(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_sha256_file(tmp_path):
    p = write_file(tmp_path, "a.txt", b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert sha256_file(p) == expected


def test_resolve_new_file_returns_full_index(conn, tmp_path):
    p = write_file(tmp_path, "a.txt", b"hello")
    r = IdentityResolver(conn, parser_version="parser/0.1.0",
                         active_modalities={"text": "bge-m3/v1"})
    outcome = r.resolve(root_id="root1", path=str(p), now_ms=100)
    assert outcome.action == ResolveOutcome.FULL_INDEX
    assert outcome.new_file_id is not None


def test_resolve_known_file_same_modalities_only_appends_root(conn, tmp_path):
    p = write_file(tmp_path, "a.txt", b"hello")
    full = hashlib.sha256(b"hello").hexdigest()
    fid = full[:32]
    upsert_file_record(conn, file_id=fid, sha256_full=full, size=5,
                       mime="text/plain",
                       modalities_done={"text": "bge-m3/v1"},
                       parser_version="parser/0.1.0", indexed_at=1)
    upsert_file_path(conn, root_id="rootA", path="/old.txt",
                     file_id=fid, mtime_ms=1)
    r = IdentityResolver(conn, parser_version="parser/0.1.0",
                         active_modalities={"text": "bge-m3/v1"})
    outcome = r.resolve(root_id="rootB", path=str(p), now_ms=100)
    assert outcome.action == ResolveOutcome.APPEND_ROOT_ONLY
    assert outcome.new_file_id == fid


def test_resolve_modify_old_orphan_goes_to_tombstone(conn, tmp_path):
    p = write_file(tmp_path, "a.txt", b"hello")
    old_full = hashlib.sha256(b"old content").hexdigest()
    old_fid = old_full[:32]
    upsert_file_record(conn, file_id=old_fid, sha256_full=old_full, size=11,
                       mime="text/plain", modalities_done={"text": "bge-m3/v1"},
                       parser_version="parser/0.1.0", indexed_at=1)
    upsert_file_path(conn, root_id="root1", path=str(p),
                     file_id=old_fid, mtime_ms=1)
    r = IdentityResolver(conn, parser_version="parser/0.1.0",
                         active_modalities={"text": "bge-m3/v1"})
    outcome = r.resolve(root_id="root1", path=str(p), now_ms=100)
    assert outcome.action == ResolveOutcome.FULL_INDEX
    assert outcome.new_file_id != old_fid
    assert outcome.old_orphan_file_id == old_fid


def test_resolve_revival_clears_tombstone(conn, tmp_path):
    p = write_file(tmp_path, "a.txt", b"hello")
    full = hashlib.sha256(b"hello").hexdigest()
    fid = full[:32]
    upsert_file_record(conn, file_id=fid, sha256_full=full, size=5,
                       mime="text/plain",
                       modalities_done={"text": "bge-m3/v1"},
                       parser_version="parser/0.1.0", indexed_at=1)
    conn.execute(
        "UPDATE file_records SET tombstoned_at = 50 WHERE file_id = ?", (fid,)
    )
    r = IdentityResolver(conn, parser_version="parser/0.1.0",
                         active_modalities={"text": "bge-m3/v1"})
    outcome = r.resolve(root_id="root1", path=str(p), now_ms=100)
    assert outcome.action == ResolveOutcome.REVIVE
    assert outcome.new_file_id == fid
    rec = get_file_record(conn, fid)
    assert rec["tombstoned_at"] is None
