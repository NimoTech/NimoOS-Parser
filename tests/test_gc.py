import shutil
from pathlib import Path

import pytest

from parser.db import init_db
from parser.gc import sweep_tombstones
from parser.repo_records import upsert_file_record


class FakeQ:
    def __init__(self):
        self.deleted = []
    def delete_file(self, file_id):
        self.deleted.append(file_id)


def test_sweep_deletes_old_tombstones(tmp_path):
    conn = init_db(tmp_path / "p.db")
    upsert_file_record(conn, file_id="a", sha256_full="a" * 64, size=1,
                       mime="text/plain", modalities_done={},
                       parser_version="p", indexed_at=1)
    upsert_file_record(conn, file_id="b", sha256_full="b" * 64, size=1,
                       mime="text/plain", modalities_done={},
                       parser_version="p", indexed_at=1)
    conn.execute("UPDATE file_records SET tombstoned_at = 100 WHERE file_id = 'a'")
    conn.execute("UPDATE file_records SET tombstoned_at = 999 WHERE file_id = 'b'")
    figures_root = tmp_path / "figures"
    (figures_root / "a").mkdir(parents=True)
    (figures_root / "b").mkdir(parents=True)
    q = FakeQ()
    # now_ms=86_400_500 so cutoff = 86_400_500 - 86_400_000 = 500
    # "a" tombstoned_at=100 < 500 → reaped; "b" tombstoned_at=999 > 500 → kept
    n = sweep_tombstones(conn, qstore=q, figures_root=figures_root,
                          grace_ms=24 * 3600 * 1000, now_ms=86_400_500)
    assert n == 1
    assert q.deleted == ["a"]
    assert not (figures_root / "a").exists()
    assert (figures_root / "b").exists()
    assert conn.execute(
        "SELECT 1 FROM file_records WHERE file_id = 'a'"
    ).fetchone() is None
    assert conn.execute(
        "SELECT 1 FROM file_records WHERE file_id = 'b'"
    ).fetchone() is not None
