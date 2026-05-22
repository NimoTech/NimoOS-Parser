import logging
import shutil
import sqlite3
from pathlib import Path

from parser.repo_records import list_tombstoned_older_than, delete_file_record

log = logging.getLogger("parser.gc")


def sweep_tombstones(
    conn: sqlite3.Connection, *, qstore, figures_root: Path,
    grace_ms: int, now_ms: int,
) -> int:
    cutoff = now_ms - grace_ms
    rows = list_tombstoned_older_than(conn, before_ms=cutoff)
    n = 0
    for row in rows:
        fid = row["file_id"]
        try:
            qstore.delete_file(file_id=fid)
            d = Path(figures_root) / fid
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            delete_file_record(conn, file_id=fid)
            n += 1
        except Exception as e:
            log.warning("gc failed for %s: %s", fid, e)
    return n
