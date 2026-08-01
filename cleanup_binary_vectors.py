#!/usr/bin/env python3
"""
cleanup_binary_vectors.py - clean up vectors from historically polluted binary files

Background: wiki_consumer's TEXT_EXT_ALLOWLIST was only added later. Before
that, binary files like .sql.gz / .MOV / .jpeg fell through pipeline_text's
else branch, got treated as text/plain, decoded directly with
decode("utf-8", errors="replace"), and chunked - polluting Qdrant.

This script scans parser.db and finds file_records whose file_id has every
one of its paths outside TEXT_EXT_ALLOWLIST (i.e. this file fundamentally
should never have been indexed), then:
  - dry-run (default): print only, no changes
  - --apply: delete points from Qdrant -> delete file_records / file_paths / parse_jobs

Data safety:
  - if any of a file_id's multiple paths matches the allowlist, it's kept
    (the same content also appeared under a legitimate extension, so the
    user may genuinely have indexed it)
  - deleting from Qdrant by file_id can't accidentally hit other files
  - make sure the nimoos-parser service is stopped before running this, to avoid a race

Usage:
  sudo /opt/nimoos-parser/venv/bin/python3 \\
    /home/nimo/nimoos/NimoOS-Parser/cleanup_binary_vectors.py [--apply]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# let the script import parser.* -- it lives at the repo root
sys.path.insert(0, str(Path(__file__).parent))

from parser.wiki_consumer import TEXT_EXT_ALLOWLIST

DB_PATH = "/var/lib/nimoos/parser/parser.db"
QDRANT_URL = "http://127.0.0.1:6333"


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def find_pollution(conn: sqlite3.Connection) -> list[dict]:
    """Returns [{file_id, sample_path, mime, vector_count, all_paths}]
    where all_paths is every (root_id, path) under a file_id; only counted as
    pollution if none of the extensions are in the allowlist."""
    fr_rows = list(conn.execute("""
        SELECT file_id, mime, size, vector_count
        FROM file_records
        WHERE tombstoned_at IS NULL
    """))

    pollution = []
    for fr in fr_rows:
        fid = fr["file_id"]
        paths = list(conn.execute(
            "SELECT root_id, path FROM file_paths WHERE file_id = ?", (fid,)
        ))
        if not paths:
            continue
        # only counts as pollution if every path's extension is NOT IN the allowlist
        exts = {Path(p["path"]).suffix.lower() for p in paths}
        if any(e in TEXT_EXT_ALLOWLIST for e in exts):
            continue
        pollution.append({
            "file_id": fid,
            "mime": fr["mime"],
            "size": fr["size"],
            "vector_count": fr["vector_count"] or 0,
            "sample_path": paths[0]["path"],
            "exts": sorted(exts),
            "n_paths": len(paths),
        })
    return pollution


def print_report(items: list[dict]) -> None:
    if not items:
        print("No polluted entries found ✓")
        return
    print(f"\n{'file_id (first 12)':14} {'ext':12} {'chunks':>7} {'size':>8}  sample path")
    print("-" * 100)
    total_chunks = 0
    total_size = 0
    for it in items:
        ext_disp = ",".join(it["exts"])[:12]
        total_chunks += it["vector_count"]
        total_size += it["size"]
        print(f"{it['file_id'][:12]:14} {ext_disp:12} {it['vector_count']:>7} "
              f"{fmt_bytes(it['size']):>8}  ...{it['sample_path'][-55:]}")
    print("-" * 100)
    print(f"Total: {len(items)} file_ids, "
          f"{total_chunks} chunks, original files {fmt_bytes(total_size)}")


def apply_cleanup(conn: sqlite3.Connection, items: list[dict]) -> None:
    from parser.qdrant_store import QdrantStore

    qstore = QdrantStore(QDRANT_URL)

    deleted_vectors = 0
    for it in items:
        fid = it["file_id"]
        print(f"  deleting {fid[:12]} ({it['vector_count']} chunks, "
              f"...{it['sample_path'][-50:]})")
        # 1) qdrant
        qstore.delete_file(file_id=fid)
        deleted_vectors += it["vector_count"]
        # 2) sqlite - delete file_paths first (safe even without a foreign key), then file_records, then parse_jobs
        conn.execute("DELETE FROM file_paths WHERE file_id = ?", (fid,))
        conn.execute("DELETE FROM file_records WHERE file_id = ?", (fid,))
    # parse_jobs has no file_id column, so clean up by path: delete jobs whose
    # extension isn't in the allowlist, using sqlite's substr + lower.
    paths_to_purge = [it["sample_path"] for it in items]
    # more robust: iterate parse_jobs and decide on the Python side
    job_rows = list(conn.execute(
        "SELECT id, path FROM parse_jobs WHERE done_at IS NOT NULL"
    ))
    purge_ids = []
    for r in job_rows:
        ext = Path(r["path"]).suffix.lower()
        if ext and ext not in TEXT_EXT_ALLOWLIST:
            purge_ids.append(r["id"])
    if purge_ids:
        conn.executemany("DELETE FROM parse_jobs WHERE id = ?",
                         [(i,) for i in purge_ids])
        print(f"  cleaned up {len(purge_ids)} binary history records in parse_jobs")
    conn.commit()
    print(f"\n✓ deleted {len(items)} file_records, ~{deleted_vectors} qdrant points")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is dry-run)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"parser.db not found: {args.db}", file=sys.stderr)
        return 1

    # confirm the service is stopped
    if args.apply:
        rc = os.system("systemctl is-active --quiet nimoos-parser.service")
        if rc == 0:
            print("nimoos-parser is running, stop it first:", file=sys.stderr)
            print("    sudo systemctl stop nimoos-parser.service", file=sys.stderr)
            return 2

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    print(f"allowlist size: {len(TEXT_EXT_ALLOWLIST)} extensions")
    print(f"parser.db: {args.db}")
    print(f"qdrant: {QDRANT_URL}")
    print(f"mode: {'APPLY (will actually delete)' if args.apply else 'DRY-RUN (print only)'}")

    items = find_pollution(conn)
    print_report(items)

    if not items:
        return 0

    if not args.apply:
        print("\nTo actually delete, add --apply, and remember to stop nimoos-parser first.")
        return 0

    print("\n>>> starting cleanup <<<")
    apply_cleanup(conn, items)
    print("\nSuggested next steps:")
    print("  sudo systemctl start nimoos-parser.service")
    print("  curl -s http://127.0.0.1:6333/collections/text_chunks | "
          "python3 -m json.tool | head")
    return 0


if __name__ == "__main__":
    sys.exit(main())
