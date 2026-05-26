#!/usr/bin/env python3
"""backfill_mtime.py — Add mtime_ms to existing Qdrant chunks that pre-date
the mtime_ms payload addition.

Strategy: scroll all text_chunks → for each unique file_id, look up its
file_paths row → set_payload(mtime_ms=<earliest mtime among paths>) on
all that file's chunks. Skip points that already have mtime_ms.

Usage:
  sudo systemctl stop nimoos-parser.service
  sudo /opt/nimoos-parser/venv/bin/python3 \\
    /home/nimo/nimoos/NimoOS-Parser/backfill_mtime.py [--apply]
  sudo systemctl start nimoos-parser.service
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DB_PATH = "/var/lib/nimoos/parser/parser.db"
QDRANT_URL = "http://127.0.0.1:6333"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write (default: dry-run)")
    args = ap.parse_args()

    from qdrant_client import QdrantClient

    if args.apply:
        rc = os.system("systemctl is-active --quiet nimoos-parser.service")
        if rc == 0:
            print("nimoos-parser is running, stop it first:", file=sys.stderr)
            print("    sudo systemctl stop nimoos-parser.service", file=sys.stderr)
            return 2

    client = QdrantClient(url=QDRANT_URL)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Look up mtime by file_id (use the smallest known mtime across all paths
    # for a deterministic value — files that exist in multiple roots get the
    # earliest registered mtime)
    file_mtime: dict[str, int] = {}
    for r in conn.execute(
        "SELECT fp.file_id, MIN(fp.mtime_ms) as mt "
        "FROM file_paths fp GROUP BY fp.file_id"
    ):
        file_mtime[r["file_id"]] = r["mt"]
    print(f"Found {len(file_mtime)} file_ids with paths in DB")

    # Scroll text_chunks
    offset = None
    seen = 0
    needs = 0
    by_file: dict[str, list] = {}
    while True:
        points, offset = client.scroll(
            collection_name="text_chunks",
            with_payload=True, with_vectors=False,
            limit=512, offset=offset,
        )
        if not points:
            break
        for p in points:
            seen += 1
            if p.payload and "mtime_ms" in p.payload and p.payload["mtime_ms"]:
                continue
            fid = p.payload.get("file_id") if p.payload else None
            if not fid:
                continue
            by_file.setdefault(fid, []).append(p.id)
            needs += 1
        if offset is None:
            break

    print(f"Scrolled {seen} points, {needs} need mtime_ms backfill across "
          f"{len(by_file)} file_ids")

    if not args.apply:
        for fid, pids in list(by_file.items())[:5]:
            print(f"  sample: file_id={fid[:12]}  {len(pids)} chunks  "
                  f"mtime={file_mtime.get(fid)}")
        print("\n(dry-run; re-run with --apply to write)")
        return 0

    written = 0
    for fid, pids in by_file.items():
        mtime = file_mtime.get(fid)
        if mtime is None:
            continue
        client.set_payload(
            collection_name="text_chunks",
            payload={"mtime_ms": mtime},
            points=pids,
            wait=True,
        )
        written += len(pids)
    print(f"\nBackfilled {written} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
