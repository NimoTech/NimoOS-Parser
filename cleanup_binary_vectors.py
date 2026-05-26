#!/usr/bin/env python3
"""
cleanup_binary_vectors.py — 清掉历史污染的 binary 文件向量

背景:wiki_consumer 的 TEXT_EXT_ALLOWLIST 是后期才加的,在那之前
.sql.gz / .MOV / .jpeg 等二进制文件被 pipeline_text 的 else 兜底当成
text/plain 直接 decode("utf-8", errors="replace") + chunk,污染了 Qdrant。

本脚本扫 parser.db,找出 file_records 里所有 file_id 的全部 paths
都不在 TEXT_EXT_ALLOWLIST 的(即:这个文件本质就不应该被索引),
然后:
  - dry-run(默认):只打印,不动
  - --apply:从 Qdrant 删点 → 删 file_records / file_paths / parse_jobs

数据安全:
  - 多 path 中只要有一条命中 allowlist 就保留(说明同样内容也曾以合法
    扩展名出现,可能是用户真的索引过)
  - 用 file_id 删 Qdrant 不会误删别的文件
  - 操作前请确保 nimoos-parser 服务停掉,避免 race

用法:
  sudo /opt/nimoos-parser/venv/bin/python3 \\
    /home/nimo/nimoos/NimoOS-Parser/cleanup_binary_vectors.py [--apply]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# 让脚本能 import parser.* —— 脚本放在 repo 根
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
    """返回 [{file_id, sample_path, mime, vector_count, all_paths}]
    其中 all_paths 是 file_id 下所有 (root_id, path),仅当全部扩展名
    都不在 allowlist 才视为污染。"""
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
        # 全部 path 的扩展名都得 NOT IN allowlist 才算污染
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
        print("没有命中污染条目 ✓")
        return
    print(f"\n{'file_id (前 12)':14} {'扩展':12} {'chunks':>7} {'size':>8}  path 样例")
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
    print(f"合计:{len(items)} 个 file_id, "
          f"{total_chunks} chunks, 原文件 {fmt_bytes(total_size)}")


def apply_cleanup(conn: sqlite3.Connection, items: list[dict]) -> None:
    from parser.qdrant_store import QdrantStore

    qstore = QdrantStore(QDRANT_URL)

    deleted_vectors = 0
    for it in items:
        fid = it["file_id"]
        print(f"  删 {fid[:12]} ({it['vector_count']} chunks, "
              f"...{it['sample_path'][-50:]})")
        # 1) qdrant
        qstore.delete_file(file_id=fid)
        deleted_vectors += it["vector_count"]
        # 2) sqlite — file_paths 先删(没外键也安全),再 file_records,再 parse_jobs
        conn.execute("DELETE FROM file_paths WHERE file_id = ?", (fid,))
        conn.execute("DELETE FROM file_records WHERE file_id = ?", (fid,))
    # parse_jobs 没有 file_id 列,按 path 清理:删那些扩展名不在 allowlist 的 job
    # 用 sqlite 的 substr + lower 处理。
    paths_to_purge = [it["sample_path"] for it in items]
    # 更稳:遍历 parse_jobs,Python 侧判定
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
        print(f"  清掉 parse_jobs 里 {len(purge_ids)} 条 binary 历史记录")
    conn.commit()
    print(f"\n✓ 删了 {len(items)} 个 file_records, ~{deleted_vectors} qdrant points")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="真删(默认 dry-run)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"找不到 parser.db: {args.db}", file=sys.stderr)
        return 1

    # 确认服务停掉
    if args.apply:
        rc = os.system("systemctl is-active --quiet nimoos-parser.service")
        if rc == 0:
            print("nimoos-parser 在跑,先停掉:", file=sys.stderr)
            print("    sudo systemctl stop nimoos-parser.service", file=sys.stderr)
            return 2

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    print(f"allowlist 大小: {len(TEXT_EXT_ALLOWLIST)} 个扩展名")
    print(f"parser.db: {args.db}")
    print(f"qdrant: {QDRANT_URL}")
    print(f"模式: {'APPLY (会真删)' if args.apply else 'DRY-RUN (只打印)'}")

    items = find_pollution(conn)
    print_report(items)

    if not items:
        return 0

    if not args.apply:
        print("\n要真删请加 --apply,记得先 stop nimoos-parser.")
        return 0

    print("\n>>> 开始清理 <<<")
    apply_cleanup(conn, items)
    print("\n下一步建议:")
    print("  sudo systemctl start nimoos-parser.service")
    print("  curl -s http://127.0.0.1:6333/collections/text_chunks | "
          "python3 -m json.tool | head")
    return 0


if __name__ == "__main__":
    sys.exit(main())
