import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Iterable

log = logging.getLogger("parser.pipeline_text")

from parser.chunk_text import chunk_markdown, chunk_plain, chunk_source
from parser.pipeline_base import IdentityResolver, ResolveOutcome
from parser.repo_records import (
    upsert_file_record, get_file_record,
    list_paths_for_file, count_paths_for_file_in_root,
)


_MD_EXT = {".md", ".markdown"}
_SOURCE_EXT = {".py", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".java",
               ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
               ".kt", ".scala", ".sh", ".bash", ".sql", ".lua"}


class TextPipeline:
    def __init__(self, conn: sqlite3.Connection, *, qstore, embedder,
                 parser_version: str) -> None:
        self.conn = conn
        self.qstore = qstore
        self.embedder = embedder
        self.parser_version = parser_version
        self.active = {"text": embedder.version}

    def index_file(self, *, root_id: str, path: str, now_ms: int) -> None:
        resolver = IdentityResolver(
            self.conn, parser_version=self.parser_version,
            active_modalities=self.active,
        )
        outcome = resolver.resolve(root_id=root_id, path=path, now_ms=now_ms)

        # handle old orphan first (if any)
        if outcome.old_orphan_file_id:
            self.qstore.tombstone_file(
                file_id=outcome.old_orphan_file_id, tombstoned_at=now_ms,
            )
        elif outcome.old_root_lost:
            old_fid, root_lost = outcome.old_root_lost
            remaining = self._collect_root_ids(old_fid)
            self.qstore.set_root_ids_for_file(
                file_id=old_fid, root_ids=remaining,
            )

        if outcome.action == ResolveOutcome.APPEND_ROOT_ONLY:
            self._append_root(outcome.new_file_id)
            return
        if outcome.action == ResolveOutcome.REVIVE:
            self._append_root(outcome.new_file_id)
            return
        # FULL_INDEX
        self._run_full(root_id=root_id, path=path, file_id=outcome.new_file_id,
                       sha256_full=outcome.new_sha256, now_ms=now_ms)

    def _append_root(self, file_id: str) -> None:
        all_roots = self._collect_root_ids(file_id)
        self.qstore.set_root_ids_for_file(
            file_id=file_id, root_ids=all_roots,
        )

    def _collect_root_ids(self, file_id: str) -> list[str]:
        rows = list_paths_for_file(self.conn, file_id)
        return sorted({r["root_id"] for r in rows})

    def _run_full(self, *, root_id: str, path: str, file_id: str,
                  sha256_full: str, now_ms: int) -> None:
        from parser.docling_extractor import DoclingExtractor, is_docling_format
        from parser import repo_allowlist
        size = os.path.getsize(path)
        ext = Path(path).suffix.lower()

        # 防御:wiki_consumer 已经按 allowlist 过滤过事件,但 rescan / 历史 job
        # 也走这里。allowlist 由 DB 持有,这是单一真理源 —— wiki_consumer 也走
        # 同一个函数,确保两处永远不会出现"一处过滤一处没过滤"的污染。
        if not repo_allowlist.is_path_indexable(self.conn, root_id=root_id,
                                                 path=path):
            log.warning("skipped: not indexable per allowlist (path=%s)", path)
            return

        if is_docling_format(ext):
            # PDF/DOCX/PPTX/XLSX/HTML → docling → markdown → chunk_markdown.
            # OCR toggled via parser_state.ocr_enabled — same singleton
            # extractor reloads on change.
            from parser.repo_state import get_state
            ocr = get_state(self.conn).get("ocr_enabled", False)
            try:
                text = DoclingExtractor.load(ocr=ocr).to_markdown(path)
                chunks = chunk_markdown(text, min_tokens=20)
                mime = f"text/markdown+docling/{ext.lstrip('.')}"
            except Exception:
                with open(path, "rb") as f:
                    text = f.read().decode("utf-8", errors="replace")
                chunks = chunk_plain(text, min_tokens=20)
                mime = "text/plain"
        elif ext in _MD_EXT:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            chunks = chunk_markdown(text, min_tokens=20)
            mime = "text/markdown"
        elif ext in _SOURCE_EXT:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            chunks = chunk_source(text, min_tokens=10)
            mime = "text/x-source"
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            chunks = chunk_plain(text, min_tokens=20)
            mime = "text/plain"
        if not chunks:
            upsert_file_record(
                self.conn, file_id=file_id, sha256_full=sha256_full,
                size=size, mime=mime,
                modalities_done={"text": self.embedder.version},
                parser_version=self.parser_version, indexed_at=now_ms,
            )
            return
        embeddings = self.embedder.embed_text([c["text"] for c in chunks])
        root_ids = self._collect_root_ids(file_id)
        if not root_ids:
            root_ids = [root_id]
        points = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_OID, f"{file_id}:body:{chunk['chunk_no']}"
            ))
            points.append({
                "id": point_id,
                "dense": emb["dense"],
                "sparse": emb["sparse"],
                "payload": {
                    "file_id": file_id,
                    "root_ids": root_ids,
                    "kind": "body",
                    "mime": mime,
                    "chunk_no": chunk["chunk_no"],
                    "text": chunk["text"],
                    "offset_start": chunk["offset_start"],
                    "offset_end": chunk["offset_end"],
                    "lang": "auto",
                    "parser_version": self.parser_version,
                    "embed_model_version": self.embedder.version,
                    "source_model_version": "",
                    "indexed_at": now_ms,
                    "tombstoned_at": None,
                },
            })
        # if this file_id existed before (e.g. parser_version drift), wipe old
        prev = get_file_record(self.conn, file_id)
        if prev is not None and prev["modalities_done"] not in ("{}", ""):
            self.qstore.delete_file(file_id=file_id)
        self.qstore.upsert_text_chunks(points)
        upsert_file_record(
            self.conn, file_id=file_id, sha256_full=sha256_full,
            size=size, mime=mime,
            modalities_done={"text": self.embedder.version},
            parser_version=self.parser_version, indexed_at=now_ms,
        )
        self.conn.execute(
            "UPDATE file_records SET vector_count = ? WHERE file_id = ?",
            (len(points), file_id),
        )

    def delete_path(self, *, root_id: str, path: str, now_ms: int) -> None:
        row = self.conn.execute(
            "SELECT file_id FROM file_paths WHERE root_id = ? AND path = ?",
            (root_id, path),
        ).fetchone()
        if not row:
            return
        file_id = row["file_id"]
        self.conn.execute(
            "DELETE FROM file_paths WHERE root_id = ? AND path = ?",
            (root_id, path),
        )
        remaining = self._collect_root_ids(file_id)
        if not remaining:
            from parser.repo_records import set_tombstone
            set_tombstone(self.conn, file_id=file_id, at_ms=now_ms)
            self.qstore.tombstone_file(file_id=file_id, tombstoned_at=now_ms)
        else:
            self.qstore.set_root_ids_for_file(
                file_id=file_id, root_ids=remaining,
            )
