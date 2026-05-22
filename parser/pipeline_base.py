import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from parser.repo_records import (
    get_file_record, list_paths_for_file, upsert_file_path, upsert_file_record,
    count_paths_for_file_in_root, set_tombstone, clear_tombstone,
)


class ResolveOutcome:
    FULL_INDEX = "full_index"
    REVIVE = "revive"
    APPEND_ROOT_ONLY = "append_root_only"
    SKIP = "skip"


@dataclass
class ResolveResult:
    action: str
    new_file_id: Optional[str]
    new_sha256: Optional[str]
    old_orphan_file_id: Optional[str]
    old_root_lost: Optional[tuple[str, str]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


class IdentityResolver:
    """Resolves a (root_id, path) event into the next action for the pipeline.

    Mirrors §5.1 of the RAG spec: short-circuit on identity match, soft-tombstone
    when a modify orphans the old file_id, revive on sha256 match against an
    existing tombstoned record.
    """

    def __init__(self, conn, *, parser_version: str,
                 active_modalities: dict[str, str]) -> None:
        self.conn = conn
        self.parser_version = parser_version
        self.active_modalities = active_modalities

    def resolve(self, *, root_id: str, path: str, now_ms: int) -> ResolveResult:
        st = os.stat(path)
        full = sha256_file(Path(path))
        new_fid = full[:32]
        old = self.conn.execute(
            "SELECT file_id FROM file_paths WHERE root_id = ? AND path = ?",
            (root_id, path),
        ).fetchone()
        old_fid = old["file_id"] if old else None

        old_orphan = None
        old_root_lost = None
        if old_fid and old_fid != new_fid:
            # Ensure new_fid has a file_record before repointing (FK constraint).
            if get_file_record(self.conn, new_fid) is None:
                upsert_file_record(
                    self.conn, file_id=new_fid, sha256_full=full,
                    size=st.st_size, mime=_guess_mime(path),
                    modalities_done={}, parser_version=self.parser_version,
                    indexed_at=now_ms,
                )
            upsert_file_path(self.conn, root_id=root_id, path=path,
                             file_id=new_fid, mtime_ms=int(st.st_mtime * 1000))
            in_root = count_paths_for_file_in_root(self.conn, old_fid, root_id)
            if in_root == 0:
                old_root_lost = (old_fid, root_id)
            any_path = self.conn.execute(
                "SELECT 1 FROM file_paths WHERE file_id = ? LIMIT 1", (old_fid,)
            ).fetchone()
            if not any_path:
                set_tombstone(self.conn, file_id=old_fid, at_ms=now_ms)
                old_orphan = old_fid

        rec = get_file_record(self.conn, new_fid)
        if rec is None:
            # Insert a provisional file_record so the FK constraint is satisfied.
            # The actual indexing worker will overwrite this with real metadata.
            upsert_file_record(
                self.conn, file_id=new_fid, sha256_full=full,
                size=st.st_size, mime=_guess_mime(path),
                modalities_done={}, parser_version=self.parser_version,
                indexed_at=now_ms,
            )
            upsert_file_path(self.conn, root_id=root_id, path=path,
                             file_id=new_fid, mtime_ms=int(st.st_mtime * 1000))
            return ResolveResult(action=ResolveOutcome.FULL_INDEX,
                                 new_file_id=new_fid, new_sha256=full,
                                 old_orphan_file_id=old_orphan,
                                 old_root_lost=old_root_lost)

        upsert_file_path(self.conn, root_id=root_id, path=path,
                         file_id=new_fid, mtime_ms=int(st.st_mtime * 1000))
        was_tombstoned = rec["tombstoned_at"] is not None
        rec_modalities = json.loads(rec["modalities_done"])
        same_models = all(
            rec_modalities.get(m) == v for m, v in self.active_modalities.items()
        )
        same_parser = rec["parser_version"] == self.parser_version
        if was_tombstoned:
            clear_tombstone(self.conn, file_id=new_fid)
            if same_models and same_parser:
                return ResolveResult(action=ResolveOutcome.REVIVE,
                                     new_file_id=new_fid, new_sha256=full,
                                     old_orphan_file_id=old_orphan,
                                     old_root_lost=old_root_lost)
            return ResolveResult(action=ResolveOutcome.FULL_INDEX,
                                 new_file_id=new_fid, new_sha256=full,
                                 old_orphan_file_id=old_orphan,
                                 old_root_lost=old_root_lost)
        if same_models and same_parser:
            return ResolveResult(action=ResolveOutcome.APPEND_ROOT_ONLY,
                                 new_file_id=new_fid, new_sha256=full,
                                 old_orphan_file_id=old_orphan,
                                 old_root_lost=old_root_lost)
        return ResolveResult(action=ResolveOutcome.FULL_INDEX,
                             new_file_id=new_fid, new_sha256=full,
                             old_orphan_file_id=old_orphan,
                             old_root_lost=old_root_lost)
