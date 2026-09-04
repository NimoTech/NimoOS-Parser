import sqlite3
import threading
import time
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS file_records (
  file_id          TEXT PRIMARY KEY,
  sha256_full      TEXT NOT NULL,
  size             INTEGER NOT NULL,
  mime             TEXT NOT NULL,
  modalities_done  TEXT NOT NULL,
  parser_version   TEXT NOT NULL,
  indexed_at       INTEGER NOT NULL,
  tombstoned_at    INTEGER,
  vector_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_file_records_tombstone
  ON file_records(tombstoned_at) WHERE tombstoned_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS file_paths (
  root_id     TEXT NOT NULL,
  path        TEXT NOT NULL,
  file_id     TEXT NOT NULL,
  mtime_ms    INTEGER NOT NULL,
  PRIMARY KEY(root_id, path),
  FOREIGN KEY(file_id) REFERENCES file_records(file_id)
);
CREATE INDEX IF NOT EXISTS idx_file_paths_file ON file_paths(file_id);

CREATE TABLE IF NOT EXISTS parse_jobs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  root_id       TEXT NOT NULL,
  path          TEXT NOT NULL,
  op            TEXT NOT NULL,
  sub_modality  TEXT,
  priority      INTEGER NOT NULL DEFAULT 100,
  attempts      INTEGER NOT NULL DEFAULT 0,
  last_error    TEXT,
  locked_until  INTEGER,
  created_at    INTEGER NOT NULL,
  picked_at     INTEGER,
  done_at       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_parse_jobs_q
  ON parse_jobs(done_at, priority, id);

CREATE TABLE IF NOT EXISTS model_versions (
  name          TEXT NOT NULL,
  version       TEXT NOT NULL,
  modality      TEXT NOT NULL,
  dim           INTEGER,
  active        INTEGER NOT NULL DEFAULT 1,
  registered_at INTEGER NOT NULL,
  PRIMARY KEY(name, version)
);

CREATE TABLE IF NOT EXISTS wiki_cursor (
  id          INTEGER PRIMARY KEY CHECK(id = 1),
  since_ms    INTEGER NOT NULL DEFAULT 0,
  last_seq    INTEGER NOT NULL DEFAULT 0,
  updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS parser_state (
  id          INTEGER PRIMARY KEY CHECK (id = 1),
  paused      INTEGER NOT NULL DEFAULT 0,
  concurrency INTEGER NOT NULL DEFAULT 2,
  device      TEXT NOT NULL DEFAULT 'auto',
  ocr_enabled INTEGER NOT NULL DEFAULT 0,
  ocr_model   TEXT NOT NULL DEFAULT '',
  cursor_gap  TEXT,
  verify_last TEXT,
  updated_at  INTEGER NOT NULL
);
INSERT OR IGNORE INTO parser_state
  (id, paused, concurrency, device, ocr_enabled, ocr_model, updated_at)
VALUES (1, 0, 2, 'auto', 0, '', strftime('%s','now')*1000);

CREATE TABLE IF NOT EXISTS allowlist_extensions (
  ext TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 1,
  source TEXT NOT NULL DEFAULT 'default',
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS allowlist_folders (
  id TEXT PRIMARY KEY,
  root_id TEXT NOT NULL,
  path_glob TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN ('allow','deny')),
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_allowlist_folders_root
  ON allowlist_folders(root_id);
"""

# Column-add migrations for parser_state tables created before each new
# column existed. Run BEFORE executescript so INSERT OR IGNORE's column
# list doesn't reference a column SQLite hasn't added yet.
_MIGRATION_DEVICE = "ALTER TABLE parser_state ADD COLUMN device TEXT NOT NULL DEFAULT 'auto';"
_MIGRATION_OCR = "ALTER TABLE parser_state ADD COLUMN ocr_enabled INTEGER NOT NULL DEFAULT 0;"
_MIGRATION_OCR_MODEL = (
    "ALTER TABLE parser_state ADD COLUMN ocr_model TEXT NOT NULL DEFAULT '';"
)
_MIGRATION_CURSOR_GAP = "ALTER TABLE parser_state ADD COLUMN cursor_gap TEXT;"
_MIGRATION_VERIFY_LAST = "ALTER TABLE parser_state ADD COLUMN verify_last TEXT;"
_MIGRATION_WIKI_CURSOR_SEQ = (
    "ALTER TABLE wiki_cursor ADD COLUMN last_seq INTEGER NOT NULL DEFAULT 0;"
)


class LockedConnection(sqlite3.Connection):
    """sqlite3.Connection serialized by a process-wide re-entrant lock.

    One connection is shared by every worker thread (asyncio.to_thread), the
    wiki poll loop and FastAPI's threadpool. Python's sqlite3 connections are
    not safe to share that way (threadsafety=1): interleaved statements from
    two threads corrupt each other's transaction state ("cannot start a
    transaction within a transaction", "cannot commit - no transaction is
    active"). Every statement-level call takes `lock`; code that needs a
    multi-statement transaction holds `lock` itself for the whole span (see
    repo_jobs.dequeue_job). The lock is re-entrant so those two compose.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()

    def execute(self, *args, **kwargs):
        with self.lock:
            return super().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        with self.lock:
            return super().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self.lock:
            return super().executescript(*args, **kwargs)

    def commit(self):
        with self.lock:
            return super().commit()

    def rollback(self):
        with self.lock:
            return super().rollback()


def init_db(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False,
                           factory=LockedConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    # Run column-add migrations BEFORE executescript, because SCHEMA_SQL
    # contains an `INSERT OR IGNORE` that names the new column — if the
    # column hasn't been backfilled on an existing DB, that insert fails
    # and aborts the whole executescript.
    cols = {
        r[1] for r in conn.execute("PRAGMA table_info(parser_state)").fetchall()
    }
    if cols and "device" not in cols:
        conn.execute(_MIGRATION_DEVICE)
    if cols and "ocr_enabled" not in cols:
        conn.execute(_MIGRATION_OCR)
    if cols and "ocr_model" not in cols:
        conn.execute(_MIGRATION_OCR_MODEL)
    if cols and "cursor_gap" not in cols:
        conn.execute(_MIGRATION_CURSOR_GAP)
    if cols and "verify_last" not in cols:
        conn.execute(_MIGRATION_VERIFY_LAST)
    wc_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(wiki_cursor)").fetchall()
    }
    if wc_cols and "last_seq" not in wc_cols:
        conn.execute(_MIGRATION_WIKI_CURSOR_SEQ)

    conn.executescript(SCHEMA_SQL)
    # ensure wiki_cursor singleton
    conn.execute(
        "INSERT OR IGNORE INTO wiki_cursor(id, since_ms, updated_at) VALUES (1, 0, ?)",
        (int(time.time() * 1000),),
    )
    # Seed allowlist_extensions from the legacy constant on first run.
    # On subsequent runs, INSERT OR IGNORE keeps user toggles intact.
    from parser.wiki_consumer import TEXT_EXT_ALLOWLIST
    now_ms = int(time.time() * 1000)
    conn.executemany(
        "INSERT OR IGNORE INTO allowlist_extensions(ext, enabled, source, updated_at) "
        "VALUES (?, 1, 'default', ?)",
        [(ext, now_ms) for ext in sorted(TEXT_EXT_ALLOWLIST)],
    )

    # ---- migrations (idempotent) ----
    # Add file_records.last_error column. SQLite has no `ADD COLUMN IF NOT EXISTS`,
    # so we catch the duplicate-column OperationalError. This is the standard
    # SQLite migration idiom.
    try:
        conn.execute("ALTER TABLE file_records ADD COLUMN last_error TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise

    # Indexes for the file list + reindex API (2026-05-28).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_records_indexed_at "
        "ON file_records(indexed_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_records_mime "
        "ON file_records(mime)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_records_last_error "
        "ON file_records(last_error) WHERE last_error IS NOT NULL"
    )
    conn.commit()

    return conn
