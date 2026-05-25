import sqlite3
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
  updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS parser_state (
  id          INTEGER PRIMARY KEY CHECK (id = 1),
  paused      INTEGER NOT NULL DEFAULT 0,
  concurrency INTEGER NOT NULL DEFAULT 2,
  updated_at  INTEGER NOT NULL
);
INSERT OR IGNORE INTO parser_state (id, paused, concurrency, updated_at)
VALUES (1, 0, 2, strftime('%s','now')*1000);
"""


def init_db(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.executescript(SCHEMA_SQL)
    # ensure wiki_cursor singleton
    conn.execute(
        "INSERT OR IGNORE INTO wiki_cursor(id, since_ms, updated_at) VALUES (1, 0, ?)",
        (int(time.time() * 1000),),
    )
    return conn
