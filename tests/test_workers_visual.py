import json
import sqlite3

import pytest

from parser.db import init_db
from parser.repo_jobs import enqueue_job
from parser.workers import WorkerPool


class RecordingVisual:
    def __init__(self):
        self.ingests, self.deletes = [], []
    def ingest_asset(self, **kw):
        self.ingests.append(kw)


@pytest.fixture
def pool(tmp_path):
    conn = init_db(tmp_path / "t.db")
    vp = RecordingVisual()
    p = WorkerPool(conn, text_pipeline=None, visual_pipeline=vp,
                   concurrency=1)
    return p, vp, conn


def _job_row(conn, jid):
    return conn.execute("SELECT * FROM parse_jobs WHERE id=?", (jid,)).fetchone()


def test_process_visual_ingest(pool, tmp_path):
    p, vp, conn = pool
    img = tmp_path / "x.jpg"; img.write_bytes(b"j")
    jid = enqueue_job(conn, root_id="photos", path=str(img),
                      op="visual_ingest", priority=200,
                      sub_modality=json.dumps({"asset_id": "a1",
                                               "mime": "image/jpeg",
                                               "meta": {"place": "Tokyo"}}),
                      now_ms=1)
    p._process(_job_row(conn, jid))
    (call,) = vp.ingests
    assert call["source"] == "photos" and call["asset_id"] == "a1"
    assert call["image_path"] == str(img) and call["meta"] == {"place": "Tokyo"}


def test_process_visual_without_pipeline_raises(pool, tmp_path):
    p, _, conn = pool
    p.visual_pipeline = None
    jid = enqueue_job(conn, root_id="photos", path="/x.jpg",
                      op="visual_ingest", sub_modality="{}", now_ms=1)
    with pytest.raises(RuntimeError):
        p._process(_job_row(conn, jid))


def test_unknown_op_still_raises(pool, tmp_path):
    p, _, conn = pool
    jid = enqueue_job(conn, root_id="r", path="/y", op="bogus", now_ms=1)
    with pytest.raises(ValueError):
        p._process(_job_row(conn, jid))
