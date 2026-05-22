import pytest

from parser.db import init_db
from parser.repo_jobs import (
    enqueue_job, dequeue_job, complete_job, fail_job, list_jobs,
    retry_failed_jobs,
)


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def test_enqueue_dequeue_in_priority_order(conn):
    enqueue_job(conn, root_id="r", path="/a", op="index", priority=200, now_ms=100)
    enqueue_job(conn, root_id="r", path="/b", op="index", priority=50, now_ms=101)
    enqueue_job(conn, root_id="r", path="/c", op="index", priority=100, now_ms=102)
    j1 = dequeue_job(conn, lease_s=10, now_ms=200)
    assert j1["path"] == "/b"
    j2 = dequeue_job(conn, lease_s=10, now_ms=201)
    assert j2["path"] == "/c"


def test_dequeue_respects_lease(conn):
    enqueue_job(conn, root_id="r", path="/a", op="index", priority=100, now_ms=100)
    j = dequeue_job(conn, lease_s=10, now_ms=200)
    assert j["path"] == "/a"
    # second dequeue inside lease window: nothing returned
    j2 = dequeue_job(conn, lease_s=10, now_ms=300)
    assert j2 is None
    # after lease expires: re-leased
    j3 = dequeue_job(conn, lease_s=10, now_ms=200 + 10_000 + 1)
    assert j3 is not None
    assert j3["path"] == "/a"


def test_complete_job_removes_from_queue(conn):
    enqueue_job(conn, root_id="r", path="/a", op="index", priority=100, now_ms=100)
    j = dequeue_job(conn, lease_s=10, now_ms=200)
    complete_job(conn, job_id=j["id"], now_ms=300)
    j2 = dequeue_job(conn, lease_s=10, now_ms=400)
    assert j2 is None


def test_fail_job_increments_attempts(conn):
    enqueue_job(conn, root_id="r", path="/a", op="index", priority=100, now_ms=100)
    j = dequeue_job(conn, lease_s=10, now_ms=200)
    fail_job(conn, job_id=j["id"], error="boom", now_ms=300)
    rows = list_jobs(conn, status="failed", limit=10)
    assert len(rows) == 0  # attempts=1, not yet failed terminally
    fail_job(conn, job_id=j["id"], error="boom", now_ms=400, max_attempts=2)
    rows = list_jobs(conn, status="failed", limit=10)
    assert len(rows) == 1
    assert rows[0]["last_error"] == "boom"


def test_retry_failed_jobs(conn):
    enqueue_job(conn, root_id="r", path="/a", op="index", priority=100, now_ms=100)
    j = dequeue_job(conn, lease_s=10, now_ms=200)
    fail_job(conn, job_id=j["id"], error="boom", now_ms=300, max_attempts=1)
    n = retry_failed_jobs(conn, file_ids=None, now_ms=1000)
    assert n == 1
    j2 = dequeue_job(conn, lease_s=10, now_ms=1100)
    assert j2 is not None
