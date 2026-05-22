import time

from parser.db import init_db
from parser.repo_jobs import enqueue_job, dequeue_job, fail_job


def test_list_jobs_pending(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    enqueue_job(conn, root_id="r", path="/a", op="index",
                priority=100, now_ms=100)
    monkeypatch.setattr("parser.routes.jobs.get_conn", lambda: conn)
    r = client.get("/v1/parser/jobs?status=pending&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["path"] == "/a"


def test_retry_failed(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    enqueue_job(conn, root_id="r", path="/a", op="index",
                priority=100, now_ms=100)
    j = dequeue_job(conn, lease_s=10, now_ms=200)
    fail_job(conn, job_id=j["id"], error="boom", now_ms=300, max_attempts=1)
    monkeypatch.setattr("parser.routes.jobs.get_conn", lambda: conn)
    r = client.post("/v1/parser/jobs/retry", json={})
    assert r.status_code == 200
    assert r.json()["retried"] == 1
