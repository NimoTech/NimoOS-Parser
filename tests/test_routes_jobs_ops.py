import time
import pytest


def _enqueue(client_conn, *, op="index", root_id="r1", path="/x.md"):
    from parser.repo_jobs import enqueue_job
    return enqueue_job(client_conn, root_id=root_id, op=op, path=path,
                        sub_modality="text", priority=0,
                        now_ms=int(time.time() * 1000))


def test_delete_pending_job(client):
    from parser.main import app_state
    job_id = _enqueue(app_state.conn, path="/cancelme.md")

    r = client.delete(f"/v1/parser/jobs/{job_id}")
    assert r.status_code == 204

    # Verify gone
    r2 = client.get("/v1/parser/jobs?status=pending&limit=200")
    ids = {j["id"] for j in r2.json()["jobs"]}
    assert job_id not in ids


def test_delete_running_job_rejected(client):
    """V1 does not support cancelling running jobs."""
    from parser.main import app_state
    job_id = _enqueue(app_state.conn, path="/running.md")
    app_state.conn.execute(
        "UPDATE parse_jobs SET locked_until = ? WHERE id = ?",
        (int(time.time() * 1000) + 300_000, job_id),
    )
    r = client.delete(f"/v1/parser/jobs/{job_id}")
    assert r.status_code == 409
    assert "running" in r.json()["detail"].lower()


def test_delete_unknown_job_returns_404(client):
    r = client.delete("/v1/parser/jobs/99999999")
    assert r.status_code == 404


def test_clear_failed_jobs(client):
    from parser.main import app_state
    # Make 2 failed jobs
    for p in ("/f1.md", "/f2.md"):
        jid = _enqueue(app_state.conn, path=p)
        app_state.conn.execute(
            "UPDATE parse_jobs SET done_at = ?, last_error = 'boom', attempts = 5 "
            "WHERE id = ?", (int(time.time() * 1000), jid),
        )

    r = client.post("/v1/parser/jobs/clear-failed")
    assert r.status_code == 200
    assert r.json()["cleared"] == 2

    r2 = client.get("/v1/parser/jobs?status=failed&limit=200")
    assert r2.json()["jobs"] == []
