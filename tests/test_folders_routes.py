import time

from parser.repo_jobs import enqueue_job


def _seed(conn):
    now = int(time.time() * 1000)
    enqueue_job(conn, root_id="r1", path="/a/b/x.pdf", op="index",
                priority=100, now_ms=now)
    enqueue_job(conn, root_id="r1", path="/a/b/y.pdf", op="index",
                priority=100, now_ms=now)
    enqueue_job(conn, root_id="r1", path="/a/c/z.pdf", op="index",
                priority=100, now_ms=now)


def test_folders_pending_groups_by_dirname(client):
    from parser.main import app_state
    _seed(app_state.conn)
    r = client.get("/v1/parser/folders/pending?limit=10")
    assert r.status_code == 200
    body = r.json()
    by_folder = {(f["root_id"], f["folder"]): f["count"] for f in body["folders"]}
    assert by_folder == {("r1", "/a/b"): 2, ("r1", "/a/c"): 1}
    assert body["total_groups"] == 2


def test_folders_pending_respects_limit(client):
    from parser.main import app_state
    _seed(app_state.conn)
    r = client.get("/v1/parser/folders/pending?limit=1")
    body = r.json()
    assert len(body["folders"]) == 1
    assert body["folders"][0]["count"] == 2  # 最高的那个
    assert body["total_groups"] == 2
