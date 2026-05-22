from parser.db import init_db
from parser.repo_jobs import list_jobs
from parser.repo_records import upsert_file_record, upsert_file_path


def test_rescan_reindex_enqueues_jobs_for_root(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    upsert_file_record(conn, file_id="abc", sha256_full="a" * 64, size=1,
                       mime="text/plain", modalities_done={},
                       parser_version="parser/0.1.0", indexed_at=1)
    upsert_file_path(conn, "root1", "/a.txt", "abc", 1)
    upsert_file_path(conn, "root1", "/b.txt", "abc", 1)
    monkeypatch.setattr("parser.routes.rescan.get_conn", lambda: conn)

    r = client.post("/v1/parser/rescan",
                     json={"root_id": "root1", "op": "reindex"})
    assert r.status_code == 200
    pending = list_jobs(conn, status="pending", limit=10)
    assert {j["path"] for j in pending} == {"/a.txt", "/b.txt"}


def test_rescan_invalid_op(client):
    r = client.post("/v1/parser/rescan",
                     json={"root_id": "root1", "op": "nope"})
    assert r.status_code == 400
