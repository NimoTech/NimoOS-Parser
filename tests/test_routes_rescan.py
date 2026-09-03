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
    # I4: enqueued op must actually be 'reindex' (not silently downgraded
    # to 'index') so reindex jobs follow the modality-aware reindex path.
    assert {j["op"] for j in pending} == {"reindex"}


def test_rescan_invalid_op(client):
    r = client.post("/v1/parser/rescan",
                     json={"root_id": "root1", "op": "nope"})
    assert r.status_code == 400


def test_rescan_verify_returns_501(client, monkeypatch, tmp_path):
    # I4: verify is a real op name in the spec but the implementation isn't
    # ready. Refuse explicitly rather than silently downgrading to index.
    conn = init_db(tmp_path / "p.db")
    monkeypatch.setattr("parser.routes.rescan.get_conn", lambda: conn)
    r = client.post("/v1/parser/rescan",
                     json={"root_id": "root1", "op": "verify"})
    assert r.status_code == 501
    # And no jobs were enqueued
    assert list_jobs(conn, status="pending", limit=10) == []


def test_rescan_reindex_skips_paths_no_longer_indexable(client, monkeypatch, tmp_path):
    # A root rescan must not resurrect records the ingest gate would refuse
    # (container dirs like .system_data, disabled extensions).
    from parser import repo_allowlist
    conn = init_db(tmp_path / "p.db")
    for fid, path in (("ok", "/DATA/a.txt"),
                      ("sys", "/DATA/.system_data/home/nimo/.claude.json"),
                      ("off", "/DATA/c.pdf")):
        upsert_file_record(conn, file_id=fid, sha256_full=fid * 16, size=1,
                           mime="text/plain", modalities_done={},
                           parser_version="parser/0.1.0", indexed_at=1)
        upsert_file_path(conn, "root1", path, fid, 1)
    repo_allowlist.set_extension_enabled(conn, ".pdf", False)
    monkeypatch.setattr("parser.routes.rescan.get_conn", lambda: conn)

    r = client.post("/v1/parser/rescan",
                     json={"root_id": "root1", "op": "reindex"})
    assert r.status_code == 200
    assert r.json() == {"queued": 1}
    assert [j["path"] for j in list_jobs(conn, status="pending", limit=10)] == ["/DATA/a.txt"]
