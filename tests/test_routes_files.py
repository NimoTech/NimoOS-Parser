from parser.db import init_db
from parser.repo_records import upsert_file_record, upsert_file_path


def test_files_endpoint_returns_expanded_records(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    upsert_file_record(conn, file_id="abc", sha256_full="a"*64, size=10,
                       mime="text/markdown",
                       modalities_done={"text": "bge-m3/v1"},
                       parser_version="parser/0.1.0", indexed_at=100)
    upsert_file_path(conn, "r1", "/a.md", "abc", 50)
    upsert_file_path(conn, "r2", "/x.md", "abc", 50)
    monkeypatch.setattr("parser.routes.files.get_conn", lambda: conn)

    r = client.get("/v1/parser/_internal/files?file_ids=abc,nope")
    assert r.status_code == 200
    body = r.json()
    assert len(body["files"]) == 1
    f = body["files"][0]
    assert f["file_id"] == "abc"
    assert f["mime"] == "text/markdown"
    assert f["modalities_done"] == {"text": "bge-m3/v1"}
    assert {p["root_id"] for p in f["paths"]} == {"r1", "r2"}
    assert body["missing"] == ["nope"]


def test_files_endpoint_caps_batch(client):
    too_many = ",".join(str(i) for i in range(250))
    r = client.get(f"/v1/parser/_internal/files?file_ids={too_many}")
    assert r.status_code == 400
