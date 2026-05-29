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


def test_list_files_public_returns_paginated(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    for i in range(3):
        upsert_file_record(conn, file_id=f"f{i}", sha256_full="s"+str(i),
                           size=10, mime="text/plain",
                           modalities_done={"text": "bge-m3/v1"},
                           parser_version="parser/0.2.0", indexed_at=100 + i)
        upsert_file_path(conn, f"r1", f"/p/{i}", f"f{i}", 0)
    monkeypatch.setattr("parser.routes.files.get_conn", lambda: conn)
    r = client.get("/v1/parser/files?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert len(body["files"]) == 3
    assert [f["file_id"] for f in body["files"]] == ["f2", "f1", "f0"]


def test_list_files_mime_prefix_filter(client, monkeypatch, tmp_path):
    conn = init_db(tmp_path / "p.db")
    upsert_file_record(conn, file_id="legacy", sha256_full="s", size=10,
                       mime="application/legacy-office/doc",
                       modalities_done={}, parser_version="parser/0.2.0",
                       indexed_at=100)
    upsert_file_path(conn, "r1", "/p/legacy", "legacy", 0)
    upsert_file_record(conn, file_id="md", sha256_full="s2", size=10,
                       mime="text/markdown", modalities_done={},
                       parser_version="parser/0.2.0", indexed_at=100)
    upsert_file_path(conn, "r1", "/p/md", "md", 0)
    monkeypatch.setattr("parser.routes.files.get_conn", lambda: conn)
    r = client.get("/v1/parser/files?mime_prefix=application/legacy-office/")
    assert r.status_code == 200
    assert {f["file_id"] for f in r.json()["files"]} == {"legacy"}


def test_list_files_invalid_sort_rejected(client):
    r = client.get("/v1/parser/files?sort=banana")
    # FastAPI regex/pattern validation → 422; if route reaches service it's 400.
    assert r.status_code in (400, 422)
