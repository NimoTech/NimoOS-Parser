import pytest


def test_get_extensions_lists_seeded(client):
    r = client.get("/v1/parser/allowlist/extensions")
    assert r.status_code == 200
    data = r.json()
    exts = {e["ext"] for e in data["extensions"]}
    assert ".pdf" in exts
    assert ".md" in exts


def test_patch_extension_toggles(client):
    r = client.patch("/v1/parser/allowlist/extensions",
                     json={"ext": ".pdf", "enabled": False})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r2 = client.get("/v1/parser/allowlist/extensions")
    pdfs = [e for e in r2.json()["extensions"] if e["ext"] == ".pdf"]
    assert pdfs[0]["enabled"] == 0


def test_patch_extension_validates_input(client):
    r = client.patch("/v1/parser/allowlist/extensions",
                     json={"ext": "", "enabled": True})
    assert r.status_code == 422


def test_folder_rules_crud_via_api(client):
    # Empty initially
    r = client.get("/v1/parser/allowlist/folders")
    assert r.status_code == 200
    assert r.json()["rules"] == []

    # Create
    r2 = client.post("/v1/parser/allowlist/folders", json={
        "root_id": "r1", "path_glob": "/Downloads/*", "action": "deny",
    })
    assert r2.status_code == 201
    rid = r2.json()["id"]
    assert rid

    # List shows it
    r3 = client.get("/v1/parser/allowlist/folders")
    assert len(r3.json()["rules"]) == 1

    # Delete
    r4 = client.delete(f"/v1/parser/allowlist/folders/{rid}")
    assert r4.status_code == 204

    # Empty again
    r5 = client.get("/v1/parser/allowlist/folders")
    assert r5.json()["rules"] == []


def test_folder_rules_validates_action(client):
    r = client.post("/v1/parser/allowlist/folders", json={
        "root_id": "r1", "path_glob": "/x/*", "action": "wat",
    })
    assert r.status_code == 422


def test_delete_unknown_folder_rule_returns_404(client):
    r = client.delete("/v1/parser/allowlist/folders/does-not-exist")
    assert r.status_code == 404
