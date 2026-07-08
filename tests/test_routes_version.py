def test_version_returns_name_and_version(client):
    r = client.get("/v1/parser/version")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Parser"
    assert isinstance(body["version"], str) and body["version"]
