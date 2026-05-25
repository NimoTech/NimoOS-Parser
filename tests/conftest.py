import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    from parser.db import init_db
    from parser.main import app_state, create_app

    conn = init_db(tmp_path / "test.db")
    prev_conn = app_state.conn
    app_state.conn = conn
    try:
        app = create_app(skip_workers=True)
        with TestClient(app) as c:
            yield c
    finally:
        app_state.conn = prev_conn
        conn.close()
