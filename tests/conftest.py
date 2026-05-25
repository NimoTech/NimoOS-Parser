import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    # DB isolation dance: we inject a tmp-path DB into app_state.conn BEFORE
    # the lifespan runs. create_app(skip_workers=True) sets app.state.skip_workers
    # to True, so _lifespan enters the skip_workers branch. That branch guards
    # its init_db() call with `if app_state.conn is None`, so because we have
    # already set app_state.conn here, the production path
    # (/var/lib/nimoos/parser/parser.db) is never opened or touched. The conn
    # we set here is therefore the one every route handler sees during the test.
    # On teardown the lifespan also skips closing the conn (opened_conn_here is
    # False), leaving cleanup to the finally block below.
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
