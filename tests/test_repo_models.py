import pytest

from parser.db import init_db
from parser.repo_models import (
    register_model, get_active_models, set_active,
    get_wiki_cursor, set_wiki_cursor,
)


@pytest.fixture
def conn(tmp_path):
    return init_db(tmp_path / "p.db")


def test_register_and_list(conn):
    register_model(conn, name="bge-m3", version="v1", modality="text",
                   dim=1024, registered_at=100)
    register_model(conn, name="bge-m3", version="v2", modality="text",
                   dim=1024, registered_at=200)
    set_active(conn, name="bge-m3", version="v2")
    actives = get_active_models(conn)
    assert actives == {"bge-m3": {"version": "v2", "modality": "text", "dim": 1024}}


def test_wiki_cursor(conn):
    assert get_wiki_cursor(conn) == 0
    set_wiki_cursor(conn, since_ms=1234, now_ms=5000)
    assert get_wiki_cursor(conn) == 1234
