from unittest.mock import MagicMock

from parser.repo_models import set_wiki_cursor
from parser.repo_state import set_cursor_gap, set_verify_last


def test_stats_exposes_wiki_cursor_and_verify_last(client, monkeypatch):
    from parser.main import app_state
    qstore = MagicMock(); qstore.count_vectors.return_value = {"text": 0, "visual": 0}
    monkeypatch.setattr(app_state, "qstore", qstore)
    set_wiki_cursor(app_state.conn, since_ms=123, last_seq=45, now_ms=1)
    set_cursor_gap(app_state.conn, {"detected_at": 9, "since_ms": 123, "last_seq": 45, "archive_cutoff_ms": 999})
    set_verify_last(app_state.conn, {"trigger": "manual", "ok": True})

    body = client.get("/v1/parser/stats").json()
    assert body["wiki_cursor"] == {"since_ms": 123, "last_seq": 45, "gap": True, "gap_detected_at": 9}
    assert body["verify_last"] == {"trigger": "manual", "ok": True}
