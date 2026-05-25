from parser.wiki_consumer import _op_for_event


def test_allowed_ext_returns_index():
    ev = {"op": "create", "path": "/x.md", "is_dir": False}
    assert _op_for_event(ev) == "index"


def test_disallowed_ext_returns_none():
    for path in ("/x.MOV", "/x.mp4", "/x.jpg", "/x.zip", "/x.immich", "/x.sql.gz"):
        ev = {"op": "create", "path": path, "is_dir": False}
        assert _op_for_event(ev) is None, f"should skip {path}"


def test_delete_event_passes_through_for_any_ext():
    # delete 任何文件都要 forward 到 parser (用于清向量)
    ev = {"op": "delete", "path": "/x.MOV", "is_dir": False}
    assert _op_for_event(ev) == "delete"


def test_unknown_ext_skipped():
    ev = {"op": "create", "path": "/x.weirdext", "is_dir": False}
    assert _op_for_event(ev) is None


def test_no_extension_skipped():
    ev = {"op": "create", "path": "/Makefile", "is_dir": False}
    assert _op_for_event(ev) is None
