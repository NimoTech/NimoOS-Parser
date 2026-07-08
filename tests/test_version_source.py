def test_parser_app_version_falls_back_to_dev(monkeypatch):
    import importlib, sys
    sys.modules.pop("parser._version", None)  # ensure absent
    import parser.config as cfg
    importlib.reload(cfg)
    assert isinstance(cfg.PARSER_APP_VERSION, str) and cfg.PARSER_APP_VERSION
