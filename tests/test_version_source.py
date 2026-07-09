import builtins, importlib, sys


def _reload_config():
    sys.modules.pop("parser.config", None)
    import parser.config as cfg
    return importlib.reload(cfg)


def test_falls_back_to_dev_when_version_module_absent(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "parser._version":
            raise ImportError("simulated absent")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("parser._version", None)
    cfg = _reload_config()
    assert cfg.PARSER_APP_VERSION == "dev"


def test_reads_generated_version(monkeypatch):
    import types
    mod = types.ModuleType("parser._version")
    mod.VERSION = "9.9.9+gtest"
    monkeypatch.setitem(sys.modules, "parser._version", mod)
    cfg = _reload_config()
    assert cfg.PARSER_APP_VERSION == "9.9.9+gtest"
