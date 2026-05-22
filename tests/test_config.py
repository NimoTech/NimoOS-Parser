from pathlib import Path

from parser.config import Settings, load_settings


def test_load_settings_uses_defaults_when_no_file(tmp_path):
    settings = load_settings(conf_path=tmp_path / "nope.conf")
    assert isinstance(settings, Settings)
    assert settings.data_path == Path("/var/lib/nimoos/parser")
    assert settings.wiki_poll_interval_s == 2
    assert settings.tombstone_grace_hours == 24
    assert settings.parser_version == "parser/0.1.0"


def test_load_settings_overrides_from_ini(tmp_path):
    conf = tmp_path / "parser.conf"
    conf.write_text(
        "[parser]\n"
        "DataPath = /tmp/nimoos-parser\n"
        "WikiPollIntervalSec = 5\n"
        "TombstoneGraceHours = 6\n"
    )
    settings = load_settings(conf_path=conf)
    assert settings.data_path == Path("/tmp/nimoos-parser")
    assert settings.wiki_poll_interval_s == 5
    assert settings.tombstone_grace_hours == 6


def test_env_overrides_ini(monkeypatch, tmp_path):
    monkeypatch.setenv("PARSER_DATA_PATH", "/tmp/from-env")
    conf = tmp_path / "parser.conf"
    conf.write_text("[parser]\nDataPath = /tmp/from-ini\n")
    settings = load_settings(conf_path=conf)
    assert settings.data_path == Path("/tmp/from-env")
