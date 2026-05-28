"""Unit tests for legacy_office_extractor.

These tests do NOT need real LibreOffice — they monkeypatch subprocess.run
and verify our wrapper's behavior (lock, profile, outdir, error handling).
A separate integration test gated on shutil.which("soffice") exercises the
real binary in test_legacy_office_extractor_integration.
"""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from parser.legacy_office_extractor import (
    convert_legacy, is_legacy_binary_office, _TARGET, _LO_GATE,
)


def test_is_legacy_binary_office_matches_expected_exts():
    assert is_legacy_binary_office(".doc")
    assert is_legacy_binary_office(".DOC")
    assert is_legacy_binary_office(".ppt")
    assert is_legacy_binary_office(".xls")
    assert is_legacy_binary_office(".wps")
    assert not is_legacy_binary_office(".docx")
    assert not is_legacy_binary_office(".pdf")


def _make_completed_process(stdout=b"", stderr=b""):
    cp = MagicMock()
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = 0
    return cp


def test_convert_legacy_invokes_soffice_with_target_and_returns_produced_path(tmp_path):
    src = tmp_path / "report.doc"
    src.write_bytes(b"\xd0\xcf\x11\xe0fake-ole-header")  # OLE magic, not parsed

    def fake_run(cmd, **kwargs):
        # Find the outdir argument and drop a fake produced file there.
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "report.docx").write_bytes(b"fake-docx-bytes")
        return _make_completed_process()

    with patch("parser.legacy_office_extractor.subprocess.run", side_effect=fake_run):
        produced = convert_legacy(str(src))

    assert produced.name == "report.docx"
    assert produced.exists()
    assert produced.read_bytes() == b"fake-docx-bytes"


def test_convert_legacy_uses_private_profile_per_call(tmp_path):
    src = tmp_path / "x.doc"
    src.write_bytes(b"x")
    seen_profiles = []

    def fake_run(cmd, **kwargs):
        # Profile is passed via "-env:UserInstallation=file://<path>"
        for arg in cmd:
            if arg.startswith("-env:UserInstallation="):
                seen_profiles.append(arg)
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        (outdir / "x.docx").write_bytes(b"d")
        return _make_completed_process()

    with patch("parser.legacy_office_extractor.subprocess.run", side_effect=fake_run):
        convert_legacy(str(src))
        convert_legacy(str(src))

    assert len(seen_profiles) == 2
    assert seen_profiles[0] != seen_profiles[1], "each call must get its own profile dir"
    assert all(p.startswith("-env:UserInstallation=file://") for p in seen_profiles)


def test_convert_legacy_maps_ext_to_target(tmp_path):
    """ppt→pptx, xls→xlsx, wps→docx."""
    targets_seen = {}

    def fake_run_factory(expected_target, src_name):
        def fake_run(cmd, **kwargs):
            target = cmd[cmd.index("--convert-to") + 1]
            targets_seen[src_name] = target
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            stem = Path(src_name).stem
            (outdir / f"{stem}.{target}").write_bytes(b"x")
            return _make_completed_process()
        return fake_run

    for src_name, expected_target in [
        ("a.ppt", "pptx"), ("b.xls", "xlsx"),
        ("c.wps", "docx"), ("d.doc", "docx"),
    ]:
        src = tmp_path / src_name
        src.write_bytes(b"x")
        with patch("parser.legacy_office_extractor.subprocess.run",
                   side_effect=fake_run_factory(expected_target, src_name)):
            convert_legacy(str(src))

    assert targets_seen == {
        "a.ppt": "pptx", "b.xls": "xlsx",
        "c.wps": "docx", "d.doc": "docx",
    }


def test_convert_legacy_raises_on_nonzero_exit(tmp_path):
    src = tmp_path / "broken.doc"
    src.write_bytes(b"x")
    err = subprocess.CalledProcessError(
        returncode=1, cmd=["soffice"], output=b"", stderr=b"bad ole header",
    )
    with patch("parser.legacy_office_extractor.subprocess.run", side_effect=err):
        with pytest.raises(subprocess.CalledProcessError):
            convert_legacy(str(src))


def test_convert_legacy_raises_when_no_output_produced(tmp_path):
    """soffice sometimes returns 0 but writes nothing (e.g. unsupported
    sub-format). We must not silently return a missing path."""
    src = tmp_path / "empty.doc"
    src.write_bytes(b"x")

    def fake_run_no_output(cmd, **kwargs):
        return _make_completed_process(stderr=b"converter did nothing")

    with patch("parser.legacy_office_extractor.subprocess.run",
               side_effect=fake_run_no_output):
        with pytest.raises(RuntimeError, match="produced no"):
            convert_legacy(str(src))


def test_convert_legacy_raises_on_timeout(tmp_path):
    src = tmp_path / "slow.doc"
    src.write_bytes(b"x")
    err = subprocess.TimeoutExpired(cmd=["soffice"], timeout=120)
    with patch("parser.legacy_office_extractor.subprocess.run", side_effect=err):
        with pytest.raises(subprocess.TimeoutExpired):
            convert_legacy(str(src))


def test_convert_legacy_cleans_up_outdir_and_profile_on_failure(tmp_path):
    """Both temp dirs must be removed even when conversion fails."""
    src = tmp_path / "bad.doc"
    src.write_bytes(b"x")

    captured = {}

    def fake_run_capture_then_fail(cmd, **kwargs):
        captured["outdir"] = cmd[cmd.index("--outdir") + 1]
        for arg in cmd:
            if arg.startswith("-env:UserInstallation=file://"):
                captured["profile"] = arg.split("file://", 1)[1]
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    with patch("parser.legacy_office_extractor.subprocess.run",
               side_effect=fake_run_capture_then_fail):
        with pytest.raises(subprocess.CalledProcessError):
            convert_legacy(str(src))

    assert not Path(captured["outdir"]).exists(), "outdir leaked"
    assert not Path(captured["profile"]).exists(), "profile leaked"


def test_convert_legacy_rejects_unknown_ext(tmp_path):
    src = tmp_path / "x.pdf"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported legacy office ext"):
        convert_legacy(str(src))
