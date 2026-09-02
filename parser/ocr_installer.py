# parser/ocr_installer.py
"""Download/verify/activate storage for switchable OCR model sets.

One directory per model id under ocr_models_dir, fixed file names
(det.onnx / rec.onnx / cls.onnx) so the extractor never re-derives upstream
basenames. Downloads land in `<id>.tmp` and are renamed into place only
after every file passes its SHA256 — a crash mid-download leaves no
half-installed model. Progress is process-local (dict + lock); the UI polls
GET /v1/parser/ocr/models.
"""
import hashlib
import logging
import shutil
import threading
from pathlib import Path

import httpx

from parser.ocr_catalog import get_entry

log = logging.getLogger("parser.ocr_installer")

FILE_NAMES = {"det": "det.onnx", "rec": "rec.onnx", "cls": "cls.onnx"}

_models_dir = Path("/opt/nimoos-parser/models/ocr")
_lock = threading.Lock()
# id -> {"status": "downloading"|"error", "progress_pct": int, "error": str|None}
_progress: dict[str, dict] = {}


def set_models_dir(path: Path) -> None:
    global _models_dir
    _models_dir = Path(path)


def model_dir(model_id: str) -> Path:
    return _models_dir / model_id


def is_installed(model_id: str) -> bool:
    d = model_dir(model_id)
    return all((d / name).is_file() for name in FILE_NAMES.values())


def snapshot() -> dict[str, dict]:
    with _lock:
        return {k: dict(v) for k, v in _progress.items()}


def _fetch(url: str, dest: Path, cb) -> None:
    """Stream url to dest, calling cb(done_bytes, total_bytes) as it goes."""
    with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0),
                      follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 18):
                    f.write(chunk)
                    done += len(chunk)
                    cb(done, total)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_progress(model_id: str, **fields) -> None:
    with _lock:
        _progress.setdefault(model_id, {}).update(fields)


def _install_worker(entry: dict) -> None:
    model_id = entry["id"]
    tmp = _models_dir / f"{model_id}.tmp"
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        tasks = list(entry["files"].items())
        for i, (task, spec) in enumerate(tasks):
            dest = tmp / FILE_NAMES[task]

            def cb(done, total, _i=i, _n=len(tasks)):
                frac = (done / total) if total else 0.0
                _set_progress(model_id, status="downloading",
                              progress_pct=int((_i + frac) / _n * 100),
                              error=None)

            cb(0, 0)
            _fetch(spec["url"], dest, cb)
            got = _sha256(dest)
            if got != spec["sha256"]:
                raise ValueError(
                    f"sha256 mismatch for {task}: got {got[:12]}…, "
                    f"want {spec['sha256'][:12]}…")
        final = model_dir(model_id)
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        with _lock:
            _progress.pop(model_id, None)
        log.info("OCR model %s installed", model_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as-is
        log.warning("OCR model %s install failed: %s", model_id, exc)
        shutil.rmtree(tmp, ignore_errors=True)
        _set_progress(model_id, status="error", progress_pct=0,
                      error=str(exc))


def start_install(model_id: str) -> str:
    entry = get_entry(model_id)
    if entry is None:
        return "unknown"
    if is_installed(model_id):
        return "installed"
    with _lock:
        row = _progress.get(model_id)
        if row and row.get("status") == "downloading":
            return "already_running"
        _progress[model_id] = {"status": "downloading", "progress_pct": 0,
                               "error": None}
    threading.Thread(target=_install_worker, args=(entry,),
                     name=f"ocr-install-{model_id}", daemon=True).start()
    return "started"


def remove(model_id: str) -> None:
    shutil.rmtree(model_dir(model_id), ignore_errors=True)
    shutil.rmtree(_models_dir / f"{model_id}.tmp", ignore_errors=True)
    with _lock:
        _progress.pop(model_id, None)
