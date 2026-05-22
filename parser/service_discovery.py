import os
from pathlib import Path


def write_url(path: Path, addr: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = addr if addr.startswith("http") else f"http://{addr}"
    path.write_text(url + "\n")


def remove_url(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
