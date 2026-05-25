"""Resolve the actual torch device (cuda|cpu) from persisted user preference.

The user picks `auto`, `cuda`, or `cpu` in parser_state. This module turns
that choice into a concrete device string that model.load() can consume.
Centralised so embed, rerank, and the indexing worker all see the same
value at any given moment.
"""

from parser.hardware import resolve_device
from parser.repo_state import get_state


def current_device(conn) -> str:
    return resolve_device(get_state(conn)["device"])
