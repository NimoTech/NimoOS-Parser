"""Async task that, when allowlist rules change, scans file_records and
tombstones any file_id whose current paths no longer pass is_path_indexable.

Stub for Task 3; real implementation in Task 4. The router code can call
schedule_allowlist_sweep() freely without breaking — it's a no-op until
Task 4 lands.
"""
import logging

log = logging.getLogger("parser.tombstone_task")


def schedule_allowlist_sweep() -> None:
    """No-op stub. Task 4 wires this to a real asyncio task."""
    log.debug("schedule_allowlist_sweep: stub no-op")
