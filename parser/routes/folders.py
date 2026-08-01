import posixpath
from collections import defaultdict

from fastapi import APIRouter, Query

router = APIRouter(prefix="/v1/parser/folders", tags=["folders"])


def _conn():
    from parser.main import app_state
    return app_state.conn


# TODO(perf): MVP aggregates in Python memory; fetchall can use significant memory
# at 100k+ pending. Switch to SQLite GROUP BY (substr+instr to build dirname) if this
# actually becomes a bottleneck.
@router.get("/pending")
async def folders_pending(limit: int = Query(20, ge=1, le=200)) -> dict:
    rows = _conn().execute(
        "SELECT root_id, path FROM parse_jobs "
        "WHERE done_at IS NULL AND last_error IS NULL"
    ).fetchall()
    bucket: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        folder = posixpath.dirname(r[1]) or "/"
        bucket[(r[0], folder)] += 1
    items = sorted(
        ({"root_id": k[0], "folder": k[1], "count": v}
         for k, v in bucket.items()),
        key=lambda x: -x["count"],
    )
    return {"folders": items[:limit], "total_groups": len(items)}
