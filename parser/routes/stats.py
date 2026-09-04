from fastapi import APIRouter

from parser.repo_jobs import list_jobs
from parser.repo_models import get_active_models, get_wiki_cursor
from parser.repo_state import get_cursor_gap, get_verify_last

router = APIRouter(prefix="/v1/parser", tags=["stats"])


def get_conn():
    from parser.main import app_state
    return app_state.conn


def get_qstore():
    from parser.main import app_state
    return app_state.qstore


def get_wiki_cursor_val(conn):
    since_ms, _last_seq = get_wiki_cursor(conn)
    return since_ms


def get_pool():
    from parser.main import app_state
    return getattr(app_state, "worker_pool", None)


def _wiki_cursor(conn) -> dict:
    since_ms, last_seq = get_wiki_cursor(conn)
    gap = get_cursor_gap(conn)
    return {"since_ms": since_ms, "last_seq": last_seq, "gap": gap is not None,
            "gap_detected_at": gap["detected_at"] if gap else None}


@router.get("/stats")
async def stats() -> dict:
    conn = get_conn()
    qstore = get_qstore()
    counts = qstore.count_vectors()
    pending = len(list_jobs(conn, status="pending", limit=10_000))
    running = len(list_jobs(conn, status="running", limit=10_000))
    failed = len(list_jobs(conn, status="failed", limit=10_000))
    actives = get_active_models(conn)
    models = [{"name": k, **v} for k, v in actives.items()]
    indexed_files = conn.execute(
        "SELECT COUNT(*) FROM file_records WHERE tombstoned_at IS NULL"
    ).fetchone()[0]
    done = conn.execute(
        "SELECT COUNT(*) FROM parse_jobs WHERE done_at IS NOT NULL"
    ).fetchone()[0]

    pool = get_pool()
    tp = pool.throughput() if pool is not None else {"done_last_10m": 0, "rate_per_min": 0.0}
    eta_s = None
    if tp["rate_per_min"] > 0 and pending > 0:
        eta_s = int(pending * 60 / tp["rate_per_min"])

    return {
        "queue_depth": {"pending": pending, "running": running,
                        "failed": failed, "done": done},
        "indexed_files": indexed_files,
        "total_vectors_text": counts["text"],
        "total_vectors_visual": counts["visual"],
        "last_cursor_ms": get_wiki_cursor_val(conn),
        "models": models,
        "done_last_10m": tp["done_last_10m"],
        "rate_per_min": tp["rate_per_min"],
        "eta_s": eta_s,
        "wiki_cursor": _wiki_cursor(conn),
        "verify_last": get_verify_last(conn),
    }
