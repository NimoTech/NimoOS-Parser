from fastapi import APIRouter

router = APIRouter(prefix="/v1/parser", tags=["models"])


def get_conn():
    from parser.main import app_state
    return app_state.conn


@router.get("/models")
async def models() -> dict:
    conn = get_conn()
    rows = conn.execute(
        "SELECT name, version, modality, dim, active FROM model_versions "
        "ORDER BY name, registered_at DESC"
    ).fetchall()
    return {"models": [dict(r) for r in rows]}
