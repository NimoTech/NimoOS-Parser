from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from parser import repo_allowlist
from parser.tombstone_task import schedule_allowlist_sweep

router = APIRouter(prefix="/v1/parser/allowlist", tags=["allowlist"])


def _conn():
    from parser.main import app_state
    return app_state.conn


class ExtensionPatch(BaseModel):
    ext: str = Field(..., min_length=2, pattern=r"^\.[A-Za-z0-9]+$")
    enabled: bool


class FolderRuleCreate(BaseModel):
    root_id: str = Field(..., min_length=1)
    path_glob: str = Field(..., min_length=1)
    action: str = Field(..., pattern=r"^(allow|deny)$")


@router.get("/extensions")
async def get_extensions() -> dict:
    rows = repo_allowlist.list_extensions(_conn())
    return {"extensions": [dict(r) for r in rows]}


@router.patch("/extensions")
async def patch_extension(body: ExtensionPatch) -> dict:
    repo_allowlist.set_extension_enabled(_conn(), body.ext, body.enabled)
    schedule_allowlist_sweep()
    return {"ok": True}


@router.get("/folders")
async def get_folders() -> dict:
    return {"rules": repo_allowlist.list_folder_rules(_conn())}


@router.post("/folders", status_code=201)
async def create_folder_rule(body: FolderRuleCreate) -> dict:
    rid = repo_allowlist.add_folder_rule(
        _conn(),
        root_id=body.root_id,
        path_glob=body.path_glob,
        action=body.action,
    )
    schedule_allowlist_sweep()
    return {"id": rid}


@router.delete("/folders/{rule_id}", status_code=204)
async def delete_folder_rule(rule_id: str):
    ok = repo_allowlist.delete_folder_rule(_conn(), rule_id)
    if not ok:
        raise HTTPException(404, f"rule not found: {rule_id}")
    schedule_allowlist_sweep()
    return None
