from fastapi import APIRouter

from parser.config import PARSER_APP_VERSION

router = APIRouter(prefix="/v1/parser", tags=["version"])


@router.get("/version")
async def version() -> dict:
    return {"name": "Parser", "version": PARSER_APP_VERSION}
