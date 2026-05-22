import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from parser.routes import embed, health, jobs, models, rerank, rescan, stats

log = logging.getLogger("parser.main")


@dataclass
class AppState:
    conn: object = None
    qstore: object = None
    settings: object = None
    consumer: object = None
    worker_pool: object = None


app_state = AppState()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    discovery_path = os.environ.get("PARSER_DISCOVERY_FILE")
    bind_addr = os.environ.get("PARSER_BIND_ADDR")
    if discovery_path and bind_addr:
        try:
            from parser.service_discovery import write_url
            write_url(Path(discovery_path), bind_addr)
        except Exception as e:
            log.warning("write_url failed: %s", e)
    notifier = None
    try:
        from sdnotify import SystemdNotifier
        notifier = SystemdNotifier()
        notifier.notify("READY=1")
    except Exception:
        notifier = None
    try:
        yield
    finally:
        if discovery_path:
            try:
                from parser.service_discovery import remove_url
                remove_url(Path(discovery_path))
            except Exception:
                pass
        if notifier is not None:
            try:
                notifier.notify("STOPPING=1")
            except Exception:
                pass


def create_app(*, skip_workers: bool = False) -> FastAPI:
    app = FastAPI(title="NimoOS-Parser", version="0.1.0",
                  lifespan=_lifespan)
    app.include_router(embed.router)
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(rerank.router)
    app.include_router(rescan.router)
    app.include_router(stats.router)
    app.include_router(models.router)
    app.state.skip_workers = skip_workers
    return app


app = create_app()
