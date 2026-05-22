from dataclasses import dataclass
from fastapi import FastAPI

from parser.routes import health, jobs


@dataclass
class AppState:
    conn: object = None
    qstore: object = None
    settings: object = None
    consumer: object = None
    worker_pool: object = None


app_state = AppState()


def create_app(*, skip_workers: bool = False) -> FastAPI:
    app = FastAPI(title="NimoOS-Parser", version="0.1.0")
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.state.skip_workers = skip_workers
    return app


app = create_app()
