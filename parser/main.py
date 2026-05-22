from fastapi import FastAPI

from parser.routes import health


def create_app(*, skip_workers: bool = False) -> FastAPI:
    app = FastAPI(title="NimoOS-Parser", version="0.1.0")
    app.include_router(health.router)
    app.state.skip_workers = skip_workers
    return app


app = create_app()
