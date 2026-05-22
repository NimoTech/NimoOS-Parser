import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    wiki_client: object = None
    gc_task: Optional[asyncio.Task] = None


app_state = AppState()


class _LazyBGEM3Adapter:
    """Embedder facade for TextPipeline that triggers BGE-M3 load on first
    embed call, so model loading doesn't block service startup."""

    def __init__(self) -> None:
        from parser.model_bge_m3 import BGEM3
        self.version = BGEM3.version
        self.dim = BGEM3.dim

    def embed_text(self, texts: list[str]) -> list[dict]:
        from parser.model_bge_m3 import BGEM3
        return BGEM3.load().embed_text(texts)


def _read_wiki_url(discovery_path: Path) -> Optional[str]:
    try:
        if not discovery_path.exists():
            return None
        content = discovery_path.read_text().strip()
        return content or None
    except OSError:
        return None


def _register_active_models(conn) -> None:
    from parser.repo_models import register_model, set_active
    now = int(time.time() * 1000)
    register_model(conn, name="bge-m3", version="v1", modality="text",
                   dim=1024, registered_at=now)
    set_active(conn, name="bge-m3", version="v1")
    register_model(conn, name="bge-reranker-v2-m3", version="v1",
                   modality="rerank", dim=None, registered_at=now)
    set_active(conn, name="bge-reranker-v2-m3", version="v1")


async def _full_lifecycle_startup(app: FastAPI) -> None:
    from parser.config import load_settings
    from parser.db import init_db
    from parser.pipeline_text import TextPipeline
    from parser.qdrant_store import QdrantStore
    from parser.wiki_client import WikiClient
    from parser.wiki_consumer import WikiConsumer
    from parser.workers import WorkerPool

    settings = load_settings()
    app_state.settings = settings

    settings.data_path.mkdir(parents=True, exist_ok=True)
    app_state.conn = init_db(settings.data_path / "parser.db")
    _register_active_models(app_state.conn)

    try:
        qstore = QdrantStore(url=settings.qdrant_url,
                             grpc_port=settings.qdrant_grpc_port)
        qstore.ensure_collections()
        app_state.qstore = qstore
    except Exception as e:
        log.warning("qdrant unavailable (%s); workers will fail until it returns", e)
        app_state.qstore = None

    wiki_url = _read_wiki_url(settings.wiki_discovery_path)
    if wiki_url:
        app_state.wiki_client = WikiClient(base_url=wiki_url)
    else:
        log.warning("wiki discovery file not found at %s; consumer disabled",
                    settings.wiki_discovery_path)
        app_state.wiki_client = None

    if app_state.qstore is not None:
        pipeline = TextPipeline(
            app_state.conn, qstore=app_state.qstore,
            embedder=_LazyBGEM3Adapter(),
            parser_version=settings.parser_version,
        )
        app_state.worker_pool = WorkerPool(
            app_state.conn, text_pipeline=pipeline,
            concurrency=settings.worker_text_concurrency,
            lease_s=settings.job_lease_s,
            wiki_client=app_state.wiki_client,
            parser_version=settings.parser_version,
        )
        await app_state.worker_pool.start()

    if app_state.wiki_client is not None:
        app_state.consumer = WikiConsumer(
            app_state.conn, app_state.wiki_client,
            poll_interval_s=settings.wiki_poll_interval_s,
            poll_limit=settings.wiki_poll_limit,
        )
        await app_state.consumer.start()

    app_state.gc_task = asyncio.create_task(_gc_loop(settings))


async def _gc_loop(settings) -> None:
    from parser.gc import sweep_tombstones
    grace_ms = settings.tombstone_grace_hours * 3600 * 1000
    figures_root = settings.data_path / "figures"
    while True:
        try:
            await asyncio.sleep(settings.gc_interval_s)
            if app_state.qstore is None or app_state.conn is None:
                continue
            await asyncio.to_thread(
                sweep_tombstones, app_state.conn,
                qstore=app_state.qstore, figures_root=figures_root,
                grace_ms=grace_ms, now_ms=int(time.time() * 1000),
            )
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.warning("gc sweep failed: %s", e)


async def _full_lifecycle_shutdown() -> None:
    if app_state.gc_task is not None:
        app_state.gc_task.cancel()
        try:
            await app_state.gc_task
        except (asyncio.CancelledError, Exception):
            pass
        app_state.gc_task = None
    if app_state.consumer is not None:
        try:
            await app_state.consumer.stop()
        except Exception as e:
            log.warning("consumer stop failed: %s", e)
        app_state.consumer = None
    if app_state.worker_pool is not None:
        try:
            await app_state.worker_pool.stop()
        except Exception as e:
            log.warning("worker_pool stop failed: %s", e)
        app_state.worker_pool = None
    if app_state.wiki_client is not None:
        try:
            await app_state.wiki_client.aclose()
        except Exception:
            pass
        app_state.wiki_client = None
    if app_state.conn is not None:
        try:
            app_state.conn.close()
        except Exception:
            pass
        app_state.conn = None


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

    if not app.state.skip_workers:
        try:
            await _full_lifecycle_startup(app)
        except Exception as e:
            log.exception("startup failed: %s", e)

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
        if not app.state.skip_workers:
            await _full_lifecycle_shutdown()
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
