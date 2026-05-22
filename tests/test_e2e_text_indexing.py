import asyncio
import os
import sqlite3
import time
import uuid
from pathlib import Path

import pytest

from parser.db import init_db
from parser.pipeline_text import TextPipeline
from parser.qdrant_store import QdrantStore
from parser.repo_jobs import enqueue_job
from parser.workers import WorkerPool


pytestmark = pytest.mark.skipif(
    os.environ.get("QDRANT_URL") is None,
    reason="set QDRANT_URL=http://127.0.0.1:6333 to run",
)


class FakeBGE:
    version = "bge-m3/v1"
    dim = 1024
    def embed_text(self, texts):
        return [{"dense": [float((i + 1) * 0.001)] * 1024,
                 "sparse": {"indices": [hash(t) % 1000],
                            "values": [0.5]}}
                for i, t in enumerate(texts)]


@pytest.mark.asyncio
async def test_e2e_text_to_qdrant(tmp_path):
    suffix = uuid.uuid4().hex[:8]
    qstore = QdrantStore(url=os.environ["QDRANT_URL"], grpc_port=6334)
    qstore.text_collection = f"text_chunks_e2e_{suffix}"
    qstore.visual_collection = f"visual_chunks_e2e_{suffix}"
    qstore.ensure_collections()

    try:
        conn = init_db(tmp_path / "p.db")
        pipe = TextPipeline(conn, qstore=qstore, embedder=FakeBGE(),
                            parser_version="parser/0.1.0")

        md = tmp_path / "hello.md"
        md.write_text(
            "# Hello\n\n" + ("This is a test paragraph. " * 30) +
            "\n\n## Sub\n\n" + ("Another section here. " * 30),
            encoding="utf-8",
        )

        enqueue_job(conn, root_id="root_e2e", path=str(md), op="index",
                    priority=100, now_ms=int(time.time() * 1000))

        pool = WorkerPool(conn, text_pipeline=pipe, concurrency=1, lease_s=10)
        await pool.start()
        for _ in range(100):
            jobs_remaining = conn.execute(
                "SELECT COUNT(*) FROM parse_jobs WHERE done_at IS NULL"
            ).fetchone()[0]
            if jobs_remaining == 0:
                break
            await asyncio.sleep(0.05)
        await pool.stop()

        counts = qstore.count_vectors()
        assert counts["text"] >= 1
    finally:
        qstore.client.delete_collection(qstore.text_collection)
        qstore.client.delete_collection(qstore.visual_collection)
