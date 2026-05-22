import os
import uuid

import pytest

from parser.qdrant_store import QdrantStore

pytestmark = pytest.mark.skipif(
    os.environ.get("QDRANT_URL") is None,
    reason="set QDRANT_URL=http://127.0.0.1:6333 to run",
)


@pytest.fixture
def store():
    s = QdrantStore(url=os.environ["QDRANT_URL"], grpc_port=6334)
    suffix = uuid.uuid4().hex[:8]
    s.text_collection = f"text_chunks_test_{suffix}"
    s.visual_collection = f"visual_chunks_test_{suffix}"
    s.ensure_collections()
    yield s
    s.client.delete_collection(s.text_collection)
    s.client.delete_collection(s.visual_collection)


def test_ensure_collections_idempotent(store):
    store.ensure_collections()
    info = store.client.get_collection(store.text_collection)
    assert info.config.params.vectors["dense"].size == 1024


def test_upsert_text_chunk_and_retrieve(store):
    point_id = str(uuid.uuid4())
    store.upsert_text_chunks([{
        "id": point_id,
        "dense": [0.1] * 1024,
        "sparse": {"indices": [1, 2], "values": [0.5, 0.3]},
        "payload": {
            "file_id": "abc", "root_ids": ["root1"],
            "kind": "body", "mime": "text/markdown",
            "chunk_no": 0, "text": "hello world",
            "parser_version": "parser/0.1.0",
            "embed_model_version": "bge-m3/v1",
            "indexed_at": 1,
        },
    }])
    pts = store.client.retrieve(store.text_collection, [point_id], with_payload=True)
    assert pts[0].payload["text"] == "hello world"


def test_payload_set_root_ids_updates_both_collections(store):
    text_point = str(uuid.uuid4())
    visual_point = str(uuid.uuid4())
    store.upsert_text_chunks([{
        "id": text_point, "dense": [0.0] * 1024,
        "sparse": {"indices": [], "values": []},
        "payload": {"file_id": "abc", "root_ids": ["root1"],
                    "kind": "body", "mime": "text/plain", "chunk_no": 0,
                    "text": "x", "parser_version": "p", "embed_model_version": "e",
                    "indexed_at": 1},
    }])
    store.upsert_visual_chunks([{
        "id": visual_point, "dense": [0.0] * 1152,
        "payload": {"file_id": "abc", "root_ids": ["root1"],
                    "kind": "image", "mime": "image/jpeg", "chunk_no": 0,
                    "parser_version": "p", "embed_model_version": "e",
                    "indexed_at": 1},
    }])
    # Pretend they were tombstoned, then revive:
    store.tombstone_file(file_id="abc", tombstoned_at=999)
    store.set_root_ids_for_file(file_id="abc", root_ids=["root1", "root2"])
    # I3: both collections must reflect the new root_ids AND clear tombstone
    text_pts = store.client.retrieve(store.text_collection, [text_point],
                                     with_payload=True)
    visual_pts = store.client.retrieve(store.visual_collection, [visual_point],
                                       with_payload=True)
    assert set(text_pts[0].payload["root_ids"]) == {"root1", "root2"}
    assert text_pts[0].payload["tombstoned_at"] is None
    assert set(visual_pts[0].payload["root_ids"]) == {"root1", "root2"}
    assert visual_pts[0].payload["tombstoned_at"] is None


def test_tombstone_and_delete(store):
    point_id = str(uuid.uuid4())
    store.upsert_text_chunks([{
        "id": point_id, "dense": [0.0] * 1024,
        "sparse": {"indices": [], "values": []},
        "payload": {"file_id": "abc", "root_ids": ["root1"],
                    "kind": "body", "mime": "text/plain", "chunk_no": 0,
                    "text": "x", "parser_version": "p", "embed_model_version": "e",
                    "indexed_at": 1},
    }])
    store.tombstone_file(file_id="abc", tombstoned_at=999)
    pts = store.client.retrieve(store.text_collection, [point_id], with_payload=True)
    assert pts[0].payload["root_ids"] == []
    assert pts[0].payload["tombstoned_at"] == 999
    store.delete_file(file_id="abc")
    pts = store.client.retrieve(store.text_collection, [point_id])
    assert pts == []
