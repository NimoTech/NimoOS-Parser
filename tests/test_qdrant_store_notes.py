"""knowledge_notes collection 的 upsert/query/delete 单测(fake client)。"""
import types

from parser.qdrant_store import QdrantStore, KNOWLEDGE_NOTES_COLLECTION
from qdrant_client import models as qm


class _FakeClient:
    def __init__(self):
        self.upserts = []
        self.deletes = []
        self.queries = []

    def upsert(self, collection_name, points, wait=True):
        self.upserts.append((collection_name, points))

    def delete(self, collection_name, points_selector, wait=True):
        self.deletes.append((collection_name, points_selector))

    def query_points(self, collection_name, query, using, query_filter,
                     limit, with_payload):
        self.queries.append((collection_name, query_filter, limit))
        pt = types.SimpleNamespace(score=0.9, payload={
            "note_id": "n1", "chunk_no": 0, "text": "hello",
            "type": "note", "status": "curated", "updated_at": 123,
            "user_id": "1",
        })
        return types.SimpleNamespace(points=[pt])


def _store():
    s = QdrantStore.__new__(QdrantStore)   # 跳过真实连接
    s.client = _FakeClient()
    s.text_collection = "text_chunks"
    s.visual_collection = "visual_chunks"
    s.agent_memory_collection = "agent_memory"
    s.notes_collection = KNOWLEDGE_NOTES_COLLECTION
    return s


def test_upsert_notes_builds_dense_and_sparse_points():
    s = _store()
    s.upsert_notes([{
        "id": "pid1", "dense": [0.1] * 4,
        "sparse": {"indices": [3, 7], "values": [0.5, 0.2]},
        "payload": {"user_id": "1", "note_id": "n1", "chunk_no": 0,
                    "text": "t", "type": "note", "status": "draft",
                    "created_by": "agent", "updated_at": 1},
    }])
    coll, points = s.client.upserts[0]
    assert coll == KNOWLEDGE_NOTES_COLLECTION
    assert isinstance(points[0], qm.PointStruct)
    assert points[0].vector["bm25"].indices == [3, 7]
    assert points[0].payload["user_id"] == "1"


def test_query_notes_filters_by_user_and_status():
    s = _store()
    hits = s.query_notes("1", [0.1] * 4, limit=5, statuses=["curated", "draft"])
    coll, qfilter, limit = s.client.queries[0]
    assert coll == KNOWLEDGE_NOTES_COLLECTION and limit == 5
    keys = [c.key for c in qfilter.must]
    assert "user_id" in keys and "status" in keys
    assert hits[0]["note_id"] == "n1" and hits[0]["score"] == 0.9


def test_delete_note_filters_on_both_user_and_note():
    s = _store()
    s.delete_note("1", "n1")
    coll, selector = s.client.deletes[0]
    assert coll == KNOWLEDGE_NOTES_COLLECTION
    keys = [c.key for c in selector.filter.must]
    assert set(keys) == {"user_id", "note_id"}
