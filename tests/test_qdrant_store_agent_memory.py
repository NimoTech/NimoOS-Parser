from types import SimpleNamespace
import parser.qdrant_store as qs


class FakeClient:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.created = []
        self.indexed = []
        self.upserts = []
        self.last_query = None
        self.query_result = SimpleNamespace(points=[])

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self.existing])

    def create_collection(self, collection_name, **kw):
        self.created.append(collection_name)
        self.existing.add(collection_name)

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexed.append((collection_name, field_name))

    def upsert(self, collection, points, wait):
        self.upserts.append((collection, points))

    def query_points(self, collection_name, query, using, query_filter,
                     limit, with_payload):
        self.last_query = SimpleNamespace(
            collection=collection_name, vector=query, using=using,
            filter=query_filter, limit=limit)
        return self.query_result


def _store(existing=()):
    s = qs.QdrantStore.__new__(qs.QdrantStore)  # bypass __init__ (no live client)
    s.text_collection = qs.TEXT_COLLECTION
    s.visual_collection = qs.VISUAL_COLLECTION
    s.agent_memory_collection = qs.AGENT_MEMORY_COLLECTION
    s.client = FakeClient(existing)
    return s


def test_ensure_creates_agent_memory_when_absent():
    s = _store(existing=())
    s.ensure_collections()
    assert qs.AGENT_MEMORY_COLLECTION in s.client.created
    assert (qs.AGENT_MEMORY_COLLECTION, "user_id") in s.client.indexed


def test_ensure_skips_agent_memory_when_present():
    s = _store(existing=(qs.TEXT_COLLECTION, qs.VISUAL_COLLECTION,
                         qs.AGENT_MEMORY_COLLECTION))
    s.ensure_collections()
    assert qs.AGENT_MEMORY_COLLECTION not in s.client.created


def test_upsert_agent_memory_builds_points():
    s = _store()
    s.upsert_agent_memory([
        {"id": "p1", "dense": [0.1] * 1024,
         "payload": {"user_id": "u1", "session_id": "s1", "chunk_no": 0,
                     "text": "hi", "created_at": 1}}])
    coll, points = s.client.upserts[0]
    assert coll == qs.AGENT_MEMORY_COLLECTION
    assert points[0].id == "p1"
    assert points[0].payload["user_id"] == "u1"


def test_query_agent_memory_filters_user_id_and_maps():
    s = _store()
    s.client.query_result = SimpleNamespace(points=[
        SimpleNamespace(score=0.9, payload={"text": "t", "session_id": "s1",
                                            "chunk_no": 2, "created_at": 5})])
    hits = s.query_agent_memory("u1", [0.1] * 1024, limit=3)
    # the filter passed to Qdrant must constrain user_id
    f = s.client.last_query.filter
    assert any(getattr(c, "key", None) == "user_id" for c in f.must)
    assert s.client.last_query.limit == 3
    assert hits == [{"text": "t", "session_id": "s1", "chunk_no": 2,
                     "created_at": 5, "score": 0.9}]
