from typing import Iterable

from qdrant_client import QdrantClient, models as qm


TEXT_COLLECTION = "text_chunks"
VISUAL_COLLECTION = "visual_chunks"

TEXT_DENSE_DIM = 1024
VISUAL_DENSE_DIM = 1152

AGENT_MEMORY_COLLECTION = "agent_memory"
AGENT_MEMORY_DENSE_DIM = 1024

KNOWLEDGE_NOTES_COLLECTION = "knowledge_notes"
KNOWLEDGE_NOTES_DENSE_DIM = 1024


class QdrantStore:
    def __init__(self, url: str, grpc_port: int = 6334) -> None:
        self.client = QdrantClient(url=url, prefer_grpc=True, grpc_port=grpc_port)
        self.text_collection = TEXT_COLLECTION
        self.visual_collection = VISUAL_COLLECTION
        self.agent_memory_collection = AGENT_MEMORY_COLLECTION
        self.notes_collection = KNOWLEDGE_NOTES_COLLECTION

    def ensure_collections(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.text_collection not in existing:
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config={
                    "dense": qm.VectorParams(size=TEXT_DENSE_DIM,
                                             distance=qm.Distance.COSINE),
                },
                sparse_vectors_config={
                    "bm25": qm.SparseVectorParams(
                        index=qm.SparseIndexParams(on_disk=False),
                    ),
                },
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100),
                on_disk_payload=True,
            )
        if self.visual_collection not in existing:
            self.client.create_collection(
                collection_name=self.visual_collection,
                vectors_config={
                    "dense": qm.VectorParams(size=VISUAL_DENSE_DIM,
                                             distance=qm.Distance.COSINE),
                },
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100),
                on_disk_payload=True,
            )
        if self.agent_memory_collection not in existing:
            self.client.create_collection(
                collection_name=self.agent_memory_collection,
                vectors_config={
                    "dense": qm.VectorParams(size=AGENT_MEMORY_DENSE_DIM,
                                             distance=qm.Distance.COSINE),
                },
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100),
                on_disk_payload=True,
            )
        if self.notes_collection not in existing:
            self.client.create_collection(
                collection_name=self.notes_collection,
                vectors_config={
                    "dense": qm.VectorParams(size=KNOWLEDGE_NOTES_DENSE_DIM,
                                             distance=qm.Distance.COSINE),
                },
                sparse_vectors_config={
                    "bm25": qm.SparseVectorParams(
                        index=qm.SparseIndexParams(on_disk=False),
                    ),
                },
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100),
                on_disk_payload=True,
            )
        for field in ("user_id", "session_id"):
            try:
                self.client.create_payload_index(
                    self.agent_memory_collection, field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        for field in ("user_id", "note_id", "type", "status"):
            try:
                self.client.create_payload_index(
                    self.notes_collection, field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        for field in ("file_id", "root_ids", "kind", "mime", "lang", "parser_version"):
            try:
                self.client.create_payload_index(
                    self.text_collection, field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        for field in ("file_id", "root_ids", "kind", "mime"):
            try:
                self.client.create_payload_index(
                    self.visual_collection, field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def upsert_text_chunks(self, points: Iterable[dict]) -> None:
        batch = []
        for p in points:
            sparse = p["sparse"]
            batch.append(qm.PointStruct(
                id=p["id"],
                vector={
                    "dense": p["dense"],
                    "bm25": qm.SparseVector(
                        indices=list(sparse["indices"]),
                        values=list(sparse["values"]),
                    ),
                },
                payload=p["payload"],
            ))
        if batch:
            self.client.upsert(self.text_collection, points=batch, wait=True)

    def upsert_visual_chunks(self, points: Iterable[dict]) -> None:
        batch = [
            qm.PointStruct(id=p["id"], vector={"dense": p["dense"]},
                           payload=p["payload"])
            for p in points
        ]
        if batch:
            self.client.upsert(self.visual_collection, points=batch, wait=True)

    def upsert_agent_memory(self, points: Iterable[dict]) -> None:
        batch = [
            qm.PointStruct(id=p["id"], vector={"dense": p["dense"]},
                           payload=p["payload"])
            for p in points
        ]
        if batch:
            self.client.upsert(self.agent_memory_collection, points=batch,
                               wait=True)

    def query_agent_memory(self, user_id: str, dense: list,
                           limit: int = 5) -> list[dict]:
        resp = self.client.query_points(
            collection_name=self.agent_memory_collection,
            query=dense,
            using="dense",
            query_filter=qm.Filter(must=[
                qm.FieldCondition(key="user_id",
                                  match=qm.MatchValue(value=str(user_id))),
            ]),
            limit=limit,
            with_payload=True,
        )
        hits = []
        for pt in resp.points:
            pl = pt.payload or {}
            hits.append({
                "text": pl.get("text", ""),
                "session_id": pl.get("session_id", ""),
                "chunk_no": pl.get("chunk_no", 0),
                "created_at": pl.get("created_at", 0),
                "score": pt.score,
            })
        return hits

    def delete_agent_memory(self, user_id: str, session_id: str) -> None:
        """Delete ALL agent-memory vectors of one session. Filters on BOTH
        user_id and session_id — cross-user isolation invariant."""
        self.client.delete(
            collection_name=self.agent_memory_collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="user_id",
                                  match=qm.MatchValue(value=str(user_id))),
                qm.FieldCondition(key="session_id",
                                  match=qm.MatchValue(value=str(session_id))),
            ])),
            wait=True,
        )

    def set_root_ids_for_file(self, file_id: str,
                              root_ids: list[str]) -> None:
        """Update root_ids + clear tombstone on BOTH text_chunks and
        visual_chunks for this file_id. Used by REVIVE and partial-root
        removal flows.

        `tombstone_file` flips both collections to root_ids=[] + tombstoned_at,
        so the symmetric revive must also touch both — otherwise visual_chunks
        for image/video/pdf_figure vectors would stay tombstoned even though
        their content is still reachable through the text collection.
        """
        for coll in (self.text_collection, self.visual_collection):
            self.client.set_payload(
                collection_name=coll,
                payload={"root_ids": root_ids, "tombstoned_at": None},
                points=qm.Filter(must=[
                    qm.FieldCondition(key="file_id",
                                      match=qm.MatchValue(value=file_id)),
                ]),
                wait=True,
            )

    def tombstone_file(self, file_id: str, tombstoned_at: int) -> None:
        for coll in (self.text_collection, self.visual_collection):
            self.client.set_payload(
                collection_name=coll,
                payload={"root_ids": [], "tombstoned_at": tombstoned_at},
                points=qm.Filter(must=[
                    qm.FieldCondition(key="file_id",
                                      match=qm.MatchValue(value=file_id)),
                ]),
                wait=True,
            )

    def delete_file(self, file_id: str) -> None:
        for coll in (self.text_collection, self.visual_collection):
            self.client.delete(
                collection_name=coll,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                    qm.FieldCondition(key="file_id",
                                      match=qm.MatchValue(value=file_id)),
                ])),
                wait=True,
            )

    def upsert_notes(self, points: Iterable[dict]) -> None:
        batch = []
        for p in points:
            sparse = p["sparse"]
            batch.append(qm.PointStruct(
                id=p["id"],
                vector={
                    "dense": p["dense"],
                    "bm25": qm.SparseVector(
                        indices=list(sparse["indices"]),
                        values=list(sparse["values"]),
                    ),
                },
                payload=p["payload"],
            ))
        if batch:
            self.client.upsert(self.notes_collection, points=batch, wait=True)

    def query_notes(self, user_id: str, dense: list, limit: int = 10,
                    statuses: list[str] | None = None,
                    types: list[str] | None = None) -> list[dict]:
        must = [qm.FieldCondition(key="user_id",
                                  match=qm.MatchValue(value=str(user_id)))]
        if statuses:
            must.append(qm.FieldCondition(
                key="status", match=qm.MatchAny(any=list(statuses))))
        if types:
            must.append(qm.FieldCondition(
                key="type", match=qm.MatchAny(any=list(types))))
        resp = self.client.query_points(
            collection_name=self.notes_collection,
            query=dense,
            using="dense",
            query_filter=qm.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        hits = []
        for pt in resp.points:
            pl = pt.payload or {}
            hits.append({
                "note_id": pl.get("note_id", ""),
                "chunk_no": pl.get("chunk_no", 0),
                "text": pl.get("text", ""),
                "type": pl.get("type", "note"),
                "status": pl.get("status", ""),
                "updated_at": pl.get("updated_at", 0),
                "score": pt.score,
            })
        return hits

    def delete_note(self, user_id: str, note_id: str) -> None:
        """Delete ALL vectors of one note. Filters on BOTH user_id and
        note_id — cross-user isolation invariant (mirrors agent_memory)."""
        self.client.delete(
            collection_name=self.notes_collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="user_id",
                                  match=qm.MatchValue(value=str(user_id))),
                qm.FieldCondition(key="note_id",
                                  match=qm.MatchValue(value=str(note_id))),
            ])),
            wait=True,
        )

    def scroll_captions(self, source: str, limit: int = 512,
                        offset: str | None = None) -> tuple[list[dict], str | None]:
        """Bulk-export caption text chunks by source (scroll pagination cursor pattern
        copied from backfill_mtime.py:65-72). Used by Photos' periodic diff pull, same
        code path for both backfill and incremental."""
        points, next_offset = self.client.scroll(
            collection_name=self.text_collection,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="root_ids", match=qm.MatchAny(any=[source])),
                qm.FieldCondition(key="kind", match=qm.MatchValue(value="caption")),
            ]),
            with_payload=["file_id", "text", "mtime_ms"],
            with_vectors=False,
            limit=limit,
            offset=offset,
        )
        items = [p.payload or {} for p in points]
        return items, next_offset

    def count_vectors(self) -> dict:
        return {
            "text": self.client.count(self.text_collection, exact=False).count,
            "visual": self.client.count(self.visual_collection, exact=False).count,
        }
