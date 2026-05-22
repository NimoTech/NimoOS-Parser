from typing import Iterable

from qdrant_client import QdrantClient, models as qm


TEXT_COLLECTION = "text_chunks"
VISUAL_COLLECTION = "visual_chunks"

TEXT_DENSE_DIM = 1024
VISUAL_DENSE_DIM = 1152


class QdrantStore:
    def __init__(self, url: str, grpc_port: int = 6334) -> None:
        self.client = QdrantClient(url=url, prefer_grpc=True, grpc_port=grpc_port)
        self.text_collection = TEXT_COLLECTION
        self.visual_collection = VISUAL_COLLECTION

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

    def set_root_ids_for_file(self, collection: str, file_id: str,
                              root_ids: list[str]) -> None:
        self.client.set_payload(
            collection_name=collection,
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

    def count_vectors(self) -> dict:
        return {
            "text": self.client.count(self.text_collection, exact=False).count,
            "visual": self.client.count(self.visual_collection, exact=False).count,
        }
