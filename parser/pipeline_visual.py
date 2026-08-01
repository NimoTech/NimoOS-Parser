"""VisualPipeline - photo/video keyframes -> English caption -> text_chunks.

Division of labor with TextPipeline: TextPipeline handles "files" (wiki
event-driven, goes through the allowlist, chunked), this pipeline handles
"assets" (fed by Photos, single chunk, bypasses the allowlist - image
extensions are already excluded from the allowlist, so this dual-track split
is deliberate). The payload structure is fully isomorphic to document chunks,
so the Search retrieval side needs zero changes.
"""
import logging
import uuid

log = logging.getLogger("parser.pipeline_visual")


def file_id_for(source: str, asset_id: str) -> str:
    # Namespaced apart from wiki documents' sha256 file_id ("photos:<id>").
    return f"{source}:{asset_id}"


def build_index_text(caption: str, meta: dict) -> str:
    """Assemble caption + shooting metadata into the text to be indexed.
    Metadata is never fed to the VLM (to avoid hallucination); it's only
    appended as a separate line into the searchable text."""
    parts = [caption.strip()]
    taken = (meta or {}).get("taken_at", "")
    place = (meta or {}).get("place", "")
    fields = [x for x in (taken, place) if x]
    if fields:
        parts.append("Taken: " + ", ".join(fields))
    return "\n".join(parts)


class VisualPipeline:
    def __init__(self, conn, *, qstore, embedder, caption_backend,
                 parser_version: str) -> None:
        self.conn = conn
        self.qstore = qstore
        self.embedder = embedder
        self.caption_backend = caption_backend
        self.parser_version = parser_version

    def ingest_asset(self, *, source: str, asset_id: str, image_path: str,
                     mime: str, meta: dict, now_ms: int) -> None:
        with open(image_path, "rb") as f:  # unreadable -> raise directly, job layer records the failure
            image_bytes = f.read()
        caption = self.caption_backend.caption(image_bytes)
        text = build_index_text(caption, meta)
        (emb,) = self.embedder.embed_text([text])
        file_id = file_id_for(source, asset_id)
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_OID, f"{file_id}:caption:0"))
        point = {
            "id": point_id,
            "dense": emb["dense"],
            "sparse": emb["sparse"],
            "payload": {
                "file_id": file_id,
                "root_ids": [source],
                "kind": "caption",
                "mime": mime,
                "chunk_no": 0,
                "text": text,
                "offset_start": 0,
                "offset_end": len(text),
                "lang": "en",
                "parser_version": self.parser_version,
                "embed_model_version": self.embedder.version,
                "source_model_version": self.caption_backend.version,
                "indexed_at": now_ms,
                "mtime_ms": now_ms,
                "tombstoned_at": None,
            },
        }
        # Idempotent: clear old chunks with the same file_id first (avoids stale
        # leftovers when the prompt version changes), then write the new chunk.
        self.qstore.delete_file(file_id=file_id)
        self.qstore.upsert_text_chunks([point])
        log.info("visual ingest done file_id=%s caption_len=%d",
                 file_id, len(caption))

    def delete_asset(self, *, source: str, asset_id: str) -> None:
        # Photos is the authoritative source for assets, so delete is a hard
        # delete (skips the document tombstone grace period: an accidental
        # delete can be fully rebuilt by Photos re-feeding it, no revive
        # semantics needed).
        self.qstore.delete_file(file_id=file_id_for(source, asset_id))
