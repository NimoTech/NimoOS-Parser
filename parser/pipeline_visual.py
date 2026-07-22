"""VisualPipeline —— 照片/视频关键帧 → 英文 caption → text_chunks。

与 TextPipeline 的分工:TextPipeline 吃“文件”(wiki 事件驱动、走 allowlist、
分块),本管线吃“资产”(Photos 投喂、单块、不走 allowlist——图片扩展名
本来就被 allowlist 排除,这是刻意的双轨)。payload 结构与文档块完全同构,
Search 检索侧零改动。
"""
import logging
import uuid

log = logging.getLogger("parser.pipeline_visual")


def file_id_for(source: str, asset_id: str) -> str:
    # 与 wiki 文档的 sha256 file_id 命名空间隔离(“photos:<id>”)。
    return f"{source}:{asset_id}"


def build_index_text(caption: str, meta: dict) -> str:
    """caption + 拍摄元数据组装成入库文本。
    元数据不喂给 VLM(防幻觉),只作为独立一行拼进可检索文本。"""
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
        with open(image_path, "rb") as f:  # 不可读直接抛,由 job 层记败
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
        # 幂等:先清同 file_id 旧块(防 prompt 版本变化残留),再写新块。
        self.qstore.delete_file(file_id=file_id)
        self.qstore.upsert_text_chunks([point])
        log.info("visual ingest done file_id=%s caption_len=%d",
                 file_id, len(caption))

    def delete_asset(self, *, source: str, asset_id: str) -> None:
        # Photos 是资产权威源,删除即硬删(不走文档的 tombstone 宽限:
        # 误删可由 Photos 重投喂完整重建,无需 revive 语义)。
        self.qstore.delete_file(file_id=file_id_for(source, asset_id))
