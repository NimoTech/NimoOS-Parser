"""Sandbox endpoint for inspecting how Parser handles a specific file.

Users upload a file via the UI; the endpoint runs the same chunker the
indexing worker uses, optionally embeds chunks + computes similarity to
a query, then returns everything in one response. Nothing is written to
Qdrant or the local DB — pure preview.
"""
import math
import posixpath

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from parser.chunk_text import chunk_markdown, chunk_plain, chunk_source

router = APIRouter(prefix="/v1/parser/test", tags=["test"])


_MD_EXT = {".md", ".markdown"}
_SOURCE_EXT = {".py", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".java",
               ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
               ".kt", ".scala", ".sh", ".bash", ".sql", ".lua"}
_TEXT_EXT = {".txt", ".rst", ".html", ".htm", ".xml", ".json", ".yaml",
             ".yml", ".toml", ".ini", ".env", ".csv", ".tsv", ".log"}

_MAX_BYTES = 5 * 1024 * 1024   # 5 MiB cap to keep the sandbox fast
_PREVIEW_DIMS = 8              # how many dense dims to surface per chunk
_RERANK_TOP_K = 20             # only rerank the top-K by cosine sim


def _top_sparse(sparse: dict, k: int = 5) -> list[dict]:
    """Return top-k (by absolute value) sparse token weights for display."""
    indices = sparse.get("indices", [])
    values = sparse.get("values", [])
    pairs = sorted(zip(indices, values), key=lambda iv: -abs(iv[1]))[:k]
    return [{"token_id": int(i), "weight": round(float(v), 4)} for i, v in pairs]


def _cos_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_bge():
    from parser.device import current_device
    from parser.main import app_state
    from parser.model_bge_m3 import BGEM3
    return BGEM3.load(device=current_device(app_state.conn))


def _get_reranker():
    from parser.device import current_device
    from parser.main import app_state
    from parser.model_reranker import BGEReranker
    return BGEReranker.load(device=current_device(app_state.conn))


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    query: str | None = Form(default=None),
    embed: bool = Form(default=True),
    rerank: bool = Form(default=False),
) -> dict:
    raw = await file.read()
    if len(raw) > _MAX_BYTES:
        raise HTTPException(
            413, f"file too large ({len(raw)} B); test sandbox max {_MAX_BYTES} B",
        )

    ext = posixpath.splitext(file.filename or "")[1].lower()
    if ext in _MD_EXT:
        mime, chunker_kind = "text/markdown", "markdown"
    elif ext in _SOURCE_EXT:
        mime, chunker_kind = "text/x-source", "source"
    elif ext in _TEXT_EXT or ext == ".pdf":
        # .pdf is intentionally treated as text here — sandbox does NOT
        # ship a PDF extractor; the user gets a raw decode, which is
        # informative for debugging "why does my PDF index poorly".
        mime, chunker_kind = "text/plain", "plain"
    else:
        raise HTTPException(
            400,
            f"extension {ext!r} not supported in test sandbox; "
            "use .md / source code / .txt / .html / .json / .csv / .log "
            "(PDF accepted as raw text decode for inspection only)",
        )

    text = raw.decode("utf-8", errors="replace")

    # min_tokens=2 for sandbox so even short snippets produce chunks the
    # user can inspect; production pipelines stay at the higher defaults.
    if chunker_kind == "markdown":
        chunks = chunk_markdown(text, min_tokens=2)
    elif chunker_kind == "source":
        chunks = chunk_source(text, min_tokens=2)
    else:
        chunks = chunk_plain(text, min_tokens=2)

    out_chunks: list[dict] = []
    vectors: list[list[float]] = []

    if embed and chunks:
        bge = _get_bge()
        embeddings = bge.embed_text([c["text"] for c in chunks])
        for c, e in zip(chunks, embeddings):
            dense = e["dense"]
            vectors.append(dense)
            out_chunks.append({
                "chunk_no": c["chunk_no"],
                "text": c["text"],
                "token_count": max(1, len(c["text"]) // 4),
                "offset_start": c["offset_start"],
                "offset_end": c["offset_end"],
                "dense_preview": [round(float(v), 4) for v in dense[:_PREVIEW_DIMS]],
                "sparse_top_terms": _top_sparse(e["sparse"], k=5),
            })
    else:
        for c in chunks:
            out_chunks.append({
                "chunk_no": c["chunk_no"],
                "text": c["text"],
                "token_count": max(1, len(c["text"]) // 4),
                "offset_start": c["offset_start"],
                "offset_end": c["offset_end"],
            })

    result: dict = {
        "mime": mime,
        "filename": file.filename,
        "size": len(raw),
        "text_length": len(text),
        "chunk_count": len(chunks),
        "chunks": out_chunks,
    }

    if query and chunks and embed:
        bge = _get_bge()
        q_emb = bge.embed_text([query])[0]
        q_dense = q_emb["dense"]
        scored = [
            {"chunk_no": c["chunk_no"], "cos_sim": _cos_sim(q_dense, vectors[i])}
            for i, c in enumerate(chunks)
        ]
        scored.sort(key=lambda s: -s["cos_sim"])

        if rerank and scored:
            top = scored[:_RERANK_TOP_K]
            top_texts = [
                {"id": str(s["chunk_no"]), "text": chunks[s["chunk_no"]]["text"]}
                for s in top
            ]
            try:
                rr = _get_reranker()
                rr_scores = rr.rerank(query, top_texts)
                rr_map = {int(r["id"]): r["score"] for r in rr_scores}
                for s in top:
                    s["rerank_score"] = rr_map.get(s["chunk_no"])
                top.sort(key=lambda s: -(s.get("rerank_score") or s["cos_sim"]))
                scored = top + scored[_RERANK_TOP_K:]
            except Exception as e:
                # Reranker has a known FlagEmbedding/transformers compat bug;
                # surface it instead of crashing the whole analyze response.
                result["rerank_error"] = str(e)

        result["query"] = query
        result["scored"] = scored

    return result
