import re
from typing import Iterable


def _rough_token_count(s: str) -> int:
    # cheap approximation: 1 token ≈ 4 chars (works ok for mixed cn/en in MVP)
    return max(1, len(s) // 4)


def _emit(buf: str, offset_start: int, idx: int, min_tokens: int) -> dict | None:
    text = buf.strip()
    if not text or _rough_token_count(text) < min_tokens:
        return None
    return {
        "chunk_no": idx,
        "text": text,
        "offset_start": offset_start,
        "offset_end": offset_start + len(buf),
    }


def _split_positions(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans for each paragraph/line segment, preserving offsets."""
    # Try double-newline splits first
    pattern = r"\n\s*\n"
    spans = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    if not spans:
        # Fall back to single newlines
        spans = [(m.start(), m.end()) for m in re.finditer(r"\n", text)]
    if not spans:
        return [(0, len(text))]
    # Build segment spans between separators
    segments = []
    prev = 0
    for sep_start, sep_end in spans:
        segments.append((prev, sep_end))  # include separator in segment
        prev = sep_end
    if prev < len(text):
        segments.append((prev, len(text)))
    return segments


def chunk_plain(
    text: str, *, target_tokens: int = 600, overlap_tokens: int = 80,
    min_tokens: int = 5,
) -> list[dict]:
    chunks: list[dict] = []
    segments = _split_positions(text)
    buf = ""
    buf_start = 0
    cursor = 0
    idx = 0
    for seg_start, seg_end in segments:
        seg_text = text[seg_start:seg_end]
        if _rough_token_count(buf + seg_text) > target_tokens and buf:
            c = _emit(buf, buf_start, idx, min_tokens)
            if c:
                chunks.append(c)
                idx += 1
            # overlap: keep tail
            tail_chars = overlap_tokens * 4
            tail = buf[-tail_chars:] if len(buf) > tail_chars else buf
            buf = tail
            buf_start = seg_start - len(tail)
        buf += seg_text
        cursor = seg_end
    if buf:
        last = buf.rstrip("\n")
        c = _emit(last, buf_start, idx, min_tokens)
        if c:
            # snap offset_end to text length so callers can reconstruct ranges
            c["offset_end"] = len(text)
            chunks.append(c)
    return chunks


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_markdown(
    text: str, *, target_tokens: int = 600, min_tokens: int = 2,
) -> list[dict]:
    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        return chunk_plain(text, target_tokens=target_tokens,
                           min_tokens=min_tokens)
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((start, end, m.group(2).strip(), text[start:end]))
    chunks: list[dict] = []
    idx = 0
    for start, end, heading, body in sections:
        body_clean = body.strip()
        if _rough_token_count(body_clean) >= min_tokens:
            chunks.append({
                "chunk_no": idx,
                "text": body_clean,
                "offset_start": start,
                "offset_end": end,
            })
            idx += 1
    return chunks


_BLOCK_BOUNDARY = re.compile(r"^(?=def |class |async def |func |fn |type |impl )",
                              re.MULTILINE)


def chunk_source(
    text: str, *, target_tokens: int = 600, min_tokens: int = 5,
) -> list[dict]:
    boundaries = [m.start() for m in _BLOCK_BOUNDARY.finditer(text)]
    if not boundaries:
        return chunk_plain(text, target_tokens=target_tokens,
                           min_tokens=min_tokens)
    boundaries.append(len(text))
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    chunks: list[dict] = []
    idx = 0
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        block = text[start:end].rstrip()
        if _rough_token_count(block) >= min_tokens:
            chunks.append({
                "chunk_no": idx,
                "text": block,
                "offset_start": start,
                "offset_end": end,
            })
            idx += 1
    return chunks
