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


def _with_section(chunks: list[dict], section: str, section_no: int | None) -> list[dict]:
    """Stamp section metadata on chunks. section_no=None means "each chunk is
    its own section" (plain text without structure)."""
    for c in chunks:
        c["section"] = section
        c["section_no"] = c["chunk_no"] if section_no is None else section_no
    return chunks


def chunk_plain(
    text: str, *, target_tokens: int = 600, overlap_tokens: int = 80,
    min_tokens: int = 5,
) -> list[dict]:
    return _with_section(
        _chunk_plain_raw(text, target_tokens=target_tokens,
                         overlap_tokens=overlap_tokens, min_tokens=min_tokens),
        "", None)


def _chunk_plain_raw(
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


def _split_section(text: str, start: int, end: int, *, target_tokens: int,
                   min_tokens: int, section: str, section_no: int,
                   chunk_no: int) -> list[dict]:
    """Turn text[start:end] into one or more chunks of at most ~target_tokens
    each, all stamped with the same section/section_no (the "parent"), with
    offsets expressed against the whole document. A section that already fits
    stays a single chunk so heading + body embed together."""
    body = text[start:end]
    stripped = body.strip()
    if not stripped or _rough_token_count(stripped) < min_tokens:
        return []
    if _rough_token_count(stripped) <= target_tokens:
        return [{
            "chunk_no": chunk_no, "text": stripped,
            "offset_start": start, "offset_end": end,
            "section": section, "section_no": section_no,
        }]
    out = []
    for c in _chunk_plain_raw(body, target_tokens=target_tokens,
                              min_tokens=min_tokens):
        out.append({
            "chunk_no": chunk_no + len(out), "text": c["text"],
            "offset_start": start + c["offset_start"],
            "offset_end": start + c["offset_end"],
            "section": section, "section_no": section_no,
        })
    return out


def chunk_markdown(
    text: str, *, target_tokens: int = 600, min_tokens: int = 2,
) -> list[dict]:
    """Chunk markdown by heading sections.

    - Text before the first heading (front matter, abstract, lead paragraph)
      is a section of its own with section == "" — it used to be dropped.
    - A section longer than target_tokens is split into several chunks that
      share one section_no; embedding truncates at ~1024 tokens, so a
      9k-token section as one chunk left ~90% of its text unsearchable.
    - `section` is the heading path ("Guide > Setup > Linux"); `section_no`
      is the section ordinal in the document. Together with file_id they give
      Search a stable parent id for merging sibling chunks back into their
      section.
    """
    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        return chunk_plain(text, target_tokens=target_tokens,
                           min_tokens=min_tokens)
    chunks: list[dict] = []
    section_no = 0
    prologue_end = matches[0].start()
    if text[:prologue_end].strip():
        chunks.extend(_split_section(
            text, 0, prologue_end, target_tokens=target_tokens,
            min_tokens=min_tokens, section="", section_no=section_no,
            chunk_no=len(chunks)))
        section_no += 1
    stack: list[tuple[int, str]] = []  # (level, heading) path to current section
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.extend(_split_section(
            text, start, end, target_tokens=target_tokens,
            min_tokens=min_tokens, section=" > ".join(h for _, h in stack),
            section_no=section_no, chunk_no=len(chunks)))
        section_no += 1
    return chunks


_BLOCK_BOUNDARY = re.compile(r"^(?=def |class |async def |func |fn |type |impl )",
                              re.MULTILINE)


def chunk_source(
    text: str, *, target_tokens: int = 600, min_tokens: int = 5,
) -> list[dict]:
    """Chunk source code by top-level block. Each block is a section; an
    oversized block is split like an oversized markdown section."""
    boundaries = [m.start() for m in _BLOCK_BOUNDARY.finditer(text)]
    if not boundaries:
        return chunk_plain(text, target_tokens=target_tokens,
                           min_tokens=min_tokens)
    boundaries.append(len(text))
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    chunks: list[dict] = []
    section_no = 0
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        first_line = text[start:end].lstrip().split("\n", 1)[0].strip()
        made = _split_section(
            text, start, end, target_tokens=target_tokens,
            min_tokens=min_tokens, section=first_line[:120],
            section_no=section_no, chunk_no=len(chunks))
        if made:
            chunks.extend(made)
            section_no += 1
    return chunks
