from parser.chunk_text import chunk_markdown, chunk_plain, chunk_source


def test_chunk_plain_short_text():
    chunks = chunk_plain("hello world. lorem ipsum.", target_tokens=400)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world. lorem ipsum."
    assert chunks[0]["offset_start"] == 0
    assert chunks[0]["offset_end"] == len("hello world. lorem ipsum.")
    assert chunks[0]["chunk_no"] == 0


def test_chunk_plain_drops_too_short():
    chunks = chunk_plain("hi", target_tokens=400, min_tokens=5)
    assert chunks == []


def test_chunk_plain_splits_long():
    text = "\n".join(f"paragraph {i} " + "word " * 200 for i in range(5))
    chunks = chunk_plain(text, target_tokens=500, overlap_tokens=50)
    assert len(chunks) >= 3
    # chunk_no monotonic
    assert [c["chunk_no"] for c in chunks] == list(range(len(chunks)))
    # ranges cover the whole text
    assert chunks[0]["offset_start"] == 0
    assert chunks[-1]["offset_end"] == len(text)


def test_chunk_markdown_preserves_heading_prefix():
    md = "# Title\n\nIntro paragraph.\n\n## Sub\n\nSubpara."
    chunks = chunk_markdown(md, target_tokens=400)
    assert any("Title" in c["text"] for c in chunks)
    assert any("Sub" in c["text"] for c in chunks)


def test_chunk_source_by_top_level_blocks():
    code = (
        "def alpha():\n    return 1\n\n"
        "def beta():\n    return 2\n\n"
        "class Gamma:\n    pass\n"
    )
    chunks = chunk_source(code, target_tokens=400)
    assert len(chunks) >= 3
    assert any("alpha" in c["text"] for c in chunks)
    assert any("Gamma" in c["text"] for c in chunks)


# --- prologue, oversized sections, heading path (audit 2026-09-03, Parser P10 / item C) ---

def test_chunk_markdown_keeps_text_before_first_heading():
    md = "Front matter and abstract paragraph that must stay searchable.\n\n# H1\n\nbody"
    chunks = chunk_markdown(md, target_tokens=400, min_tokens=2)
    assert chunks[0]["text"].startswith("Front matter"), chunks[0]
    assert chunks[0]["offset_start"] == 0
    assert chunks[0]["section"] == ""  # no heading above it
    assert md[chunks[0]["offset_start"]:chunks[0]["offset_end"]].strip() == chunks[0]["text"]


def test_chunk_markdown_splits_oversized_section_to_target_tokens():
    body = "\n\n".join("paragraph %d %s" % (i, "word " * 60) for i in range(40))  # ~9k tokens
    md = "# Big\n\n" + body
    chunks = chunk_markdown(md, target_tokens=200, min_tokens=2)
    assert len(chunks) > 5, "one heading section must not become one giant chunk"
    for c in chunks:
        assert len(c["text"]) // 4 <= 200 * 2, "chunk far above target_tokens: %d chars" % len(c["text"])
        assert c["section"] == "Big"
        assert c["text"] in md[c["offset_start"]:c["offset_end"]]
    assert len({c["section_no"] for c in chunks}) == 1, "sub-chunks share their section (parent)"
    assert [c["chunk_no"] for c in chunks] == list(range(len(chunks)))


def test_chunk_markdown_section_is_heading_path():
    md = "# Guide\n\nintro\n\n## Setup\n\nsteps\n\n### Linux\n\napt\n\n## Usage\n\nrun"
    chunks = chunk_markdown(md, target_tokens=400, min_tokens=1)
    by_first_line = {c["text"].splitlines()[0]: c for c in chunks}
    assert by_first_line["# Guide"]["section"] == "Guide"
    assert by_first_line["## Setup"]["section"] == "Guide > Setup"
    assert by_first_line["### Linux"]["section"] == "Guide > Setup > Linux"
    assert by_first_line["## Usage"]["section"] == "Guide > Usage"
    assert len({c["section_no"] for c in chunks}) == 4, "each heading is its own section"


def test_chunk_source_splits_oversized_block():
    code = "def huge():\n" + "".join("    x%d = %d  # %s\n" % (i, i, "c" * 40) for i in range(400))
    chunks = chunk_source(code, target_tokens=200, min_tokens=2)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c["text"]) // 4 <= 200 * 2
        assert c["text"] in code[c["offset_start"]:c["offset_end"]]
    assert len({c["section_no"] for c in chunks}) == 1


def test_chunk_plain_chunks_carry_section_fields():
    chunks = chunk_plain("hello world. lorem ipsum.", target_tokens=400)
    assert chunks[0]["section"] == ""
    assert chunks[0]["section_no"] == 0
