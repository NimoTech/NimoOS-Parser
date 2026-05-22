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
