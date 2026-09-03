"""Parser-side container-dir gate (mirrors NimoOS-Wiki pkg/ignore baseline).

Wiki stopped emitting events under /DATA/.system_data in July, but Parser
kept the records it had already indexed and, worse, the parser_version
drift sweep re-parsed them (2026-09-03: ~/.claude.json and docker container
logs ended up in text_chunks). Parser must not trust upstream alone.
"""
from parser.pathgate import has_container_ancestor


def test_system_data_ancestor_is_denied():
    assert has_container_ancestor("/DATA/.system_data/home/nimo/.claude.json")


def test_nested_container_dir_is_denied():
    assert has_container_ancestor("/DATA/docs/lost+found/x.md")
    assert has_container_ancestor("/DATA/Photos/@eaDir/thumb.jpg")


def test_trash_prefix_dirs_are_denied():
    assert has_container_ancestor("/DATA/.Trash-1000/files/a.md")


def test_regular_paths_are_allowed():
    assert not has_container_ancestor("/DATA/Documents/report.md")
    assert not has_container_ancestor("/mnt/usb/notes.txt")


def test_only_directory_segments_count():
    # a *file* whose basename happens to match a container dir is not gated
    assert not has_container_ancestor("/DATA/Documents/immich")
    assert not has_container_ancestor("/DATA/Documents/.system_data")
