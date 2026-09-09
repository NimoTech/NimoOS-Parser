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


# --- GATED_PATH_PREFIXES: the agent notes layer is never a document source ---
from parser.pathgate import has_gated_prefix, is_gated_path


def test_notes_root_is_gated_for_every_user_dir():
    assert has_gated_prefix("/DATA/Notes/1/log.md")
    assert has_gated_prefix("/DATA/Notes/1/index.md")
    assert has_gated_prefix("/DATA/Notes/42/xeon-e3-tdp-3547e314.md")
    assert is_gated_path("/DATA/Notes/1/nested/deeper.md")


def test_notes_prefix_is_normalised_before_matching():
    assert has_gated_prefix("/DATA//Notes/1/./log.md")
    assert has_gated_prefix("/DATA/Documents/../Notes/1/log.md")


def test_notes_prefix_does_not_bleed_into_similar_names():
    assert not has_gated_prefix("/DATA/Notes")          # the dir itself, no trailing segment
    assert not has_gated_prefix("/DATA/NotesArchive/a.md")
    assert not has_gated_prefix("/DATA/Documents/Notes/a.md")
    assert not is_gated_path("/DATA/Documents/Notes/a.md")


def test_is_gated_path_still_covers_container_dirs():
    assert is_gated_path("/DATA/.system_data/x.json")
    assert not is_gated_path("/DATA/Documents/report.md")
