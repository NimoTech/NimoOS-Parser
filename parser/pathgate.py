"""Container-directory gate: paths Parser must never index, whatever upstream
(Wiki file events, a PARSER_VERSION drift sweep, an admin rescan) hands it.

Mirrors the non-configurable baseline in NimoOS-Wiki `pkg/ignore/ignore.go`
(NAS / OS noise dirs, app-owned data dirs, `.system_data`). Wiki already
filters these at the source, but Parser keeps its own copy of the rule so a
record that slipped in earlier — or arrives via a path that bypasses Wiki —
is still refused and retired. Keep the two lists in sync.
"""
import posixpath

CONTAINER_DIRS = frozenset({
    "@eaDir", "#recycle", "@__thumb",
    ".AppleDouble", ".fseventsd", ".Spotlight-V100", ".Trashes",
    ".DocumentRevisions-V100", "__MACOSX",
    "Network Trash Folder", "Temporary Items",
    "lost+found", ".snapshots",
    "immich", ".system_data",
})

CONTAINER_DIR_PREFIXES = (".Trash-",)

# Absolute directory prefixes Parser never indexes as documents, in any root.
# /DATA/Notes is the agent's knowledge-notes layer (NimoOS-AI
# agent/notes/store.py DEFAULT_NOTES_ROOT). Every note there is already
# embedded into the dedicated `notes` collection, so indexing the .md files
# again only hands the answer model its own past answers plus the layer's
# log.md/index.md bookkeeping (2026-09-09 Intel2408 ask eval: 18% of evidence
# items came from here, 18/55 questions saw notes distilled earlier in the
# same run). Like CONTAINER_DIRS this beats an explicit allow rule.
GATED_PATH_PREFIXES = ("/DATA/Notes/",)


def is_container_dir(basename: str) -> bool:
    if basename in CONTAINER_DIRS:
        return True
    return any(basename.startswith(p) for p in CONTAINER_DIR_PREFIXES)


def has_gated_prefix(path: str) -> bool:
    """True when `path` (normalised) lies under one of GATED_PATH_PREFIXES."""
    norm = posixpath.normpath(path)
    return any(norm.startswith(p) for p in GATED_PATH_PREFIXES)


def is_gated_path(path: str) -> bool:
    """Single non-configurable gate: container-dir ancestor OR gated prefix."""
    return has_container_ancestor(path) or has_gated_prefix(path)


def has_container_ancestor(path: str) -> bool:
    """True when any *directory* segment of `path` is a container dir.

    Only ancestors count: a file whose basename happens to equal a container
    name is not gated, matching Wiki's "container directory itself" rule.
    """
    parent = posixpath.dirname(posixpath.normpath(path))
    return any(is_container_dir(seg) for seg in parent.split("/") if seg)
