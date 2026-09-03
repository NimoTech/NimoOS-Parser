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


def is_container_dir(basename: str) -> bool:
    if basename in CONTAINER_DIRS:
        return True
    return any(basename.startswith(p) for p in CONTAINER_DIR_PREFIXES)


def has_container_ancestor(path: str) -> bool:
    """True when any *directory* segment of `path` is a container dir.

    Only ancestors count: a file whose basename happens to equal a container
    name is not gated, matching Wiki's "container directory itself" rule.
    """
    parent = posixpath.dirname(posixpath.normpath(path))
    return any(is_container_dir(seg) for seg in parent.split("/") if seg)
