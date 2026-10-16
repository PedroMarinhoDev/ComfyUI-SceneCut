"""Directory listings for the file picker of SceneCut Load Video (Path).

Read-only: names, sizes and dates of folders and video files, never file contents. Which parts of
the disk can be listed is decided by the caller (`roots`); `None` means anywhere.
"""
from __future__ import annotations

import os
import string
from typing import Optional

VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".m2v",
    ".ts", ".mts", ".m2ts", ".mxf", ".3gp", ".ogv", ".vob", ".dv", ".y4m",
}
MAX_ENTRIES = 5000


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(path))))


def within(path: str, roots: Optional[list[str]]) -> bool:
    if roots is None:
        return True
    p = _norm(path)
    for r in roots:
        r = _norm(r)
        if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
            return True
    return False


def system_roots() -> list[str]:
    """Filesystem roots: `/` on POSIX, existing drive letters on Windows."""
    if os.name != "nt":
        return ["/"]
    if hasattr(os, "listdrives"):
        return list(os.listdrives())
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


def list_dir(path: str, roots: Optional[list[str]] = None, show_all: bool = False,
             show_hidden: bool = False) -> dict:
    """Entries of `path`: sub-folders, then video files (or every file with `show_all`)."""
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(full):
        raise FileNotFoundError(f"not a folder: {full}")
    if not within(full, roots):
        raise PermissionError("this folder is outside the folders SceneCut may browse")

    dirs, files, truncated = [], [], False
    with os.scandir(full) as it:
        for entry in it:
            name = entry.name
            if not show_hidden and (name.startswith(".") or name.startswith("$")):
                continue
            try:
                if entry.is_dir():
                    dirs.append(name)
                elif entry.is_file():
                    ext = os.path.splitext(name)[1].lower()
                    if show_all or ext in VIDEO_EXTENSIONS:
                        st = entry.stat()
                        files.append({"name": name, "size": st.st_size, "mtime": int(st.st_mtime),
                                      "video": ext in VIDEO_EXTENSIONS})
            except OSError:
                continue
            if len(dirs) + len(files) >= MAX_ENTRIES:
                truncated = True
                break

    dirs.sort(key=str.casefold)
    files.sort(key=lambda f: f["name"].casefold())
    parent = os.path.dirname(full.rstrip(os.sep)) or full
    if parent == full or not within(parent, roots):
        parent = None
    return {"path": full, "parent": parent, "sep": os.sep, "dirs": dirs, "files": files,
            "truncated": truncated}
