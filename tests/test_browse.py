"""Listings behind the Load Video file picker."""
import os

import pytest

from scenecut.browse import list_dir, within


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "Films").mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / "a.MP4").write_bytes(b"x" * 10)
    (tmp_path / "b.mkv").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / ".hidden.mp4").write_bytes(b"x")
    return tmp_path


def test_lists_folders_and_videos_only(tree):
    d = list_dir(str(tree))
    assert d["dirs"] == ["Films"]
    assert [f["name"] for f in d["files"]] == ["a.MP4", "b.mkv"]
    assert d["files"][0]["size"] == 10 and d["files"][0]["video"]
    assert d["parent"] == os.path.dirname(str(tree))


def test_all_files_and_hidden(tree):
    names = [f["name"] for f in list_dir(str(tree), show_all=True, show_hidden=True)["files"]]
    assert names == [".hidden.mp4", "a.MP4", "b.mkv", "notes.txt"]
    assert ".cache" in list_dir(str(tree), show_hidden=True)["dirs"]


def test_roots_restrict_listing_and_parent(tree):
    roots = [str(tree)]
    d = list_dir(str(tree), roots)
    assert d["parent"] is None                          # cannot climb out of the root
    assert list_dir(str(tree / "Films"), roots)["parent"] == str(tree)
    with pytest.raises(PermissionError):
        list_dir(os.path.dirname(str(tree)), roots)
    assert not within(str(tree) + "-sibling", roots)    # prefix of the name is not inside


def test_missing_folder(tree):
    with pytest.raises(FileNotFoundError):
        list_dir(str(tree / "nope"))
