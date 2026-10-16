"""Viewer review edits, cut fixes and the cut-placement rule; hand-made inputs and a tiny video."""
from fractions import Fraction

import numpy as np
import pytest

from scenecut.boundary import best_split
from scenecut.media import VideoInfo
from scenecut.review import apply_edits, apply_review, auto_fix, fingerprint, manual_fix
from scenecut.scenes import Cut, SceneList, Segment

INFO = VideoInfo(path="/x.mp4", width=64, height=48, fps=Fraction(25), frame_count=150,
                 duration=6.0, has_audio=False, codec="h264", pix_fmt="yuv420p")


def scenes(cuts, total=150, info=INFO):
    b = [0] + list(cuts) + [total]
    return SceneList(source=info, frame_count=total,
                     segments=[Segment(s, e, cut=Cut(frame=s) if s else None) for s, e in zip(b, b[1:])])


def test_best_split_finds_the_switch_and_prefers_the_current_cut():
    # frames 0-2 look like A, 3-5 like B
    to_a = [0.9, 0.9, 0.9, 0.2, 0.2, 0.2]
    to_b = [0.2, 0.2, 0.2, 0.9, 0.9, 0.9]
    assert best_split(100, to_a, to_b)[0] == 103
    assert best_split(100, to_a, to_b, prefer=102, margin=0.0)[0] == 103
    # an ambiguous frame stays where the detector put the cut
    to_a[3], to_b[3] = 0.55, 0.55
    assert best_split(100, to_a, to_b, prefer=104, margin=0.05)[0] == 104
    assert best_split(100, to_a, to_b, prefer=103, margin=0.05)[0] == 103


def test_edits_move_cuts_and_skip_what_no_longer_fits():
    sl = scenes([50, 100])
    out, n = apply_edits(sl, [{"cut": 50, "to": 52, "mode": "auto"}, {"cut": 52, "to": 51},
                              {"cut": 77, "to": 78}, {"cut": 100, "to": 200}])
    assert n == 2 and out.cuts == [51, 100]
    assert out.segments[1].cut.review == {"from": 50, "mode": "manual"}
    assert sl.cuts == [50, 100]                            # the input is untouched
    out.check()


def test_review_applies_only_to_its_own_scene_list():
    sl = scenes([50, 100])
    review = {"base": fingerprint(sl), "edits": [{"cut": 100, "to": 98}], "selection": [2, 0, 9]}
    out, status = apply_review(sl, review)
    assert status == {"state": "applied", "edits": 1, "selected": 2}
    assert out.cuts == [50, 98] and out.selection == [0, 2]
    assert out.history[-1]["stage"] == "review"
    d = out.to_dict()
    assert [s["index"] for s in d["selectedScenes"]] == [0, 2] and d["selectedScenes"][1]["start"] == 98

    other = scenes([60, 100])
    same, status = apply_review(other, review)
    assert status["state"] == "stale" and same is other
    assert apply_review(sl, "")[1] == {"state": "empty"}


def test_manual_fix_joins_the_nearest_cut():
    sl = scenes([50, 100])
    assert manual_fix(sl, 50) == {"index": 1, "cut": 50, "to": 51, "mode": "manual"}    # first frame -> scene before
    assert manual_fix(sl, 52)["to"] == 53                                              # three frames join it
    assert manual_fix(sl, 98) == {"index": 2, "cut": 100, "to": 98, "mode": "manual"}  # near the end -> scene after
    assert manual_fix(sl, 10)["to"] == 10                                              # first scene: only its end
    assert "error" in manual_fix(scenes([]), 10)


def test_auto_fix_moves_a_cut_that_is_a_frame_early(tmp_path):
    import av

    # two flat colours with a gradient; the real cut is at frame 40
    path = str(tmp_path / "two.mkv")
    with av.open(path, "w") as out:
        s = out.add_stream("libx264rgb", rate=25)
        s.width, s.height, s.pix_fmt = 64, 48, "rgb24"
        s.options = {"qp": "0"}
        yy, xx = np.mgrid[0:48, 0:64]
        for i in range(80):
            img = np.zeros((48, 64, 3), np.uint8)
            if i < 40:
                img[..., 0] = (xx * 4).clip(0, 255)
                img[..., 2] = 40 + i
            else:
                img[..., 1] = (yy * 5).clip(0, 255)
                img[..., 2] = 200 - i
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    info = VideoInfo(**{**INFO.__dict__, "path": path, "frame_count": 80})
    sl = scenes([39], total=80, info=info)                # the detector put it one frame early
    r = auto_fix(sl, 39, path)
    assert (r["cut"], r["to"]) == (39, 40)
    good = scenes([40], total=80, info=info)
    assert auto_fix(good, 41, path)["to"] == 40           # a right cut stays
    assert "error" in auto_fix(good, 75, path)            # no cut near the playhead
