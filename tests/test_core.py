"""Scene-list stages on hand-made inputs; no model, no video."""
import json
from fractions import Fraction

import numpy as np
import pytest

from scenecut.detect import Boundary, RawDetection
from scenecut.media import VideoInfo
from scenecut.refine import (build_scene_list, handle_transitions, merge_short, merge_ultra_short,
                             snap_cuts)
from scenecut.scenes import SceneList

INFO = VideoInfo(path="/x.mp4", width=320, height=240, fps=Fraction(30), frame_count=300,
                 duration=10.0, has_audio=False, codec="h264", pix_fmt="yuv420p")

SHOT, DISSOLVE = 0, 1
NEW, HARD, TRANS = 0, 1, 3


def raw(ends, intra=None, inter=None, total=300, diff=None):
    intra = intra or [SHOT] * len(ends)
    inter = inter or [NEW] + [HARD] * (len(ends) - 1)
    bs = [Boundary(e, i, r, 0.9) for e, i, r in zip(ends, intra, inter)]
    d = np.zeros(total, np.float32) if diff is None else diff
    return RawDetection(frame_count=total, boundaries=bs, diff=d, luma=np.zeros(total, np.float32))


def spans(sl):
    return [(s.start, s.end, s.kind) for s in sl.segments]


def test_build_labels_transitions_and_fills_tail():
    sl = build_scene_list(raw([100, 112, 250], intra=[SHOT, DISSOLVE, SHOT],
                              inter=[NEW, TRANS, TRANS]), INFO)
    assert spans(sl) == [(0, 100, "shot"), (100, 112, "transition"), (112, 300, "shot")]
    assert sl.segments[1].transition == "dissolve"
    assert sl.segments[2].cut.confidence == pytest.approx(0.9)


@pytest.mark.parametrize("mode,expected", [
    ("merge_previous", [(0, 112, "shot"), (112, 300, "shot")]),
    ("merge_next", [(0, 100, "shot"), (100, 300, "shot")]),
    ("split", [(0, 106, "shot"), (106, 300, "shot")]),
    ("keep", [(0, 100, "shot"), (100, 112, "transition"), (112, 300, "shot")]),
])
def test_transition_modes(mode, expected):
    sl = build_scene_list(raw([100, 112, 300], intra=[SHOT, DISSOLVE, SHOT]), INFO)
    out = handle_transitions(sl, mode)
    assert spans(out) == expected
    if mode != "keep":
        assert any(a["transition"] == "dissolve" for s in out.segments for a in s.absorbed)
        assert out.rejected, "the removed boundary must be recorded"


def test_merge_short_prefers_longer_neighbour_and_records():
    sl = build_scene_list(raw([100, 105, 140, 300]), INFO)       # 5-frame shot between 100 and 35
    out = merge_short(sl, 12)
    assert spans(out) == [(0, 105, "shot"), (105, 140, "shot"), (140, 300, "shot")]
    assert out.rejected[-1]["stage"] == "min_length" and out.rejected[-1]["frame"] == 100


def test_merge_short_never_touches_transitions():
    sl = build_scene_list(raw([100, 104, 108, 300], intra=[SHOT, DISSOLVE, SHOT, SHOT]), INFO)
    out = merge_short(sl, 12)
    assert spans(out)[1] == (100, 104, "transition")
    assert spans(out)[2] == (104, 300, "shot")                   # 4-frame shot merged forward


def test_snap_moves_to_sharpest_frame_within_radius():
    diff = np.zeros(300, np.float32)
    diff[102] = 50.0                                             # the real cut is 2 frames later
    sl = build_scene_list(raw([100, 300], diff=diff), INFO)
    out = snap_cuts(sl, 2)
    assert out.cuts == [102]
    assert out.segments[1].cut.detected_frame == 100
    assert snap_cuts(sl, 1).cuts == [100]                        # flat window: no evidence, no move


def test_ultra_short_merges_fragments():
    sl = build_scene_list(raw([100, 101, 300]), INFO)
    out = merge_ultra_short(sl, 0.05)                             # 0.05 s at 30 fps = 2 frames
    assert len(out.segments) == 2


def test_json_round_trip():
    sl = handle_transitions(build_scene_list(raw([100, 112, 300], intra=[SHOT, DISSOLVE, SHOT]),
                                             INFO), "merge_previous")
    d = json.loads(json.dumps(sl.to_dict(include_signals=True)))
    back = SceneList.from_dict(d)
    assert spans(back) == spans(sl)
    assert back.fps == Fraction(30)
    assert np.array_equal(back.signals["diff"], sl.signals["diff"])
    assert back.rejected == sl.rejected
