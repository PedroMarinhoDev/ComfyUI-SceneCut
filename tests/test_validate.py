"""Validator logic on hand-made features; no model, no video, no GPU."""
from fractions import Fraction

import numpy as np
import pytest

from scenecut import validate as V
from scenecut.media import VideoInfo
from scenecut.scenes import Cut, SceneList, Segment

CFG = V.ValidatorConfig()
INFO = VideoInfo(path="/x.mp4", width=320, height=240, fps=Fraction(30), frame_count=600,
                 duration=20.0, has_audio=False, codec="h264", pix_fmt="yuv420p")


def scenes(cuts, total=600):
    b = [0] + list(cuts) + [total]
    return SceneList(source=INFO, frame_count=total,
                     segments=[Segment(s, e, cut=Cut(frame=s) if s else None) for s, e in zip(b, b[1:])])


def feat(cut, s_cls=0.2, s_dense=0.3, s_adj=0.1, s_adj_dense=0.3, novelty=0.85, flash=None, **kw):
    f = {"cut": cut, "s_cls": s_cls, "s_dense": s_dense, "s_cls_stable": s_cls, "s_dense_stable": s_dense,
         "s_adj": s_adj, "s_adj_dense": s_adj_dense, "novelty": novelty, "pre_cont": 0.99,
         "post_cont": 0.99, "motion": 0.01, "pixel_sim": 0.05, "sscd": None, "flash_cut": None,
         "flash": flash or {"is_flash": False, "conf": 0.0, "amp": 0.0, "dur": 0, "baseline": 50.0,
                            "spike_frames": [], "boundary_hit": False}}
    f.update(kw)
    return f


def same_shot(cut):
    return feat(cut, s_cls=0.95, s_dense=0.9, s_adj=0.97, s_adj_dense=0.95, novelty=0.04, pixel_sim=0.8)


def test_adaptive_threshold_regimes():
    assert V.adaptive_threshold([0.3] * 5, CFG) == CFG.default_threshold         # too few cuts
    assert V.adaptive_threshold([0.90] * 20, CFG) == pytest.approx(0.91)         # flat: mean + bump
    spread = list(np.linspace(0.05, 0.4, 30)) + [0.95, 0.96]
    assert V.adaptive_threshold(spread, CFG) == CFG.thr_lo                       # clamped to floor


def frames_with_luma(values: dict, total=600, base=50.0):
    fr = V._Frames()
    for i in range(total):
        fr.luma[i] = values.get(i, base)
        fr.vhi[i] = values.get(i, base)
    return fr


def test_flash_event_spike_vs_step():
    spike = V.flash_event(frames_with_luma({99: 160.0, 100: 170.0}), 100, 600, CFG)
    assert spike["is_flash"] and spike["dur"] == 2 and spike["kind"] == "bright"
    assert spike["boundary_hit"] and 99 in spike["spike_frames"]

    step = V.flash_event(frames_with_luma({i: 170.0 for i in range(100, 600)}), 100, 600, CFG)
    assert not step["is_flash"]                     # brightness does not return: a real change

    flat = V.flash_event(frames_with_luma({}), 100, 600, CFG)
    assert not flat["is_flash"] and flat["conf"] == 0.0


def test_decide_cascade():
    thr = 0.8
    assert V.decide(feat(100), thr, CFG) == (True, "keep_hard_discontinuity")
    assert V.decide(same_shot(100), thr, CFG) == (False, "merge_same_shot")
    # same-shot score but adjacent frames differ a lot: refuse the hard merge
    assert V.decide(same_shot(100) | {"s_adj": 0.5}, thr, CFG)[0] is True
    fl = {"is_flash": True, "conf": 0.9, "amp": 100.0, "dur": 2, "baseline": 50.0,
          "spike_frames": [99, 100, 101], "boundary_hit": True}
    assert V.decide(same_shot(100) | {"flash": fl}, thr, CFG) == (False, "merge_flash")
    assert V.decide(feat(100, flash=fl, s_cls=0.1, s_dense=0.2), thr, CFG) == (True, "keep_flash_strong_cut")


def test_sscd_only_counts_near_threshold_or_for_flashes():
    thr = 0.8
    far = feat(100, sscd=0.99)                       # far below the threshold: SSCD ignored
    assert V.sscd_used(far, thr, CFG) is None
    near = feat(100, s_cls=0.75, s_dense=0.75, s_adj=0.9, s_adj_dense=0.9, novelty=0.1,
                pixel_sim=0.8, sscd=0.95)
    assert V.sscd_used(near, thr, CFG) == 0.95
    assert V.decide(near, thr, CFG) == (False, "merge_same_shot_sscd")


def test_apply_merges_and_records():
    sl = scenes([100, 200, 300])
    feats = {100: feat(100), 200: same_shot(200), 300: feat(300)}
    out = V.apply(sl, feats, V.config_with(threshold=0.8))
    assert out.cuts == [100, 300]
    rej = out.rejected[-1]
    assert rej["frame"] == 200 and rej["stage"] == "validate" and rej["reason"] == "merge_same_shot"
    assert rej["validation"]["decision"] == "merge"
    kept = out.segments[1].cut.validation
    assert kept["decision"] == "keep" and kept["reason"] == "keep_hard_discontinuity"
    assert out.history[-1]["stage"] == "validate" and out.history[-1]["params"]["thresholdUsed"] == 0.8


def test_short_guard_keeps_cuts_beside_short_shots():
    sl = scenes([100, 105])                          # a 5-frame shot (<= 0.3 s at 30 fps)
    feats = {100: same_shot(100), 105: same_shot(105)}
    out = V.apply(sl, feats, V.config_with(threshold=0.8))
    assert out.cuts == [100, 105]
    assert out.segments[1].cut.validation["guard"] == "short_guard_keep"


def test_transitions_are_not_judged():
    sl = scenes([100, 112, 300])
    sl.segments[1].kind, sl.segments[1].transition = "transition", "dissolve"
    assert V.eligible_cuts(sl) == [300]
    out = V.apply(sl, {300: same_shot(300)}, V.config_with(threshold=0.8))
    assert out.cuts == [100, 112]


def test_flash_sliver_merges_into_its_neighbour():
    fl = {"is_flash": False, "conf": 0.5, "amp": 80.0, "dur": 2, "baseline": 50.0,
          "spike_frames": [100, 101, 102], "boundary_hit": True}
    sl = scenes([101, 103])                          # 2-frame sliver at 101..102
    feats = {101: feat(101, flash=fl), 103: feat(103)}
    out = V.apply(sl, feats, V.config_with(threshold=0.8))
    assert len(out.segments) == 2
    assert out.rejected[-1]["stage"] == "validate_flash"


def test_flash_cut_follows_what_the_frames_look_like():
    """A flash at the start of a shot must not drag the previous shot's last frame into it."""
    import torch

    fr = V._Frames()
    a, b = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])
    for i in range(80, 121):
        fr.cls[i] = a if i < 101 else b
        fr.luma[i] = fr.vhi[i] = 50.0
    for i in (101, 102):                        # the flash: first frames of the new shot, brighter
        fr.luma[i] = fr.vhi[i] = 90.0
        fr.cls[i] = torch.nn.functional.normalize(b + 0.3 * a, dim=0)
    fe = V.flash_event(fr, 101, 600, CFG)
    assert fe["is_flash"] and fe["run_frames"] == [101, 102]
    assert V._flash_side_cut(fr, 101, 600, fe, CFG) == 101          # stays on the detector's frame
    assert V._flash_side_cut(fr, 100, 600, V.flash_event(fr, 100, 600, CFG), CFG) == 101


def test_a_cut_after_a_transition_is_compared_with_the_scene_before_it():
    """The frames just before a cut that ends a dissolve already show the next shot; the cut stays
    when frames from before the dissolve look different, and merges when they look the same."""
    sl = scenes([100, 300])
    sl.segments[2].cut.relation = "transition"
    win = V.lookback_windows(sl, CFG)
    assert list(win) == [300] and win[300] == (300 - 36 - 8, 300 - 36)        # 1.2 s back at 30 fps
    feats = {100: feat(100), 300: same_shot(300)}
    feats[300]["s_lookback"] = 0.2                                              # a different scene before
    out = V.apply(sl, feats, CFG)
    assert out.cuts == [100, 300] and out.segments[2].cut.validation["reason"] == "keep_after_transition"
    feats[300]["s_lookback"] = 0.9                                              # the same shot moving
    assert V.apply(sl, feats, CFG).cuts == [100]
    short = scenes([100, 120])                                                  # no room in the previous shot
    short.segments[2].cut.relation = "transition"
    assert V.lookback_windows(short, CFG) == {}
