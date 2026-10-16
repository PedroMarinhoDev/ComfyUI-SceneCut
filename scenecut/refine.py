"""Stages that turn the detector's proposals into a clean shot list.

Each takes a SceneList and returns a modified copy; removed boundaries are recorded, never lost.

  build_scene_list   detector ranges -> segments (shots and labelled transitions)
  handle_transitions fold transitions into neighbouring shots, split them, or keep them
  merge_short        shots shorter than a floor are merged into their longer neighbour
  snap_cuts          move each hard cut to the frame with the largest pixel change nearby
  merge_ultra_short  final clean-up of fragments of a couple of frames
"""
from __future__ import annotations

import numpy as np

from .detect import RawDetection
from .media import VideoInfo
from .scenes import Cut, SceneList, Segment

TRANSITION_MODES = ("merge_previous", "merge_next", "split", "keep")


# OmniShotCut's inter labels, as a cut's relation: how the segment after the cut begins.
RELATIONS = {"hard_cut": "hard_cut", "sudden_jump": "sudden_jump", "transition": "transition",
             "transition_source": "transition"}


def _relation(prev, b) -> str:
    """Prefer the label from the window that saw this cut in its trusted region (`prev.next_inter`);
    a range's own label is `new_start` whenever its start fell before its window."""
    for label in (prev.next_inter_label if prev is not None else None, b.inter_label):
        if label in RELATIONS:
            return RELATIONS[label]
    return "hard_cut"


def build_scene_list(raw: RawDetection, info: VideoInfo) -> SceneList:
    """One segment per detector range. A range's own label says whether it is a shot or a
    transition; the confidence of the range that ends at a boundary is that boundary's
    confidence."""
    total = raw.frame_count
    segments: list[Segment] = []
    prev = None
    for start, end, b in raw.ranges():
        kind = "shot" if b.intra_label in ("shot", "padding") else "transition"
        cut = None
        if segments:
            cut = Cut(frame=start, source="detector", relation=_relation(prev, b),
                      confidence=prev.confidence)
        segments.append(Segment(start=start, end=end, kind=kind,
                                transition=b.intra_label if kind == "transition" else None, cut=cut))
        prev = b
    if not segments:
        segments = [Segment(0, total)]
    elif segments[-1].end < total:
        segments[-1].end = total
    # Probe and decoder disagree on length only for broken files; the decoder is the truth.
    if info.frame_count != total:
        info = VideoInfo(**{**info.__dict__, "frame_count": total})
    sl = SceneList(source=info, frame_count=total, segments=segments,
                   signals={"diff": raw.diff, "luma": raw.luma})
    sl.history.append({"stage": "detect", "params": dict(raw.params), "before": len(segments),
                       "after": len(segments),
                       "transitions": sum(s.kind == "transition" for s in segments),
                       "rejectedLowConfidence": raw.rejected_low_confidence})
    sl.check()
    return sl


def handle_transitions(sl: SceneList, mode: str = "merge_previous") -> SceneList:
    """merge_previous: a transition becomes the tail of the shot before it (the cut lands where
                       the next shot is fully in).
       merge_next:     it becomes the head of the shot after it (the cut lands where the
                       transition starts).
       split:          the cut goes in the middle of the transition.
       keep:           transitions stay as their own segments."""
    if mode not in TRANSITION_MODES:
        raise ValueError(f"transition mode must be one of {TRANSITION_MODES}")
    out = sl.copy()
    before = len(out.segments)
    n = sum(s.kind == "transition" for s in out.segments)
    if mode == "keep" or n == 0:
        out.record("transitions", {"mode": mode}, before)
        return out

    i = 0
    while i < len(out.segments):
        seg = out.segments[i]
        if seg.kind != "transition" or len(out.segments) == 1:
            i += 1
            continue
        has_prev, has_next = i > 0, i < len(out.segments) - 1
        if mode == "split" and has_prev and has_next:
            prev, nxt = out.segments[i - 1], out.segments[i + 1]
            mid = (seg.start + seg.end) // 2
            mid = min(max(mid, prev.start + 1), nxt.end - 1)
            record = {"start": seg.start, "end": seg.end, "transition": seg.transition}
            out.rejected.append({"frame": seg.start, "stage": "transitions",
                                 "reason": f"{seg.transition} split", **_cut_rec(seg)})
            out.rejected.append({"frame": nxt.start, "stage": "transitions",
                                 "reason": f"{seg.transition} split", **_cut_rec(nxt)})
            prev.end = mid
            prev.absorbed.append(record)
            nxt.start = mid
            nxt.absorbed.insert(0, record)
            nxt.cut = Cut(frame=mid, source="transition", relation="transition",
                          confidence=(nxt.cut.confidence if nxt.cut else None))
            del out.segments[i]
            continue
        # merge_previous, or the transition sits at the very start/end of the video
        if (mode == "merge_previous" and has_prev) or not has_next:
            out.remove_boundary(i, "transitions", f"{seg.transition} merged into previous shot")
            i = max(0, i - 1)
        else:
            out.remove_boundary(i + 1, "transitions", f"{seg.transition} merged into next shot")
    out.record("transitions", {"mode": mode}, before, transitions=n)
    out.check()
    return out


def _cut_rec(seg) -> dict:
    return {"cut": seg.cut.to_dict()} if seg.cut is not None else {}


def merge_short(sl: SceneList, min_frames: int, stage: str = "min_length") -> SceneList:
    """Merge every shot shorter than `min_frames` into a neighbouring shot, the longer one when
    both are shots. Transitions are never merged here and never absorb a short shot."""
    out = sl.copy()
    before = len(out.segments)
    min_frames = int(min_frames)
    segs = out.segments
    i = 0
    while min_frames > 1 and i < len(segs) and len(segs) > 1:
        seg = segs[i]
        if seg.kind != "shot" or seg.length >= min_frames:
            i += 1
            continue
        prev_ok = i > 0 and segs[i - 1].kind == "shot"
        next_ok = i < len(segs) - 1 and segs[i + 1].kind == "shot"
        if not prev_ok and not next_ok:
            i += 1
            continue
        reason = f"shot of {seg.length} frames < {min_frames}"
        if prev_ok and (not next_ok or segs[i - 1].length >= segs[i + 1].length):
            out.remove_boundary(i, stage, reason)
            i = max(0, i - 1)
        else:
            out.remove_boundary(i + 1, stage, reason)
            if i > 0:
                i -= 1
    out.record(stage, {"minFrames": min_frames}, before)
    out.check()
    return out


def snap_cuts(sl: SceneList, radius: int = 2) -> SceneList:
    """Move each boundary between two shots to the frame within `radius` whose difference to its
    predecessor is largest (the detector localises a cut to within a frame or two; the pixels say
    exactly where). Uses the per-frame difference signal recorded during detection, so it costs
    no decoding. Boundaries touching a transition segment are left where the model put them."""
    out = sl.copy()
    before = len(out.segments)
    diff = out.signals.get("diff")
    if radius <= 0 or diff is None or len(diff) != out.frame_count or len(out.segments) < 2:
        out.record("snap", {"radius": radius, "skipped": True}, before, moved=0)
        return out
    n = out.frame_count
    moved = 0
    segs = out.segments
    for i in range(1, len(segs)):
        seg, prev = segs[i], segs[i - 1]
        c = seg.start
        best = c
        if seg.kind == "shot" and prev.kind == "shot":
            lo, hi = max(1, c - radius), min(n - 1, c + radius)
            if hi >= lo:
                cand = lo + int(np.argmax(diff[lo:hi + 1]))
                # Move only on evidence: in flat footage every candidate ties, stay put.
                if diff[cand] > diff[c]:
                    best = cand
        best = max(best, prev.start + 1)
        if best != c:
            moved += 1
            if seg.cut is not None and seg.cut.detected_frame is None:
                seg.cut.detected_frame = c
        prev.end = best
        seg.start = best
        if seg.cut is not None:
            seg.cut.frame = best
    # Monotonic forcing can push the last start onto the end; drop such an empty tail.
    while len(segs) > 1 and segs[-1].start >= n:
        segs[-2].end = n
        del segs[-1]
    segs[-1].end = n
    if segs[-1].length <= 0:
        out.remove_boundary(len(segs) - 1, "snap", "empty after snapping")
    out.record("snap", {"radius": radius}, before, moved=moved)
    out.check()
    return out


def merge_ultra_short(sl: SceneList, max_seconds: float = 0.05) -> SceneList:
    """Merge segments of at most `max_seconds` (at least one frame) into their longer neighbour,
    whatever their kind: at that length they are decoder or detector artefacts."""
    out = sl.copy()
    before = len(out.segments)
    fps = float(out.fps) if out.fps else 0.0
    thr = max(1, int(round(fps * max_seconds))) if fps > 0 else 1
    segs = out.segments
    i = 0
    while i < len(segs) and len(segs) > 1:
        seg = segs[i]
        if seg.length > thr:
            i += 1
            continue
        reason = f"fragment of {seg.length} frames"
        if i == 0:
            out.remove_boundary(1, "ultra_short", reason)
        elif i == len(segs) - 1:
            out.remove_boundary(i, "ultra_short", reason)
            i = max(0, i - 1)
        elif segs[i - 1].length >= segs[i + 1].length:
            out.remove_boundary(i, "ultra_short", reason)
            i = max(0, i - 1)
        else:
            out.remove_boundary(i + 1, "ultra_short", reason)
    out.record("ultra_short", {"maxSeconds": max_seconds, "maxFrames": thr}, before)
    out.check()
    return out
