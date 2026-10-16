"""What the user did in the viewer: cuts moved by a frame or two, and the scenes they selected.

A review is a small JSON object kept in the viewer node (so it is saved with the workflow):

    {"base": "<fingerprint of the scene list it was made on>",
     "edits": [{"cut": 900, "to": 901, "mode": "auto"}, ...],     # applied in order
     "selection": [3, 7, 25]}                                     # segment indices

Edits only move boundaries, so segment indices (and the selection) stay valid. A review made on a
different scene list (the video or the detection settings changed upstream) is not applied.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .boundary import best_split
from .media import read_range
from .scenes import SceneList

LAYOUT = 32                  # side of the colour-layout thumbnail used to compare frames
MAX_REACH_SECONDS = 1.0      # how far from the playhead a cut may be for "fix cut"


def fingerprint(sl: SceneList) -> str:
    raw = json.dumps([sl.source.path, sl.frame_count, [(s.start, s.end, s.kind) for s in sl.segments]])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def parse(review) -> dict:
    if isinstance(review, dict):
        return review
    try:
        out = json.loads(review or "{}")
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def apply_edits(sl: SceneList, edits: list[dict], out: Optional[SceneList] = None) -> tuple[SceneList, int]:
    """Apply cut moves in order to a copy of `sl`; edits that no longer fit are skipped."""
    out = out or sl.copy()
    starts = {s.start: i for i, s in enumerate(out.segments)}
    applied = 0
    for e in edits or []:
        try:
            cut, to = int(e["cut"]), int(e["to"])
        except (KeyError, TypeError, ValueError):
            continue
        i = starts.get(cut)
        if i is None or i == 0 or cut == to:
            continue
        try:
            out.move_boundary(i, to, mode=str(e.get("mode", "manual")))
        except ValueError:
            continue
        del starts[cut]
        starts[to] = i
        applied += 1
    return out, applied


def apply_review(sl: SceneList, review) -> tuple[SceneList, dict]:
    """The scene list as reviewed, and what happened: {state: empty|applied|stale, ...}."""
    r = parse(review)
    edits, selection = r.get("edits") or [], r.get("selection")
    if not edits and not selection:
        return sl, {"state": "empty"}
    if r.get("base") != fingerprint(sl):
        return sl, {"state": "stale", "edits": len(edits), "selected": len(selection or [])}
    out, applied = apply_edits(sl, edits)
    n = len(out.segments)
    sel = sorted({int(i) for i in (selection or []) if isinstance(i, (int, float)) and 0 <= int(i) < n})
    out.selection = sel or None
    out.record("review", {"edits": len(edits), "selected": len(sel)}, n, moved=applied)
    return out, {"state": "applied", "edits": applied, "selected": len(sel)}


# --------------------------------------------------------------------------------------------- #
# "Fix cut": put the cut nearest the playhead where the picture really changes
# --------------------------------------------------------------------------------------------- #
def _layout(frames: np.ndarray) -> torch.Tensor:
    """Brightness- and contrast-normalised colour layout, L2-normalised: a camera flash or a fade
    in level barely moves it, a different picture does."""
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(LAYOUT, LAYOUT), mode="area")
    x = x - x.mean(dim=(1, 2, 3), keepdim=True)
    x = x / x.flatten(1).std(dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
    return F.normalize(x.flatten(1), dim=1)


def nearest_cut(sl: SceneList, frame: int) -> Optional[int]:
    """Index of the segment whose start is the cut nearest `frame`."""
    if len(sl.segments) < 2:
        return None
    return min(range(1, len(sl.segments)), key=lambda i: (abs(sl.segments[i].start - frame), i))


def manual_fix(sl: SceneList, frame: int) -> dict:
    """The playhead frame (and the frames between it and the nearest cut of its scene) joins the
    scene on the other side of that cut."""
    k = max(i for i, s in enumerate(sl.segments) if s.start <= frame)
    seg = sl.segments[k]
    to_start, to_end = frame - seg.start, seg.end - 1 - frame
    if k > 0 and (to_start <= to_end or k == len(sl.segments) - 1):
        index, to = k, frame + 1                       # frames start..frame join the scene before
    elif k + 1 < len(sl.segments):
        index, to = k + 1, frame                       # frames frame..end-1 join the scene after
    else:
        return {"error": "There is no cut to move in a video with a single scene."}
    a, b = sl.segments[index - 1], sl.segments[index]
    if not a.start < to < b.end:
        return {"error": "That would leave a scene with no frames."}
    return {"index": index, "cut": b.start, "to": to, "mode": "manual"}


def auto_fix(sl: SceneList, frame: int, video: str, embed: Optional[Callable] = None,
             reach: Optional[int] = None) -> dict:
    """Move the cut nearest `frame` to where the frames stop looking like the scene before and
    start looking like the scene after. `video` is the source or its proxy (same numbering);
    `embed` optionally maps uint8 frames (N,H,W,3) to L2-normalised vectors (DINOv3), used
    alongside the colour layout."""
    fps = float(sl.fps) if sl.fps else 25.0
    index = nearest_cut(sl, frame)
    if index is None:
        return {"error": "There are no cuts in this scene list."}
    cut = sl.segments[index].start
    limit = max(6, int(round(MAX_REACH_SECONDS * fps)))
    if abs(cut - frame) > limit:
        return {"error": f"No cut within {MAX_REACH_SECONDS:g} s of the playhead: put it on a frame "
                         "next to the cut you want to fix."}
    a, b = sl.segments[index - 1], sl.segments[index]
    r = reach or max(6, abs(frame - cut) + 4)
    lo, hi = max(a.start + 1, min(cut, frame) - r), min(b.end - 1, max(cut, frame) + r)
    anchors = 4
    pre = list(range(max(a.start, lo - 1 - anchors), lo))                 # in the scene before
    post = list(range(hi + 1, min(b.end, hi + 2 + anchors)))              # in the scene after
    if not pre or not post:
        return {"error": "The scenes around this cut are too short to compare."}
    span = list(range(lo, hi + 1))
    first, last = pre[0], post[-1]
    frames = read_range(video, first, last, 224, 224)
    if any(i not in frames for i in range(first, last + 1)):
        return {"error": "Could not read the frames around the cut."}
    stack = np.stack([frames[i] for i in range(first, last + 1)])
    feats = [_layout(stack)]
    if embed is not None:
        feats.append(embed(stack))
    to_before = np.zeros(len(span))
    to_after = np.zeros(len(span))
    for f in feats:
        f = f.float()
        ref_a = F.normalize(f[[i - first for i in pre]].mean(0), dim=0)
        ref_b = F.normalize(f[[i - first for i in post]].mean(0), dim=0)
        x = f[[i - first for i in span]]
        to_before += (x @ ref_a).numpy() / len(feats)
        to_after += (x @ ref_b).numpy() / len(feats)
    new, costs = best_split(lo, to_before, to_after, prefer=cut, margin=0.02)
    new = max(a.start + 1, min(b.end - 1, new))
    return {"index": index, "cut": cut, "to": int(new), "mode": "auto",
            "frames": span, "toBefore": [round(float(v), 4) for v in to_before],
            "toAfter": [round(float(v), 4) for v in to_after], "method": "dinov3+layout" if embed else "layout"}
