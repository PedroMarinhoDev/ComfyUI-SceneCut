"""Thumbnails and contact sheets.

Thumbnails come from one sequential decode at thumbnail size, so frame numbers are exact. A
contact sheet is a grid of pages; each tile shows a scene's frame(s) and a caption with its
index, start time, duration and what kind of boundary begins it.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .media import iter_frames, prefetch
from .scenes import SceneList, Segment, timecode

# Boundary colours, shared with the viewer.
COLORS = {
    "hard_cut": (66, 133, 244),
    "sudden_jump": (171, 71, 188),
    "transition": (251, 140, 0),
    "new_start": (120, 120, 120),
    "transition_source": (251, 140, 0),
    "manual": (0, 172, 193),
    "start": (90, 90, 90),
}
TAGS = {"hard_cut": "cut", "sudden_jump": "jump cut", "transition": "after transition",
        "manual": "manual cut", "start": "start"}
BG = (24, 24, 27)
FG = (235, 235, 240)
DIM = (150, 150, 160)


def thumb_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    w = max(16, int(target_width) // 2 * 2)
    h = max(16, int(round(w * height / max(1, width))) // 2 * 2)
    return w, h


def pick_frames(seg: Segment, per_scene: int) -> list[int]:
    """Frames that represent a segment: its middle, or start/middle/end kept clear of the cut."""
    if per_scene <= 1 or seg.length < 3:
        return [seg.start + seg.length // 2]
    margin = min(2, seg.length // 4)
    first, last = seg.start + margin, seg.end - 1 - margin
    if per_scene == 2:
        return [first, last]
    return [int(round(x)) for x in np.linspace(first, last, per_scene)]


def grab_frames(path: str, frames: Iterable[int], size: tuple[int, int],
                progress: Optional[Callable[[int], None]] = None,
                check_abort: Optional[Callable[[], None]] = None) -> dict[int, np.ndarray]:
    """{frame: uint8 RGB (h, w, 3)} for the requested frames, decoded sequentially."""
    wanted = sorted(set(int(f) for f in frames))
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    w, h = size
    last = wanted[-1]
    idx, j = 0, 0
    chunks = prefetch(iter_frames(path, w, h))
    try:
        for chunk in chunks:
            n = len(chunk)
            while j < len(wanted) and wanted[j] < idx + n:
                out[wanted[j]] = chunk[wanted[j] - idx].copy()
                j += 1
            idx += n
            if progress:
                progress(idx)
            if check_abort:
                check_abort()
            if idx > last:
                break
    finally:
        chunks.close()
    return out


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def boundary_kind(sl: SceneList, i: int) -> str:
    seg = sl.segments[i]
    if i == 0 or seg.cut is None:
        return "start"
    if seg.cut.source == "manual":
        return "manual"
    return seg.cut.relation or "hard_cut"


def contact_sheet(sl: SceneList, thumbs: dict[int, np.ndarray], indices: list[int],
                  per_scene: int, tile_w: int, tile_h: int, columns: int,
                  rows: int) -> list[Image.Image]:
    """Pages of tiles, `columns` x `rows` scenes per page, all pages the same size."""
    pad, cap_h = 6, max(34, tile_h // 4)
    strip_w = tile_w * per_scene + 2 * (per_scene - 1)
    cell_w, cell_h = strip_w + 2 * pad, tile_h + cap_h + 2 * pad
    per_page = max(1, columns * rows)
    pages = []
    big, small = _font(max(11, cap_h // 2 - 2)), _font(max(10, cap_h // 2 - 4))
    for p0 in range(0, max(1, len(indices)), per_page):
        page_idx = indices[p0:p0 + per_page]
        img = Image.new("RGB", (columns * cell_w, rows * cell_h), BG)
        draw = ImageDraw.Draw(img)
        for k, i in enumerate(page_idx):
            seg = sl.segments[i]
            x0, y0 = (k % columns) * cell_w + pad, (k // columns) * cell_h + pad
            for m, f in enumerate(pick_frames(seg, per_scene)):
                t = thumbs.get(f)
                x = x0 + m * (tile_w + 2)
                if t is not None:
                    img.paste(Image.fromarray(t), (x, y0))
                else:
                    draw.rectangle([x, y0, x + tile_w - 1, y0 + tile_h - 1], fill=(50, 50, 55))
            kind = boundary_kind(sl, i)
            color = COLORS["transition"] if seg.kind == "transition" else COLORS.get(kind, COLORS["hard_cut"])
            draw.rectangle([x0, y0, x0 + strip_w - 1, y0 + 3], fill=color)
            if seg.kind == "transition":
                draw.rectangle([x0, y0, x0 + strip_w - 1, y0 + tile_h - 1], outline=color, width=3)
            secs = sl.seconds(seg.length)
            label = f"#{i + 1}"
            tag = seg.transition if seg.kind == "transition" else TAGS.get(kind, kind.replace("_", " "))
            if seg.absorbed:
                tag += " +" + ",".join(sorted({a.get("transition") or "?" for a in seg.absorbed}))
            ty = y0 + tile_h + 3
            draw.text((x0, ty), label, fill=FG, font=big)
            lw = draw.textlength(label + "  ", font=big)
            draw.text((x0 + lw, ty), tag, fill=color, font=big)
            draw.text((x0, ty + cap_h // 2), f"{timecode(sl.seconds(seg.start))}  {secs:.2f}s  "
                                             f"f{seg.start}-{seg.end - 1}", fill=DIM, font=small)
        pages.append(img)
    return pages
