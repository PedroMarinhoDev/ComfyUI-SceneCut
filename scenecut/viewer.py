"""What the timeline viewer shows: scene data, thumbnail sprites, and a standalone HTML report.

The same viewer code (web/viewer_core.js) runs inside the ComfyUI node and in the report; both
read the JSON built here. Thumbnails are packed into JPEG sprite sheets, three per scene by
default (start, middle, end, a couple of frames clear of the cuts).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from PIL import Image

from .preview import grab_frames, pick_frames, thumb_size
from .scenes import SceneList

DATA_VERSION = 1
SPRITE_MAX = 4096          # sprite sheet edge, px


def _round(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


def make_sprites(sl: SceneList, video: str, out_dir: str, stem: str, per_scene: int = 3,
                 tile_width: int = 160, progress: Optional[Callable[[int, int], None]] = None,
                 check_abort: Optional[Callable[[], None]] = None) -> dict:
    """Decode the representative frames of every segment from `video` (the proxy or the source:
    same frame numbering) and pack them into sprite sheets. Returns the sprite description, with
    `scene_tiles[i]` the tile numbers of segment i."""
    tw, th = thumb_size(sl.source.width, sl.source.height, tile_width)
    frames_per_seg = [pick_frames(s, per_scene) for s in sl.segments]
    wanted = sorted({f for fs in frames_per_seg for f in fs})
    last = wanted[-1] if wanted else 0
    thumbs = grab_frames(video, wanted, (tw, th),
                         progress=(lambda d: progress(d, last)) if progress else None,
                         check_abort=check_abort)
    cols = max(1, SPRITE_MAX // tw)
    rows = max(1, SPRITE_MAX // th)
    per_page = cols * rows
    tile_of: dict[int, int] = {f: k for k, f in enumerate(wanted)}
    pages, sizes = [], []
    for p0 in range(0, max(1, len(wanted)), per_page):
        chunk = wanted[p0:p0 + per_page]
        n_rows = max(1, -(-len(chunk) // cols))
        sheet = Image.new("RGB", (cols * tw if len(chunk) >= cols else max(1, len(chunk)) * tw, n_rows * th),
                          (30, 30, 34))
        for k, f in enumerate(chunk):
            t = thumbs.get(f)
            if t is not None:
                sheet.paste(Image.fromarray(t), ((k % cols) * tw, (k // cols) * th))
        name = f"{stem}_thumbs_{len(pages)}.jpg"
        sheet.save(os.path.join(out_dir, name), quality=82, optimize=True)
        pages.append(name)
        sizes.append(list(sheet.size))
    return {"tileW": tw, "tileH": th, "cols": cols, "perPage": per_page, "pages": pages, "pageSizes": sizes,
            "frames": wanted, "sceneTiles": [[tile_of[f] for f in fs] for fs in frames_per_seg]}


def build_data(sl: SceneList, sprites: Optional[dict] = None, proxy: Optional[dict] = None) -> dict:
    """Everything the viewer draws, in one JSON-friendly dict."""
    src = sl.source
    fps = float(src.fps) if src.fps else 0.0
    segs = []
    for i, s in enumerate(sl.segments):
        cut = s.cut
        rec = {"start": s.start, "end": s.end, "kind": s.kind}
        if s.transition:
            rec["transition"] = s.transition
        if i and cut is not None:
            rec["relation"] = "manual" if cut.source == "manual" else (cut.relation or "hard_cut")
            if cut.confidence is not None:
                rec["confidence"] = round(cut.confidence, 4)
            if cut.detected_frame is not None and cut.detected_frame != s.start:
                rec["detectedFrame"] = cut.detected_frame
            if cut.validation:
                rec["validation"] = _round(cut.validation)
            if cut.review:
                rec["edited"] = dict(cut.review)
        if s.absorbed:
            rec["absorbed"] = s.absorbed
        if sprites:
            rec["tiles"] = sprites["sceneTiles"][i]
        segs.append(rec)
    rejected = []
    for r in sl.rejected:
        rec = {"frame": r["frame"], "stage": r.get("stage", ""), "reason": r.get("reason", "")}
        c = r.get("cut") or {}
        if c.get("confidence") is not None:
            rec["confidence"] = round(c["confidence"], 4)
        if c.get("relation"):
            rec["relation"] = c["relation"]
        if r.get("validation"):
            rec["validation"] = _round(r["validation"])
        rejected.append(rec)
    data = {
        "version": DATA_VERSION,
        "source": {"name": os.path.basename(src.path), "path": src.path, "fps": fps,
                   "fpsFraction": f"{src.fps.numerator}/{src.fps.denominator}" if src.fps else None,
                   "frames": sl.frame_count, "duration": sl.seconds(sl.frame_count),
                   "width": src.width, "height": src.height, "hasAudio": src.has_audio},
        "segments": segs,
        "rejected": sorted(rejected, key=lambda r: r["frame"]),
        "history": _round(sl.history),
        "summary": sl.summary(),
        "selection": sl.selection,
    }
    if sprites:
        data["sprites"] = {k: sprites[k] for k in ("tileW", "tileH", "cols", "perPage", "pages", "pageSizes")}
    if proxy:
        data["proxy"] = proxy
    return data


def write_data(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


def report_html(data: dict, viewer_js: str, title: str) -> str:
    """A self-contained page: the viewer code and the data inlined; proxy and sprites referenced
    by relative path, so the folder can be zipped and opened anywhere."""
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title_html = title.replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_html} - SceneCut</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #141417; color: #e8e8ec; font: 13px system-ui, sans-serif; }}
  #app {{ position: fixed; inset: 0; }}
</style></head>
<body><div id="app"></div>
<script type="module">
{viewer_js}
</script>
<script type="module">
const data = {payload};
const viewer = new window.SceneCutViewer(document.getElementById("app"), {{ resolve: (name) => name, standalone: true }});
viewer.load(data);
</script>
</body></html>
"""
