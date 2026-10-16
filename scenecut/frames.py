"""Frames in and out of scene lists, for ComfyUI's IMAGE batches.

Out: the frames of a scene list's selected scenes (all of them, or one per scene), decoded
exactly. In: an image batch becomes one scene of a lossless video file, so everything that works
on a video (proxy, thumbnails, viewer, frames out) works on it unchanged.
"""
from __future__ import annotations

import hashlib
import os
from fractions import Fraction
from typing import Callable, Optional

import numpy as np

from .media import probe
from .preview import grab_frames
from .scenes import SceneList, Segment

IMAGE_MODES = {
    "off": None,
    "all frames": "all",
    "first frame per scene": "first",
    "middle frame per scene": "middle",
}
IMAGE_MODE_HELP = (
    "Also send frames to the images output, from the selected scenes (all scenes when none are "
    "selected). all frames: every frame, in order. first / middle frame per scene: one image per "
    "scene. off: the images output carries a single black frame. Frames are full resolution "
    "float32 (about 25 MB per 1080p frame), so prefer one per scene or select scenes first.")


def frames_for(sl: SceneList, mode: str) -> list[int]:
    """Frame numbers sent to the images output for a mode of IMAGE_MODES (a value)."""
    out: list[int] = []
    for i in sl.selected_indices():
        s = sl.segments[i]
        if mode == "all":
            out.extend(range(s.start, s.end))
        elif mode == "first":
            out.append(s.start)
        elif mode == "middle":
            out.append(s.start + s.length // 2)
    return out


def estimate_bytes(n: int, width: int, height: int) -> int:
    return n * width * height * 3 * 4


def check_memory(n: int, width: int, height: int, available: Optional[int]) -> None:
    need = estimate_bytes(n, width, height)
    if available is not None and need > 0.8 * available:
        raise MemoryError(
            f"{n} frames at {width}x{height} need {need / 1024 ** 3:.1f} GB as images, more than the "
            f"{available / 1024 ** 3:.1f} GB of free memory. Select fewer scenes in the SceneCut "
            "Viewer, or output one frame per scene.")


def read_frames(path: str, frames: list[int], width: int, height: int,
                progress: Optional[Callable[[int], None]] = None,
                check_abort: Optional[Callable[[], None]] = None) -> np.ndarray:
    """uint8 (N, height, width, 3), in the order of `frames` (repeats allowed)."""
    got = grab_frames(path, frames, (width, height), progress=progress, check_abort=check_abort)
    missing = [f for f in frames if f not in got]
    if missing:
        raise RuntimeError(f"could not decode frames {missing[:5]} of {path}")
    return np.stack([got[f] for f in frames]) if frames else np.zeros((0, height, width, 3), np.uint8)


# --------------------------------------------------------------------------------------------- #
# Image batch -> lossless video -> one-scene list
# --------------------------------------------------------------------------------------------- #
def images_key(images: np.ndarray, fps: float) -> str:
    """Content key of an image batch (shape, fps and a strided sample of the pixels)."""
    h = hashlib.sha1(repr((images.shape, str(images.dtype), round(float(fps), 6))).encode())
    flat = images.reshape(-1)
    step = max(1, flat.size // 2_000_000)
    h.update(np.ascontiguousarray(flat[::step]).tobytes())
    h.update(np.ascontiguousarray(images[-1]).tobytes())
    return h.hexdigest()[:16]


def images_to_video(images: np.ndarray, fps: float, path: str) -> str:
    """Write uint8 RGB frames (N,H,W,3) to a lossless H.264 RGB file (bit-exact on decode)."""
    import av

    if os.path.isfile(path):
        return path
    n, h, w, _ = images.shape
    rate = Fraction(fps).limit_denominator(1001)
    tmp = path + ".part.mkv"
    with av.open(tmp, "w", format="matroska") as out:
        s = out.add_stream("libx264rgb", rate=rate)
        s.width, s.height, s.pix_fmt = w, h, "rgb24"
        s.options = {"qp": "0", "preset": "veryfast", "g": "12"}
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(images[i]), format="rgb24")
            for packet in s.encode(frame):
                out.mux(packet)
        for packet in s.encode():
            out.mux(packet)
    os.replace(tmp, path)
    return path


def single_scene(path: str) -> SceneList:
    """A scene list with one scene covering the whole video."""
    info = probe(path)
    sl = SceneList(source=info, frame_count=info.frame_count, segments=[Segment(0, info.frame_count)])
    sl.history.append({"stage": "images", "params": {"frames": info.frame_count}, "before": 1, "after": 1})
    sl.check()
    return sl
