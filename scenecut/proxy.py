"""Preview proxies: a small H.264 copy of a video that any browser can play and seek quickly.

Every source frame is kept (no frame-rate conversion), so frame n of the proxy is frame n of the
source and the viewer can map time to frame exactly. Keyframes every 12 frames and no B-frames
make seeking cheap. Audio, if any, is kept as AAC.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Callable, Optional

from .media import ffmpeg_binary, probe

KEYINT = 12


def cache_key(path: str, *parts) -> str:
    st = os.stat(path)
    raw = "|".join(str(x) for x in (os.path.realpath(path), st.st_size, int(st.st_mtime)) + parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def proxy_size(width: int, height: int, target_height: int) -> tuple[int, int]:
    h = min(even(target_height), even(height))
    return even(width * h / max(1, height)), h


def make_proxy(src: str, dst: str, height: int = 360, frame_count: Optional[int] = None,
               progress: Optional[Callable[[int, int], None]] = None,
               check_abort: Optional[Callable[[], None]] = None) -> str:
    """Write the proxy of `src` to `dst` (reused if it already exists)."""
    if os.path.isfile(dst):
        return dst
    info = probe(src, count_frames=frame_count is None)
    total = frame_count or info.frame_count
    w, h = proxy_size(info.width, info.height, height)
    tmp = dst + ".part.mp4"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if ffmpeg_binary():
        _ffmpeg_proxy(src, tmp, w, h, info.has_audio, total, progress, check_abort)
    else:
        _pyav_proxy(src, tmp, w, h, total, progress, check_abort)
    os.replace(tmp, dst)
    return dst


def _ffmpeg_proxy(src, tmp, w, h, has_audio, total, progress, check_abort):
    cmd = [ffmpeg_binary(), "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "96k", "-ac", "2"]
    cmd += ["-vf", f"scale={w}:{h}:flags=bicubic,format=yuv420p", "-fps_mode", "passthrough",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-g", str(KEYINT),
            "-keyint_min", str(KEYINT), "-sc_threshold", "0", "-bf", "0",
            "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", tmp]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            creationflags=flags)
    try:
        for line in proc.stdout:
            if line.startswith("frame=") and progress:
                try:
                    progress(int(line.split("=", 1)[1]), total)
                except ValueError:
                    pass
            if check_abort:
                check_abort()
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg could not make the preview proxy: {err.strip()[-1500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
            if os.path.exists(tmp):
                os.remove(tmp)


def _pyav_proxy(src, tmp, w, h, total, progress, check_abort):
    """Video-only fallback when there is no ffmpeg binary."""
    import av

    with av.open(src) as inp, av.open(tmp, "w", format="mp4", options={"movflags": "+faststart"}) as out:
        vin = inp.streams.video[0]
        vin.thread_type = "AUTO"
        rate = vin.average_rate or vin.guessed_rate or 30
        codec = "libx264" if "libx264" in av.codecs_available else "h264"
        vout = out.add_stream(codec, rate=rate)
        vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
        vout.options = {"crf": "27", "preset": "veryfast", "g": str(KEYINT), "bf": "0",
                        "keyint_min": str(KEYINT), "sc_threshold": "0"}
        for n, frame in enumerate(inp.decode(vin)):
            img = frame.reformat(width=w, height=h, format="yuv420p", interpolation="BICUBIC")
            img.pts = None
            for packet in vout.encode(img):
                out.mux(packet)
            if progress and n % 50 == 0:
                progress(n, total)
            if check_abort and n % 50 == 0:
                check_abort()
        for packet in vout.encode():
            out.mux(packet)
