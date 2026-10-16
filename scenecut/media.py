"""Probing and frame-exact decoding.

Frame *n* is always the n-th frame the decoder outputs for the first video stream, counted from
0. Nothing here seeks by frame number: container seeks land on keyframes and frame-number seeks
on long-GOP codecs can return a neighbouring frame, so every consumer reads sequentially.

Two decode backends produce the same frames:

  ffmpeg   the ffmpeg binary, piping raw RGB. Fastest (multi-threaded decode and scale).
  pyav     PyAV, which ComfyUI already depends on. Used when no ffmpeg binary is found.

Both scale with libswscale's bicubic filter, ffmpeg's default.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    frame_count: int
    duration: float
    has_audio: bool
    codec: str
    pix_fmt: str

    @property
    def fps_float(self) -> float:
        return float(self.fps) if self.fps else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = f"{self.fps.numerator}/{self.fps.denominator}"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VideoInfo":
        d = dict(d)
        d["fps"] = Fraction(d["fps"])
        return cls(**d)


@lru_cache(maxsize=1)
def ffmpeg_binary() -> Optional[str]:
    """The ffmpeg executable: $SCENECUT_FFMPEG, then PATH, then imageio-ffmpeg if installed."""
    env = os.environ.get("SCENECUT_FFMPEG")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe(path: str, count_frames: bool = True) -> VideoInfo:
    """Stream properties via PyAV. `count_frames` demuxes the whole file (no decoding, a few
    seconds for a feature film) to count video packets, which is what the decoder will output for
    every common codec; the container's own frame count is often missing or wrong."""
    import av

    if not os.path.isfile(path):
        raise FileNotFoundError(f"video not found: {path}")
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        vs = container.streams.video[0]
        has_audio = bool(container.streams.audio)
        fps = vs.average_rate or vs.guessed_rate or vs.base_rate
        fps = Fraction(fps) if fps else Fraction(0)
        if vs.duration is not None and vs.time_base is not None:
            duration = float(vs.duration * vs.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        else:
            duration = 0.0
        frames = int(vs.frames or 0)
        codec = vs.codec_context.name
        pix_fmt = vs.codec_context.pix_fmt or ""
        width, height = vs.codec_context.width, vs.codec_context.height
        if count_frames:
            n = 0
            for packet in container.demux(vs):
                if packet.size:
                    n += 1
            frames = n or frames
    if frames <= 0 and fps and duration:
        frames = int(round(duration * float(fps)))
    if not duration and fps and frames:
        duration = frames / float(fps)
    return VideoInfo(path=os.path.abspath(path), width=width, height=height, fps=fps,
                     frame_count=frames, duration=duration, has_audio=has_audio,
                     codec=codec, pix_fmt=pix_fmt)


def iter_frames(path: str, width: int, height: int, chunk: int = 64,
                backend: str = "auto") -> Iterator[np.ndarray]:
    """Every frame of the first video stream, in order, as uint8 RGB chunks (N, height, width, 3)."""
    if backend == "auto":
        backend = "ffmpeg" if ffmpeg_binary() else "pyav"
    if backend == "ffmpeg":
        yield from _iter_ffmpeg(path, width, height, chunk)
    elif backend == "pyav":
        yield from _iter_pyav(path, width, height, chunk)
    else:
        raise ValueError(f"unknown decode backend: {backend}")


def iter_frames_at(path: str, indices, width: int, height: int) -> Iterator[tuple[int, np.ndarray]]:
    """(index, uint8 RGB frame) for each requested frame index, in increasing order.

    Every frame is decoded (that is what keeps the numbering exact) but only the requested ones
    are scaled and converted, and decoding stops after the last one. Uses PyAV, so it needs no
    ffmpeg binary and no command line, however many frames are asked for."""
    import av

    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return
    last = wanted[-1]
    j = 0
    with av.open(path) as container:
        vs = container.streams.video[0]
        vs.thread_type = "AUTO"
        for n, frame in enumerate(container.decode(vs)):
            if n == wanted[j]:
                img = frame.reformat(width=width, height=height, format="rgb24",
                                     interpolation="BICUBIC").to_ndarray()
                yield n, img
                j += 1
                if j == len(wanted):
                    return
            if n >= last:
                return


def read_range(path: str, first: int, last: int, width: int, height: int) -> dict[int, np.ndarray]:
    """{n: uint8 RGB frame} for frames first..last (inclusive), quickly, for interactive use.

    Seeks to a keyframe before `first` and numbers the decoded frames by timestamp, relative to
    the timestamp of frame 0. The numbering is checked as it goes (consecutive, on the frame
    grid); if the stream does not behave (variable frame rate, broken timestamps) it falls back
    to the exact sequential decode from the start, which is always right but slower."""
    import av

    first, last = max(0, int(first)), int(last)
    if last < first:
        return {}
    try:
        with av.open(path) as container:
            vs = container.streams.video[0]
            vs.thread_type = "AUTO"
            # PyAV's own rational type (newer releases) has no int(); plain Fractions do
            rate, tb = _fraction(vs.average_rate or vs.guessed_rate), _fraction(vs.time_base)
            frame0 = next(container.decode(vs), None)
            if first > 0 and rate and tb and frame0 is not None and frame0.pts is not None:
                p0 = frame0.pts
                margin = max(int(rate) * 2, 24)              # land a keyframe or two early
                target = p0 + int(max(0, first - margin) / rate / tb)
                container.seek(target, stream=vs, backward=True, any_frame=False)
                out, prev = {}, None
                for frame in container.decode(vs):
                    if frame.pts is None:
                        break
                    x = float((frame.pts - p0) * tb * rate)
                    n = int(round(x))
                    if abs(x - n) > 0.3 or (prev is not None and n != prev + 1) or (prev is None and n > first):
                        break
                    prev = n
                    if n >= first:
                        out[n] = frame.reformat(width=width, height=height, format="rgb24",
                                                interpolation="BICUBIC").to_ndarray()
                    if n >= last:
                        return out
    except (av.error.FFmpegError, StopIteration):
        pass
    return dict(iter_frames_at(path, range(first, last + 1), width, height))


def _fraction(x) -> Optional[Fraction]:
    return Fraction(x.numerator, x.denominator) if x else None


def prefetch(items: Iterator, depth: int = 4) -> Iterator:
    """Run `items` on a background thread, `depth` items ahead, so decoding overlaps whatever the
    consumer does with each item. Exceptions are re-raised in the consumer; closing the consumer
    stops the producer."""
    import queue

    q: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = threading.Event()
    end = object()

    def produce():
        try:
            for item in items:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
            q.put((end, None))
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            q.put((end, exc))
        finally:
            close = getattr(items, "close", None)
            if close:
                close()

    t = threading.Thread(target=produce, daemon=True, name="scenecut-decode")
    t.start()
    try:
        while True:
            item = q.get()
            if isinstance(item, tuple) and len(item) == 2 and item[0] is end:
                if item[1] is not None:
                    raise item[1]
                return
            yield item
    finally:
        stop.set()
        while t.is_alive():
            try:
                q.get_nowait()
            except Exception:
                t.join(timeout=0.05)


def _iter_ffmpeg(path: str, width: int, height: int, chunk: int) -> Iterator[np.ndarray]:
    exe = ffmpeg_binary()
    if not exe:
        raise RuntimeError("ffmpeg binary not found")
    cmd = [exe, "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-"]
    frame_bytes = width * height * 3
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=frame_bytes * chunk, creationflags=flags)
    # Drain stderr on a thread so a chatty decoder cannot fill the pipe and stall us.
    err: list[bytes] = []
    drain = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
    drain.start()
    try:
        while True:
            buf = _read_exact(proc.stdout, frame_bytes * chunk)
            n = len(buf) // frame_bytes
            if n:
                yield np.frombuffer(buf[: n * frame_bytes], np.uint8).reshape(n, height, width, 3)
            if len(buf) < frame_bytes * chunk:
                break
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        drain.join(timeout=5)
    if proc.returncode not in (0, None, -9):
        msg = (err[0] if err else b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {msg[-2000:]}")


def _read_exact(stream, size: int) -> bytes:
    parts, got = [], 0
    while got < size:
        b = stream.read(size - got)
        if not b:
            break
        parts.append(b)
        got += len(b)
    return b"".join(parts)


def _iter_pyav(path: str, width: int, height: int, chunk: int) -> Iterator[np.ndarray]:
    import av

    with av.open(path) as container:
        vs = container.streams.video[0]
        vs.thread_type = "AUTO"
        graph = av.filter.Graph()
        src = graph.add_buffer(template=vs)
        scale = graph.add("scale", f"{width}:{height}:flags=bicubic")
        fmt = graph.add("format", "rgb24")
        sink = graph.add("buffersink")
        src.link_to(scale)
        scale.link_to(fmt)
        fmt.link_to(sink)
        graph.configure()

        batch = np.empty((chunk, height, width, 3), np.uint8)
        n = 0

        def drain():
            nonlocal n
            out = []
            while True:
                try:
                    f = sink.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    return out
                batch[n] = f.to_ndarray()
                n += 1
                if n == chunk:
                    out.append(batch.copy())
                    n = 0

        for frame in container.decode(vs):
            graph.push(frame)
            yield from drain()
        graph.push(None)
        yield from drain()
        if n:
            yield batch[:n].copy()
