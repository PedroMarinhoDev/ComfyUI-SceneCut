"""Cut a video into one file per scene, frame-exact, with its audio.

Frame *n* is the n-th frame the decoder outputs, as everywhere in SceneCut. To get there without
decoding the whole video for every clip, frames are identified by timestamp: the sorted timestamps
of the video packets are exactly the presentation times of frames 0, 1, 2, ... (one packet per
frame, which holds for every common codec, constant or variable frame rate). Clips close together
are written in one sequential pass; a clip far ahead is reached with one keyframe seek. If a
decoded timestamp is not in the list (a broken or unusual file), the export starts over and counts
frames from the beginning instead, which is always right, only slower.

Audio is decoded and sliced at the exact times of the clip's first frame and of the frame after its
last, so consecutive clips put back together give the original sound.

Everything runs on PyAV (part of ComfyUI), so no ffmpeg binary is needed.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Optional

import numpy as np

from .media import probe
from .scenes import SceneList


@dataclass(frozen=True)
class ClipCodec:
    id: str
    label: str
    encoder: str
    ext: str
    pix_fmt: Optional[str]           # None: keep the source's when the encoder takes it
    options: dict
    audio: str
    audio_options: dict = field(default_factory=dict)
    quality: str = "crf"             # crf | cq | none
    gpu: bool = False
    tag: Optional[str] = None


CODECS: dict[str, ClipCodec] = {c.id: c for c in (
    ClipCodec("h264", "H.264 (mp4)", "libx264", ".mp4", "yuv420p", {"preset": "fast"}, "aac",
              {"b": "192k"}),
    ClipCodec("h264_nvenc", "H.264, NVIDIA GPU (mp4)", "h264_nvenc", ".mp4", "yuv420p",
              {"preset": "p5", "tune": "hq", "rc": "vbr", "b": "0"}, "aac", {"b": "192k"},
              quality="cq", gpu=True),
    ClipCodec("h265", "H.265 / HEVC (mp4)", "libx265", ".mp4", "yuv420p",
              {"preset": "fast", "x265-params": "log-level=error"}, "aac", {"b": "192k"}, tag="hvc1"),
    ClipCodec("h265_10bit", "H.265 10-bit (mp4)", "libx265", ".mp4", "yuv420p10le",
              {"preset": "fast", "x265-params": "log-level=error"}, "aac", {"b": "192k"}, tag="hvc1"),
    ClipCodec("hevc_nvenc", "H.265, NVIDIA GPU (mp4)", "hevc_nvenc", ".mp4", "yuv420p",
              {"preset": "p5", "tune": "hq", "rc": "vbr", "b": "0"}, "aac", {"b": "192k"},
              quality="cq", gpu=True, tag="hvc1"),
    ClipCodec("prores", "ProRes 422 HQ (mov)", "prores_ks", ".mov", "yuv422p10le",
              {"profile": "3", "vendor": "apl0"}, "pcm_s16le", quality="none"),
    ClipCodec("ffv1", "FFV1 lossless (mkv)", "ffv1", ".mkv", None, {"level": "3", "slicecrc": "1"},
              "flac", quality="none"),
)}
DEFAULT_CODEC = "h264"
JSON_ONLY = "none"

# Constant-quality values per setting: x264/x265 CRF, NVENC CQ.
QUALITY = {"high": {"crf": 16, "cq": 19}, "medium": {"crf": 20, "cq": 23}, "small": {"crf": 26, "cq": 29}}
DEFAULT_QUALITY = "high"

FFV1_FORMATS = {"yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "gbrp", "gray"}
SEEK_GAP_SECONDS = 5.0       # clips further apart than this are reached by seeking
SEEK_MARGIN_SECONDS = 2.0    # seek this far before a clip (audio interleaving, keyframes)


class ExportCancelled(Exception):
    pass


class _Renumber(Exception):
    """A decoded timestamp is not in the frame index: number frames by counting instead."""


# --------------------------------------------------------------------------------------------- #
# Codecs
# --------------------------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def codec_works(codec_id: str) -> bool:
    """Whether this machine can encode with it (PyAV has it, and a GPU encoder finds its GPU)."""
    import av

    c = CODECS[codec_id]
    if c.encoder not in av.codecs_available:
        return False
    try:
        cc = av.CodecContext.create(c.encoder, "w")
        cc.width, cc.height = 256, 144
        cc.time_base, cc.framerate = Fraction(1, 24), Fraction(24)
        cc.pix_fmt = c.pix_fmt or "yuv420p"
        cc.options = dict(c.options)
        cc.open()
        return True
    except Exception:  # noqa: BLE001 - no GPU, driver too old, encoder missing
        return False


def available_codecs() -> list[dict]:
    return [{"id": c.id, "label": c.label, "gpu": c.gpu, "quality": c.quality != "none", "ext": c.ext}
            for c in CODECS.values() if codec_works(c.id)]


# --------------------------------------------------------------------------------------------- #
# Frame index
# --------------------------------------------------------------------------------------------- #
@dataclass
class FrameIndex:
    pts: np.ndarray                  # sorted presentation timestamps of frames 0..n-1
    time_base: Fraction
    rate: Fraction                   # nominal frame rate (duration of the last frame)

    def time(self, n: int) -> float:
        """Seconds of frame n's presentation time; n == len gives the end of the last frame."""
        if n < len(self.pts):
            return float(self.pts[n] * self.time_base)
        return float(self.pts[-1] * self.time_base) + (float(1 / self.rate) if self.rate else 0.0)

    def find(self, pts: Optional[int]) -> Optional[int]:
        if pts is None:
            return None
        i = int(np.searchsorted(self.pts, pts))
        return i if i < len(self.pts) and int(self.pts[i]) == int(pts) else None


_INDEX_CACHE: dict = {}


def frame_index(path: str) -> Optional[FrameIndex]:
    """Timestamps of every frame, from the packets (no decoding). None when packets lack them."""
    import av

    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, int(st.st_mtime))
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    with av.open(path) as container:
        vs = container.streams.video[0]
        stamps = []
        for packet in container.demux(vs):
            if not packet.size:
                continue
            ts = packet.pts if packet.pts is not None else packet.dts
            if ts is None:
                stamps = None
                break
            stamps.append(int(ts))
        rate = vs.average_rate or vs.guessed_rate or vs.base_rate
        tb = vs.time_base
    out = None
    if stamps:
        arr = np.unique(np.asarray(stamps, np.int64))
        if len(arr) == len(stamps):                   # duplicates: timestamps don't identify frames
            out = FrameIndex(arr, Fraction(tb), Fraction(rate) if rate else Fraction(0))
    while len(_INDEX_CACHE) >= 8:
        _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
    _INDEX_CACHE[key] = out
    return out


# --------------------------------------------------------------------------------------------- #
# Writing one clip
# --------------------------------------------------------------------------------------------- #
@dataclass
class Clip:
    scene: int                       # segment index in the scene list
    start: int
    end: int                         # exclusive
    file: str
    frames_written: int = 0


class _VideoPart:
    """A clip's video, encoded to a temporary file while its frames stream past. Only one is open at
    a time: GPU encoders allow few sessions at once, and each CPU encoder holds frames in memory."""

    def __init__(self, path: str, codec: ClipCodec, quality: str, width: int, height: int,
                 src_format: str, time_base: Fraction, rate: Fraction):
        import av

        self.path = path
        self.out = av.open(path, "w")
        v = self.out.add_stream(codec.encoder, rate=rate) if rate else self.out.add_stream(codec.encoder)
        self.width, self.height = width - width % 2, height - height % 2
        v.width, v.height = self.width, self.height
        self.pix_fmt = codec.pix_fmt or (src_format if src_format in FFV1_FORMATS else "yuv444p")
        v.pix_fmt = self.pix_fmt
        v.codec_context.time_base = time_base
        v.time_base = time_base
        opts = dict(codec.options)
        if codec.quality != "none":
            opts[codec.quality] = str(QUALITY.get(quality, QUALITY[DEFAULT_QUALITY])[codec.quality])
        v.options = opts
        if codec.tag:
            try:
                v.codec_context.codec_tag = codec.tag
            except Exception:  # noqa: BLE001 - cosmetic (QuickTime compatibility)
                pass
        self.v = v
        self.codec = codec
        self.started = False

    def add(self, frame, pts: int) -> None:
        import av

        if frame.format.name != self.pix_fmt or frame.width != self.width or frame.height != self.height:
            frame = frame.reformat(width=self.width, height=self.height, format=self.pix_fmt)
        frame.pts = pts
        frame.time_base = self.v.codec_context.time_base
        try:
            packets = self.v.encode(frame)
        except av.error.FFmpegError as e:
            if self.started or not self.codec.gpu:
                raise
            raise RuntimeError(f"The NVIDIA encoder could not start ({e}). The GPU may be out of memory or "
                               "running too many encoders: export with H.264 (CPU) instead.") from e
        self.started = True
        for packet in packets:
            self.out.mux(packet)

    def close(self) -> None:
        for packet in self.v.encode():
            self.out.mux(packet)
        self.out.close()

    def abort(self) -> None:
        try:
            self.out.close()
        except Exception:  # noqa: BLE001
            pass
        if os.path.exists(self.path):
            os.remove(self.path)


def _finish_clip(video_path: str, final: str, codec: ClipCodec, audio_src, audio_frames: list) -> None:
    """The clip file: the encoded video copied as it is, and the clip's sound encoded beside it."""
    import av

    if audio_src is None or not audio_frames:
        os.replace(video_path, final)
        return
    tmp = final + ".part" + codec.ext
    try:
        with av.open(video_path) as src, av.open(tmp, "w") as out:
            vin = src.streams.video[0]
            vout = out.add_stream_from_template(vin)
            a = out.add_stream(codec.audio, rate=audio_src.sample_rate, layout=audio_src.layout)
            if codec.audio_options:
                a.options = dict(codec.audio_options)
            a.time_base = Fraction(1, audio_src.sample_rate)
            apackets = [p for f in audio_frames for p in a.encode(f)] + list(a.encode())
            vpackets = []
            for packet in src.demux(vin):
                if packet.dts is None and packet.pts is None:
                    continue                          # the demuxer's flush packet
                packet.stream = vout
                vpackets.append(packet)

            def t(p):
                ts = p.dts if p.dts is not None else p.pts
                return float(ts * p.time_base) if ts is not None and p.time_base else 0.0

            for packet in sorted(vpackets + apackets, key=t):    # interleaved by time
                out.mux(packet)
        os.replace(tmp, final)
    finally:
        for p in (tmp, video_path):
            if os.path.exists(p):
                os.remove(p)


def _slice_audio(frame, a: int, b: int, pts: int):
    """Samples [a, b) of an audio frame as a new frame with timestamp `pts` (in 1/sample_rate)."""
    import av

    arr = frame.to_ndarray()
    if frame.format.is_planar:
        part = arr[:, a:b]
    else:
        ch = len(frame.layout.channels)
        part = arr[:, a * ch:b * ch]
    out = av.AudioFrame.from_ndarray(np.ascontiguousarray(part), format=frame.format.name,
                                     layout=frame.layout.name)
    out.sample_rate = frame.sample_rate
    out.pts = pts
    out.time_base = Fraction(1, frame.sample_rate)
    return out


# --------------------------------------------------------------------------------------------- #
# The export
# --------------------------------------------------------------------------------------------- #
def clip_name(stem: str, scene: int, total: int, ext: str) -> str:
    width = max(3, len(str(total)))
    return f"{stem}_{scene + 1:0{width}d}{ext}"


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^\w.\- ]+", "_", stem, flags=re.UNICODE).strip(" .")[:80] or "video"


def export_clips(path: str, ranges: list[tuple[int, int, int]], out_dir: str, codec: str = DEFAULT_CODEC,
                 quality: str = DEFAULT_QUALITY, audio: bool = True, stem: Optional[str] = None,
                 scene_total: Optional[int] = None,
                 progress: Optional[Callable[[int, int, int], None]] = None,
                 check_abort: Optional[Callable[[], None]] = None,
                 seek_gap: Optional[float] = None) -> list[Clip]:
    """Write one file per (scene, start, end) range into `out_dir`. Returns the clips written.
    `progress(frames_done, frames_total, clips_done)`; `check_abort()` raises to stop (partial files
    are removed)."""
    if codec not in CODECS:
        raise ValueError(f"unknown clip codec {codec!r}; one of {', '.join(CODECS)}")
    c = CODECS[codec]
    if not codec_works(codec):
        raise RuntimeError(f"{c.label} is not available here"
                           + (" (it needs an NVIDIA GPU and driver)" if c.gpu else ""))
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or safe_stem(path)
    total = scene_total or (max(r[0] for r in ranges) + 1 if ranges else 1)
    clips = [Clip(s, a, b, clip_name(stem, s, total, c.ext)) for s, a, b in sorted(ranges, key=lambda r: r[1]) if b > a]
    if not clips:
        return []
    info = probe(path, count_frames=False)
    index = frame_index(path)
    if index is not None and info.frame_count and abs(len(index.pts) - info.frame_count) > 1:
        index = None
    state = {"done": 0, "total": sum(cl.end - cl.start for cl in clips), "clips": 0}
    try:
        try:
            if index is None:
                raise _Renumber()
            _export(path, clips, out_dir, c, quality, audio, index, state, progress, check_abort,
                    seek_gap if seek_gap is not None else SEEK_GAP_SECONDS)
        except _Renumber:
            for cl in clips:
                cl.frames_written = 0
                p = os.path.join(out_dir, cl.file)
                if os.path.exists(p):
                    os.remove(p)
            state.update(done=0, clips=0)
            _export(path, clips, out_dir, c, quality, audio, index, state, progress, check_abort, None)
    except BaseException:
        for cl in clips:
            for p in (os.path.join(out_dir, cl.file),):
                if os.path.exists(p) and cl.frames_written < cl.end - cl.start:
                    os.remove(p)
        raise
    return clips


def _export(path, clips, out_dir, codec, quality, audio, index, state, progress, check_abort, seek_gap):
    """seek_gap None: count frames from the start of the file (fallback)."""
    import av

    with av.open(path) as container:
        vs = container.streams.video[0]
        vs.thread_type = "AUTO"
        a_src = container.streams.audio[0] if (audio and container.streams.audio) else None
        rate = vs.average_rate or vs.guessed_rate or vs.base_rate
        rate = Fraction(rate) if rate else Fraction(0)
        tb = Fraction(vs.time_base)
        counting = seek_gap is None or index is None

        # runs of clips handled by one sequential pass
        runs: list[list[Clip]] = []
        for cl in clips:
            if runs and (counting or (index.time(cl.start) - index.time(runs[-1][-1].end)) <= seek_gap):
                runs[-1].append(cl)
            else:
                runs.append([cl])

        def time_of(n: int) -> float:
            if not counting:
                return index.time(n)
            t0 = index.time(0) if index is not None else 0.0
            return t0 + (n / float(rate) if rate else 0.0)

        for run in runs:
            _run(container, vs, a_src, run, out_dir, codec, quality, index, counting, rate, tb, time_of,
                 state, progress, check_abort)


def _run(container, vs, a_src, run, out_dir, codec, quality, index, counting, rate, tb, time_of,
         state, progress, check_abort):
    from av.video.frame import PictureType

    first, last = run[0].start, run[-1].end
    if not counting:                                 # counting always starts from a freshly opened file
        target = index.time(first) - SEEK_MARGIN_SECONDS
        ts = int(index.pts[0]) if target <= index.time(0) else int(target / tb)
        container.seek(ts, stream=vs, backward=True)

    bounds = [(time_of(cl.start), time_of(cl.end)) for cl in run]
    run_end_t = bounds[-1][1]
    audio_ctx = a_src.codec_context if a_src is not None else None
    part: Optional[_VideoPart] = None                # the clip being encoded
    part_i = -1
    done_video: dict[int, str] = {}                  # clip -> its finished video file, waiting for sound
    sound: dict[int, list] = {i: [] for i in range(len(run))}
    video_done = False
    audio_done = a_src is None
    audio_t = float("-inf")                          # end time of the last audio frame decoded
    count = -1
    k = 0                                            # first clip of the run whose video is not complete

    def video_tmp(i: int) -> str:
        return os.path.join(out_dir, run[i].file + ".video" + codec.ext)

    def finish_ready(final: bool = False):
        for i in sorted(done_video):
            if final or audio_done or audio_t >= bounds[i][1]:
                _finish_clip(done_video.pop(i), os.path.join(out_dir, run[i].file), codec, audio_ctx, sound.pop(i, []))
                state["clips"] += 1

    def close_part():
        nonlocal part, part_i
        if part is not None:
            part.close()
            done_video[part_i] = part.path
            part, part_i = None, -1

    streams = [vs] + ([a_src] if a_src is not None else [])
    try:
        for packet in container.demux(*streams):
            if packet.stream.type == "video":
                if video_done:
                    continue
                for frame in packet.decode():
                    if counting:
                        count += 1
                        n = count
                    else:
                        n = index.find(frame.pts)
                        if n is None:
                            raise _Renumber()
                    if n < first:
                        continue
                    while k < len(run) and n >= run[k].end:
                        k += 1
                    if part is not None and part_i != k:
                        close_part()
                    if n >= last or k >= len(run):
                        video_done = True
                        break
                    cl = run[k]
                    if n < cl.start:
                        continue
                    if part is None:
                        part = _VideoPart(video_tmp(k), codec, quality, frame.width, frame.height,
                                          frame.format.name, tb, rate)
                        part_i = k
                    if counting:
                        rel = round((time_of(n) - time_of(cl.start)) / float(tb))
                    else:
                        rel = int(index.pts[n]) - int(index.pts[cl.start])
                    frame.pict_type = PictureType.NONE   # the encoder places its own keyframes
                    part.add(frame, rel)
                    cl.frames_written += 1
                    state["done"] += 1
                    if cl.frames_written >= cl.end - cl.start:
                        close_part()
                    if state["done"] % 24 == 0:
                        if progress:
                            progress(state["done"], state["total"], state["clips"])
                        if check_abort:
                            check_abort()
            elif not audio_done:
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    sr = frame.sample_rate
                    t0 = float(frame.pts * frame.time_base)
                    t1 = t0 + frame.samples / sr
                    audio_t = max(audio_t, t1)
                    if t0 >= run_end_t:
                        audio_done = True
                        break
                    for i, (ca, cb) in enumerate(bounds):
                        if t1 <= ca or t0 >= cb or i not in sound:
                            continue
                        a = max(0, int(round((ca - t0) * sr)))
                        b = min(frame.samples, int(round((cb - t0) * sr)))
                        if b > a:
                            rel = max(0, int(round((t0 - ca) * sr)) + a)
                            sound[i].append(_slice_audio(frame, a, b, rel))
            finish_ready()
            if video_done and audio_done:
                break
        close_part()
        finish_ready(final=True)
        if progress:
            progress(state["done"], state["total"], state["clips"])
    except BaseException:
        if part is not None:
            part.abort()
        for path in done_video.values():
            if os.path.exists(path):
                os.remove(path)
        raise


# --------------------------------------------------------------------------------------------- #
# A scene list's export folder: the clips and scenes.json
# --------------------------------------------------------------------------------------------- #
def write_export(sl: SceneList, indices: list[int], out_dir: str, codec: str = DEFAULT_CODEC,
                 quality: str = DEFAULT_QUALITY, audio: bool = True, include_signals: bool = False,
                 progress: Optional[Callable[[int, int, int], None]] = None,
                 check_abort: Optional[Callable[[], None]] = None) -> dict:
    """Clips of segments `indices` (none when codec is JSON_ONLY) and scenes.json, in `out_dir`.
    scenes.json is the whole scene list (with its selection) plus a `clips` table."""
    os.makedirs(out_dir, exist_ok=True)
    stem = safe_stem(sl.source.path)
    indices = sorted(set(int(i) for i in indices if 0 <= int(i) < len(sl.segments)))
    clips: list[Clip] = []
    if codec != JSON_ONLY:
        ranges = [(i, sl.segments[i].start, sl.segments[i].end) for i in indices]
        clips = export_clips(sl.source.path, ranges, out_dir, codec, quality, audio, stem=stem,
                             scene_total=len(sl.segments), progress=progress, check_abort=check_abort)
    d = sl.to_dict(include_signals=include_signals)
    d["export"] = {"scenes": [i for i in indices], "codec": codec,
                   "quality": quality if codec != JSON_ONLY and CODECS[codec].quality != "none" else None,
                   "audio": bool(audio) and codec != JSON_ONLY}
    d["clips"] = [{"scene": cl.scene + 1, "index": cl.scene, "file": cl.file, "start": cl.start,
                   "end": cl.end, "frames": cl.end - cl.start,
                   "startTime": round(sl.seconds(cl.start), 4), "endTime": round(sl.seconds(cl.end), 4)}
                  for cl in sorted(clips, key=lambda c: c.scene)]
    path = os.path.join(out_dir, "scenes.json")
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    return {"folder": out_dir, "json": path, "clips": [cl.file for cl in sorted(clips, key=lambda c: c.scene)],
            "frames": sum(cl.end - cl.start for cl in clips)}
