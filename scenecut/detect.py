"""Windowed OmniShotCut inference over a frame stream.

The video is never held in memory or written to disk: frames arrive from the decoder, a buffer
keeps the current window, and each window runs as soon as it is complete. Window layout (stride
`window - overlap`, the overlap split in half between neighbours, the last window padded with
black) and the merge of per-window boundaries reproduce the reference implementation, so the
same checkpoint gives the same boundaries.

The same pass records two cheap per-frame signals that later stages use without decoding again:

  diff   mean absolute RGB difference to the previous frame (frame 0: 0)
  luma   mean Rec.601 luma
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import torch

from .omnishotcut import INTER_LABELS, INTRA_LABELS, OmniShotCut, strict_fp32

# Boundaries proposed by neighbouring windows within this many frames are the same boundary.
DUPLICATE_TOLERANCE = 2


@dataclass
class Boundary:
    """One shot proposed by the model: it ends (exclusive) at `end`."""
    end: int
    intra: int            # index into INTRA_LABELS
    inter: int            # index into INTER_LABELS
    confidence: float     # peak of the range head's distribution
    # How the shot *after* this boundary begins, from the window that saw the boundary in its
    # trusted region. The first query of every window is labelled new_start (it has no visible
    # predecessor), so a shot's own inter label is often a window artefact; this one is not.
    next_inter: Optional[int] = None

    @property
    def intra_label(self) -> str:
        return INTRA_LABELS[self.intra] if 0 <= self.intra < len(INTRA_LABELS) else str(self.intra)

    @property
    def inter_label(self) -> str:
        return INTER_LABELS[self.inter] if 0 <= self.inter < len(INTER_LABELS) else str(self.inter)

    @property
    def next_inter_label(self) -> Optional[str]:
        if self.next_inter is None or not 0 <= self.next_inter < len(INTER_LABELS):
            return None
        return INTER_LABELS[self.next_inter]


@dataclass
class RawDetection:
    frame_count: int
    boundaries: list[Boundary]
    rejected_low_confidence: int = 0
    diff: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    luma: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    params: dict = field(default_factory=dict)

    def ranges(self) -> list[tuple[int, int, Boundary]]:
        """Contiguous [start, end) ranges, one per boundary, starting at frame 0."""
        out, start = [], 0
        for b in self.boundaries:
            end = min(b.end, self.frame_count)
            if end <= start:
                continue
            out.append((start, end, b))
            start = end
        return out


class _Signals:
    def __init__(self):
        self.diff: list[np.ndarray] = []
        self.luma: list[np.ndarray] = []
        self.prev: Optional[np.ndarray] = None

    def add(self, chunk: np.ndarray) -> None:
        f = chunk.astype(np.int16)
        prev = f[:1] if self.prev is None else self.prev[None]
        d = np.abs(np.diff(np.concatenate([prev, f]), axis=0)).mean(axis=(1, 2, 3))
        self.diff.append(d.astype(np.float32))
        self.luma.append((f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114)
                         .mean(axis=(1, 2)).astype(np.float32))
        self.prev = f[-1]

    def result(self):
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.float32)  # noqa: E731
        return cat(self.diff), cat(self.luma)


def _window_boundaries(logits: dict, i: int, window_start: int, valid_start: int, valid_end: int,
                       valid_len: int, confidence: float, stats: dict) -> list[Boundary]:
    """Turn one window's query predictions into the boundaries this window is trusted for.

    Queries are read in order; each one's range head says where its shot ends, chained from the
    previous query's end. A query whose end does not advance is idle. A boundary is kept only when
    it falls inside this window's valid region; below `confidence` it is dropped (the shot then
    merges into the next) but the chain still advances, so later queries stay aligned."""
    intra = logits["intra"][i].softmax(-1)[:, :-1].argmax(-1).tolist()
    inter = logits["inter"][i].softmax(-1)[:, :-1].argmax(-1).tolist()
    peak = logits["range"][i].softmax(-1)[:, :-1].max(-1)
    ends, confs = peak.indices.tolist(), peak.values.tolist()

    active, start = [], 0          # the window's shots in order: (query, end)
    for q in range(len(ends)):
        end = min(int(ends[q]), valid_len)
        if start >= end:
            continue
        active.append((q, end))
        start = end
        if end >= valid_len:
            break

    out = []
    for j, (q, end) in enumerate(active):
        g = window_start + end
        if not valid_start < g <= valid_end:
            continue
        if confs[q] < confidence:
            stats["rejected"] += 1
            continue
        nxt = int(inter[active[j + 1][0]]) if j + 1 < len(active) else None
        out.append(Boundary(g, int(intra[q]), int(inter[q]), float(confs[q]), next_inter=nxt))
    return out


def _merge(full: list[Boundary], new: list[Boundary]) -> None:
    for b in sorted(new, key=lambda b: b.end):
        if full and abs(b.end - full[-1].end) <= DUPLICATE_TOLERANCE:
            continue
        full.append(b)


@torch.inference_mode()
def detect_stream(model: OmniShotCut, chunks: Iterable[np.ndarray], overlap: int = 20,
                  confidence: float = 0.0, batch_windows: int = 1,
                  progress: Optional[Callable[[int], None]] = None,
                  check_abort: Optional[Callable[[], None]] = None) -> RawDetection:
    """Run OmniShotCut over uint8 RGB chunks (N, H, W, 3) already at the model's process size.

    `progress(frames_done)` is called after each batch of windows; `check_abort()` may raise to
    stop. `batch_windows` runs several windows per forward pass (speed only)."""
    cfg = model.cfg
    window = cfg.window
    if not 0 <= overlap < window:
        raise ValueError(f"overlap must be in [0, {window - 1}]")
    if not 0.0 <= confidence < 1.0:
        raise ValueError("confidence must be in [0, 1)")
    stride = window - overlap
    left = overlap // 2
    right = overlap - left
    device = next(model.parameters()).device

    signals = _Signals()
    stats = {"rejected": 0}
    boundaries: list[Boundary] = []
    buf = np.zeros((0, cfg.process_height, cfg.process_width, 3), np.uint8)
    buf_start = 0            # global frame index of buf[0]
    win_start = 0            # next window to run
    total = 0
    pending: list[tuple[np.ndarray, int, int, int, int]] = []
    done = False

    def flush():
        if not pending:
            return
        x = torch.from_numpy(np.stack([p[0] for p in pending])).to(device, non_blocking=True)
        with strict_fp32():
            logits = {k: v.float().cpu() for k, v in model(x).items()}
        for i, (_, ws, vs, ve, vl) in enumerate(pending):
            _merge(boundaries, _window_boundaries(logits, i, ws, vs, ve, vl, confidence, stats))
        pending.clear()
        if progress:
            progress(total)

    it = iter(chunks)
    while not done:
        try:
            chunk = next(it)
        except StopIteration:
            done = True
            chunk = None
        if chunk is not None and len(chunk):
            if chunk.shape[1:] != buf.shape[1:]:
                raise ValueError(f"frames must be {cfg.process_width}x{cfg.process_height}")
            signals.add(chunk)
            buf = np.concatenate([buf, chunk])
            total += len(chunk)
        if check_abort:
            check_abort()

        # A window can run once a frame past its end exists (then it is not the last one), or at
        # the end of the stream (then it is, and it is padded).
        while win_start < total and (win_start + window < total or done):
            win_end = win_start + window
            lo = win_start - buf_start
            frames = buf[lo:lo + window]
            valid_len = len(frames)
            if valid_len < window:
                frames = np.concatenate(
                    [frames, np.zeros((window - valid_len,) + frames.shape[1:], np.uint8)])
            last = win_end >= total
            valid_start = 0 if win_start == 0 else win_start + left
            valid_end = total if last else win_end - right
            pending.append((frames, win_start, valid_start, valid_end, valid_len))
            if len(pending) >= batch_windows:
                flush()
            if last:
                win_start = total
                break
            win_start += stride
        # Drop frames no future window needs.
        keep_from = min(win_start, total) - buf_start
        if keep_from > 0:
            buf = buf[keep_from:]
            buf_start += keep_from
    flush()

    if total == 0:
        raise RuntimeError("the video produced no frames")
    diff, luma = signals.result()
    return RawDetection(
        frame_count=total,
        boundaries=boundaries,
        rejected_low_confidence=stats["rejected"],
        diff=diff, luma=luma,
        params={"overlap": overlap, "confidence": confidence, "window": window},
    )


def detect_video(model: OmniShotCut, path: str, overlap: int = 20, confidence: float = 0.0,
                 batch_windows: int = 8, backend: str = "auto",
                 progress: Optional[Callable[[int], None]] = None,
                 check_abort: Optional[Callable[[], None]] = None) -> RawDetection:
    """Decode `path` at the model's resolution on a background thread and detect as it arrives."""
    from .media import iter_frames, prefetch

    frames = prefetch(iter_frames(path, model.cfg.process_width, model.cfg.process_height,
                                  backend=backend))
    try:
        return detect_stream(model, frames, overlap=overlap, confidence=confidence,
                             batch_windows=batch_windows, progress=progress,
                             check_abort=check_abort)
    finally:
        frames.close()
