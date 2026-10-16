"""The scene list: what every SceneCut node passes along.

A `SceneList` covers a video's frames `[0, frame_count)` with gapless `Segment`s. Each segment
records the boundary it starts at (`Segment.cut`), so detector confidence, snapping and validator
evidence travel with the shot they belong to. Nothing is deleted: a boundary removed by any stage
is moved to `rejected` with the stage and the reason, and a transition folded into a shot stays
listed on that shot in `absorbed`.

Frames are the unit; seconds are derived from the exact frame rate. All of it round-trips through
JSON (`to_dict` / `from_dict`), so a scene list can be saved, edited by other tools, and loaded
into a different workflow.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np

from .media import VideoInfo

FORMAT = "scenecut.scenes"
VERSION = 1


@dataclass
class Cut:
    """Evidence about one boundary, at frame `frame` (the first frame of the new segment)."""
    frame: int
    source: str = "detector"           # detector | manual | transition
    relation: str = "hard_cut"         # how the new segment begins (OmniShotCut inter label)
    confidence: Optional[float] = None # detector localisation confidence
    detected_frame: Optional[int] = None  # before snapping
    validation: Optional[dict] = None  # validator features and decision
    review: Optional[dict] = None      # moved by hand in the viewer: {from, mode}

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Cut":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Segment:
    start: int
    end: int                            # exclusive
    kind: str = "shot"                  # shot | transition
    transition: Optional[str] = None    # dissolve, wipe, ... when kind == "transition"
    cut: Optional[Cut] = None           # the boundary at `start` (None for the first segment)
    absorbed: list = field(default_factory=list)  # transitions folded into this shot

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = {"start": self.start, "end": self.end, "kind": self.kind}
        if self.transition:
            d["transition"] = self.transition
        if self.cut is not None:
            d["cut"] = self.cut.to_dict()
        if self.absorbed:
            d["absorbed"] = self.absorbed
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(start=int(d["start"]), end=int(d["end"]), kind=d.get("kind", "shot"),
                   transition=d.get("transition"),
                   cut=Cut.from_dict(d["cut"]) if d.get("cut") else None,
                   absorbed=list(d.get("absorbed", [])))


@dataclass
class SceneList:
    source: VideoInfo
    frame_count: int
    segments: list[Segment]
    rejected: list[dict] = field(default_factory=list)   # {frame, stage, reason, cut: {...}}
    history: list[dict] = field(default_factory=list)    # {stage, params, before, after}
    signals: dict = field(default_factory=dict)          # name -> np.ndarray, one value per frame
    selection: Optional[list[int]] = None                # chosen segment indices (None = all)

    # ---- time ----
    @property
    def fps(self) -> Fraction:
        return self.source.fps

    def seconds(self, frame: int) -> float:
        return float(frame / self.fps) if self.fps else 0.0

    # ---- queries ----
    @property
    def cuts(self) -> list[int]:
        return [s.start for s in self.segments[1:]]

    @property
    def shots(self) -> list[Segment]:
        return [s for s in self.segments if s.kind == "shot"]

    def check(self) -> None:
        """Raise if the segments do not tile [0, frame_count) in order."""
        if not self.segments:
            raise ValueError("scene list has no segments")
        if self.segments[0].start != 0 or self.segments[-1].end != self.frame_count:
            raise ValueError("segments must cover the whole video")
        for a, b in zip(self.segments, self.segments[1:]):
            if a.end != b.start or a.length <= 0:
                raise ValueError(f"segments not contiguous at frame {a.end}")
        if self.segments[-1].length <= 0:
            raise ValueError("empty last segment")

    # ---- editing primitives used by every stage ----
    def copy(self) -> "SceneList":
        """Deep copy for a stage to modify. Signals are shared (read-only)."""
        signals = self.signals
        self.signals = {}
        try:
            c = copy.deepcopy(self)
        finally:
            self.signals = signals
        c.signals = signals
        return c

    def remove_boundary(self, index: int, stage: str, reason: str, **extra) -> None:
        """Merge segment `index` into segment `index - 1`, recording the boundary as rejected."""
        if index <= 0 or index >= len(self.segments):
            raise IndexError(index)
        a, b = self.segments[index - 1], self.segments[index]
        rec = {"frame": b.start, "stage": stage, "reason": reason}
        if b.cut is not None:
            rec["cut"] = b.cut.to_dict()
        rec.update(extra)
        self.rejected.append(rec)
        a.end = b.end
        if b.kind == "transition":
            a.absorbed.append({"start": b.start, "end": b.end, "transition": b.transition})
        a.absorbed.extend(b.absorbed)
        if a.kind == "transition" and b.kind == "shot":
            # A transition swallowing a shot would hide the shot; keep the result a shot.
            a.absorbed.insert(0, {"start": a.start, "end": b.start, "transition": a.transition})
            a.kind, a.transition = "shot", None
        del self.segments[index]

    def move_boundary(self, index: int, frame: int, **review) -> None:
        """Move the boundary at the start of segment `index` to `frame`, strictly inside the two
        segments it separates. `review` (e.g. mode="auto") is stored on the cut."""
        if index <= 0 or index >= len(self.segments):
            raise IndexError(index)
        a, b = self.segments[index - 1], self.segments[index]
        if not a.start < frame < b.end:
            raise ValueError(f"frame {frame} is outside the scenes around the cut ({a.start + 1}-{b.end - 1})")
        old = b.start
        a.end = b.start = int(frame)
        if b.cut is None:
            b.cut = Cut(frame=int(frame), source="manual")
        first = (b.cut.review or {}).get("from", old)
        b.cut.frame = int(frame)
        b.cut.review = {"from": first, **review}

    def selected_indices(self) -> list[int]:
        """The chosen segments, or all of them when nothing is selected."""
        if self.selection is None:
            return list(range(len(self.segments)))
        return [i for i in sorted(set(self.selection)) if 0 <= i < len(self.segments)]

    def record(self, stage: str, params: dict, before: int, **extra) -> None:
        entry = {"stage": stage, "params": params, "before": before, "after": len(self.segments)}
        entry.update(extra)
        self.history.append(entry)

    # ---- serialisation ----
    def to_dict(self, include_signals: bool = False) -> dict:
        d = {
            "format": FORMAT,
            "version": VERSION,
            "source": self.source.to_dict(),
            "frameCount": self.frame_count,
            "segments": [s.to_dict() for s in self.segments],
            "rejected": self.rejected,
            "history": self.history,
        }
        if self.selection is not None:
            d["selection"] = self.selection
            # Convenience for other tools: the selected scenes with times.
            d["selectedScenes"] = [
                {"index": i, "start": self.segments[i].start, "end": self.segments[i].end,
                 "startTime": round(self.seconds(self.segments[i].start), 4),
                 "endTime": round(self.seconds(self.segments[i].end), 4)}
                for i in self.selected_indices()]
        if include_signals and self.signals:
            d["signals"] = {k: _encode_array(v) for k, v in self.signals.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SceneList":
        if d.get("format") != FORMAT:
            raise ValueError("not a SceneCut scene list")
        if int(d.get("version", 0)) > VERSION:
            raise ValueError(f"scene list version {d['version']} is newer than this SceneCut")
        sl = cls(source=VideoInfo.from_dict(d["source"]), frame_count=int(d["frameCount"]),
                 segments=[Segment.from_dict(s) for s in d["segments"]],
                 rejected=list(d.get("rejected", [])), history=list(d.get("history", [])),
                 signals={k: _decode_array(v) for k, v in d.get("signals", {}).items()},
                 selection=d.get("selection"))
        sl.check()
        return sl

    def summary(self) -> str:
        n = len(self.segments)
        shots = len(self.shots)
        dur = self.seconds(self.frame_count)
        lines = [
            f"{shots} shots" + (f" + {n - shots} transitions" if n != shots else "")
            + f" in {_hms(dur)} ({self.frame_count} frames @ {float(self.fps):.3f} fps)",
            f"mean shot {dur / max(1, shots):.2f}s, {len(self.cuts) / max(1e-6, dur / 60):.1f} cuts/min",
        ]
        for h in self.history:
            lines.append(f"  {h['stage']}: {h['before']} -> {h['after']}")
        return "\n".join(lines)


def _hms(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}" if h else f"{m:d}:{s:06.3f}"


def timecode(seconds: float) -> str:
    return _hms(seconds)


def _encode_array(a: np.ndarray) -> dict:
    a = np.asarray(a, np.float32)
    return {"dtype": "float32", "shape": list(a.shape),
            "data": base64.b64encode(a.tobytes()).decode("ascii")}


def _decode_array(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["data"]), np.float32).reshape(d["shape"]).copy()
