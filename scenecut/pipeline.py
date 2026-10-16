"""The detection pipeline as plain functions, shared by the simple and the advanced nodes."""
from __future__ import annotations

from typing import Callable, Optional

from . import validate
from .detect import detect_video
from .media import VideoInfo, probe
from .omnishotcut import OmniShotCut
from .presets import Preset
from .refine import (build_scene_list, handle_transitions, merge_short, merge_ultra_short,
                     snap_cuts)
from .scenes import SceneList
from .validate import ValidatorConfig


def seconds_to_frames(seconds: float, fps) -> int:
    return max(0, int(round(float(seconds) * float(fps)))) if fps else 0


def detect(model: OmniShotCut, path: str, overlap: int = 20, confidence: float = 0.0,
           batch_windows: int = 8, backend: str = "auto", info: Optional[VideoInfo] = None,
           progress: Optional[Callable[[int, int], None]] = None,
           check_abort: Optional[Callable[[], None]] = None) -> SceneList:
    """Detector output as a scene list: every proposed range, transitions labelled, untouched."""
    info = info or probe(path)
    total = info.frame_count
    raw = detect_video(model, path, overlap=overlap, confidence=confidence,
                       batch_windows=batch_windows, backend=backend,
                       progress=(lambda done: progress(done, total)) if progress else None,
                       check_abort=check_abort)
    return build_scene_list(raw, info)


def refine(sl: SceneList, transitions: str = "merge_previous", min_shot_seconds: float = 0.4,
           snap_radius: int = 2, ultra_short_seconds: Optional[float] = 0.05) -> SceneList:
    sl = handle_transitions(sl, transitions)
    sl = merge_short(sl, seconds_to_frames(min_shot_seconds, sl.fps))
    sl = snap_cuts(sl, snap_radius)
    if ultra_short_seconds:
        sl = merge_ultra_short(sl, ultra_short_seconds)
    return sl


def run_preset(model: OmniShotCut, path: str, preset: Preset, dino=None, sscd=None,
               cache: Optional[dict] = None, **kw) -> SceneList:
    """Everything the simple Detect node does. `dino` (a `validate.DinoEmbedder`) turns
    validation on when the preset asks for it; `sscd` is its optional tie-breaker."""
    validating = dino is not None and preset.validate
    sl = detect(model, path, overlap=preset.overlap, confidence=preset.confidence, **kw)
    # The fragment clean-up runs last, after validation, as in the advanced chain.
    sl = refine(sl, transitions=preset.transitions, min_shot_seconds=preset.min_shot_seconds,
                snap_radius=preset.snap_radius,
                ultra_short_seconds=None if validating else preset.ultra_short_seconds)
    if validating:
        cfg = ValidatorConfig(window=preset.validate_window,
                              flash_sensitivity=preset.flash_sensitivity, use_sscd=sscd is not None)
        sl = validate.run(sl, dino, cfg, sscd, cache=cache, progress=kw.get("progress"),
                          check_abort=kw.get("check_abort"))
        sl = merge_ultra_short(sl, preset.ultra_short_seconds)
    return sl
