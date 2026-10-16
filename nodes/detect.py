from __future__ import annotations

from comfy_api.latest import io, ui

from ..scenecut import pipeline
from ..scenecut.media import probe
from ..scenecut.presets import DEFAULT_PRESET, PRESETS, get_preset
from ..scenecut.refine import TRANSITION_MODES, merge_ultra_short
from ..scenecut.validate import ValidatorConfig
from .common import (CATEGORY, CATEGORY_ADVANCED, PRESET_LABELS, Detector, Progress, Scenes,
                     detector_choices, device_choices, device_tooltip, get_detector, image_output_input,
                     images_output, scene_images, video_path)
from .validation import get_validator, run_validation

TRANSITION_TOOLTIP = (
    "What to do with transitions (dissolves, wipes, fades...). merge_previous: the transition "
    "becomes the end of the shot before it. merge_next: the start of the shot after it. split: the "
    "cut goes in the middle. keep: transitions stay as their own segments. Merged transitions are "
    "still recorded on the shot that absorbed them.")


def _preset_help() -> str:
    return "\n".join(f"{p.label}: {p.description}" for p in PRESETS.values())


def _run_detect(handle, path, overlap, confidence, batch, decoder):
    info = probe(path)
    model = handle.ready()
    progress = Progress(info.frame_count)
    return pipeline.detect(model, path, overlap=overlap, confidence=confidence,
                           batch_windows=batch, backend=decoder, info=info,
                           progress=progress, check_abort=progress.check_abort)


class SceneCutDetect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutDetect",
            display_name="SceneCut Detect",
            category=CATEGORY,
            search_aliases=["scene detection", "shot detection", "cut detection", "shot boundary"],
            description="Find every cut in a video with OmniShotCut, clean the result up with a "
                        "preset and check each cut with DINOv3. Models download on first use "
                        "(OmniShotCut 164 MB; DINOv3 1.2 GB and SSCD 178 MB when validating).",
            inputs=[
                io.Video.Input("video"),
                io.Combo.Input("preset", options=list(PRESET_LABELS),
                               default=PRESETS[DEFAULT_PRESET].label, tooltip=_preset_help()),
                io.Boolean.Input("validate", default=True,
                                 tooltip="Check every cut with DINOv3 and merge false ones "
                                         "(flashes, fast motion, a continuous shot split in two). "
                                         "Slower; off uses the detector's cuts as refined."),
                io.Combo.Input("device", options=device_choices(), default="default",
                               tooltip=device_tooltip()),
                image_output_input(),
            ],
            outputs=[
                Scenes.Output(display_name="scenes"),
                io.Int.Output(display_name="shot_count"),
                io.String.Output(display_name="summary"),
                images_output(),
            ],
        )

    @classmethod
    def execute(cls, video, preset, validate=True, device="default", output_images="off") -> io.NodeOutput:
        p = get_preset(PRESET_LABELS.get(preset, DEFAULT_PRESET))
        path = video_path(video)
        sl = _run_detect(get_detector(device=device), path, p.overlap, p.confidence, 8, "auto")
        # The fragment clean-up runs last, after validation, as in the advanced chain.
        sl = pipeline.refine(sl, transitions=p.transitions, min_shot_seconds=p.min_shot_seconds,
                             snap_radius=p.snap_radius,
                             ultra_short_seconds=None if validate else p.ultra_short_seconds)
        if validate:
            handle = get_validator(device=device)
            cfg = ValidatorConfig(window=p.validate_window, flash_sensitivity=p.flash_sensitivity,
                                  use_sscd=handle.sscd_path is not None)
            sl = merge_ultra_short(run_validation(sl, handle, cfg), p.ultra_short_seconds)
        summary = sl.summary()
        return io.NodeOutput(sl, len(sl.shots), summary, scene_images(sl, output_images),
                             ui=ui.PreviewText(summary))


class SceneCutDetectorLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutDetectorLoader",
            display_name="SceneCut Detector Loader",
            category=CATEGORY_ADVANCED,
            description="Load OmniShotCut from models/scenecut (downloaded there on first use).",
            inputs=[
                io.Combo.Input("model", options=detector_choices()),
                io.Combo.Input("device", options=device_choices(), default="default",
                               tooltip=device_tooltip()),
            ],
            outputs=[Detector.Output(display_name="detector")],
        )

    @classmethod
    def execute(cls, model, device) -> io.NodeOutput:
        return io.NodeOutput(get_detector(model, device))


class SceneCutDetectAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutDetectAdvanced",
            display_name="SceneCut Detect (Advanced)",
            category=CATEGORY_ADVANCED,
            description="Raw OmniShotCut output: every proposed shot and transition, nothing merged. "
                        "Follow with SceneCut Refine. Its output is cached, so tuning later stages "
                        "does not rerun detection.",
            inputs=[
                io.Video.Input("video"),
                Detector.Input("detector", optional=True,
                               tooltip="From SceneCut Detector Loader. Default: OmniShotCut on the GPU."),
                io.Int.Input("overlap", default=20, min=0, max=90, step=2,
                             tooltip="Frames shared by neighbouring 100-frame windows. Each window "
                                     "is trusted only for its middle; more overlap gives boundaries "
                                     "near window edges more context."),
                io.Float.Input("confidence", default=0.0, min=0.0, max=0.95, step=0.01,
                               tooltip="Drop boundaries whose localisation confidence is below this "
                                       "(0 keeps everything the model proposes). The dropped shot "
                                       "merges into the next one."),
                io.Int.Input("batch_windows", default=8, min=1, max=64, advanced=True,
                             tooltip="Windows per forward pass. Speed only; results are identical."),
                io.Combo.Input("decoder", options=["auto", "ffmpeg", "pyav"], default="auto",
                               advanced=True,
                               tooltip="auto uses the ffmpeg binary when present, otherwise PyAV. "
                                       "Both give identical frames."),
                image_output_input(),
            ],
            outputs=[
                Scenes.Output(display_name="scenes"),
                io.String.Output(display_name="summary"),
                images_output(),
            ],
        )

    @classmethod
    def execute(cls, video, overlap, confidence, batch_windows, decoder, output_images="off",
                detector=None) -> io.NodeOutput:
        handle = detector or get_detector()
        sl = _run_detect(handle, video_path(video), overlap, confidence, batch_windows, decoder)
        summary = sl.summary()
        return io.NodeOutput(sl, summary, scene_images(sl, output_images), ui=ui.PreviewText(summary))


class SceneCutRefine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutRefine",
            display_name="SceneCut Refine",
            category=CATEGORY_ADVANCED,
            description="Turn raw detections into clean shots: handle transitions, merge shots "
                        "that are too short, snap cuts to the exact frame.",
            inputs=[
                Scenes.Input("scenes"),
                io.Combo.Input("transitions", options=list(TRANSITION_MODES), default="merge_previous",
                               tooltip=TRANSITION_TOOLTIP),
                io.Float.Input("min_shot_seconds", default=0.40, min=0.0, max=10.0, step=0.01,
                               tooltip="Shots shorter than this merge into their longer neighbour. "
                                       "0 disables."),
                io.Int.Input("snap_radius", default=2, min=0, max=12,
                             tooltip="Move each cut up to this many frames to the frame with the "
                                     "largest pixel change. 0 disables."),
                io.Float.Input("ultra_short_seconds", default=0.05, min=0.0, max=1.0, step=0.01,
                               tooltip="Final clean-up: fragments this short merge into a neighbour "
                                       "whatever their kind. 0 disables."),
                image_output_input(),
            ],
            outputs=[
                Scenes.Output(display_name="scenes"),
                io.Int.Output(display_name="shot_count"),
                io.String.Output(display_name="summary"),
                images_output(),
            ],
        )

    @classmethod
    def execute(cls, scenes, transitions, min_shot_seconds, snap_radius, ultra_short_seconds,
                output_images="off") -> io.NodeOutput:
        sl = pipeline.refine(scenes, transitions=transitions, min_shot_seconds=min_shot_seconds,
                             snap_radius=snap_radius, ultra_short_seconds=ultra_short_seconds or None)
        summary = sl.summary()
        return io.NodeOutput(sl, len(sl.shots), summary, scene_images(sl, output_images),
                             ui=ui.PreviewText(summary))
