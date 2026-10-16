from __future__ import annotations

import json
import os

import folder_paths
from comfy_api.latest import io, ui

from ..scenecut.export import (CODECS, DEFAULT_CODEC, DEFAULT_QUALITY, JSON_ONLY, QUALITY, codec_works,
                               write_export)
from ..scenecut.scenes import SceneList
from .common import CATEGORY, Progress, Scenes, expand_prefix, image_output_input, images_output, scene_images

CLIP_CHOICES = {"none (scene list only)": JSON_ONLY, **{c.label: c.id for c in CODECS.values()}}
CLIP_DEFAULT = CODECS[DEFAULT_CODEC].label


def _resolve_json(path: str) -> str:
    """A scene list file from a path: absolute, or under output/ then input/; a folder written by
    Save Scenes or the viewer's Export resolves to the scenes.json inside it."""
    path = os.path.expanduser(path.strip().strip('"'))
    cands = [path] if os.path.isabs(path) else [os.path.join(base, path) for base in
                                                (folder_paths.get_output_directory(), folder_paths.get_input_directory())]
    for full in cands:
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "scenes.json")):
            return os.path.join(full, "scenes.json")
        if os.path.isfile(full):
            return full
    return cands[0]


def new_export_folder(sl, prefix: str = "scenecut/{video}/scenes") -> tuple[str, str]:
    """A new numbered folder under the output folder (scenes_00001, scenes_00002, ...):
    (full path, path relative to output)."""
    for _ in range(100):
        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            expand_prefix(prefix, sl), folder_paths.get_output_directory())
        folder = os.path.join(out_dir, f"{name}_{counter:05}")
        try:
            os.makedirs(folder)
        except FileExistsError:                       # another export took this number first
            continue
        return folder, os.path.join(subfolder, f"{name}_{counter:05}")
    raise RuntimeError("could not create an export folder")


class SceneCutSaveScenes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutSaveScenes",
            display_name="SceneCut Save Scenes",
            category=CATEGORY,
            is_output_node=True,
            search_aliases=["export clips", "split video", "save shots", "cut video into scenes"],
            description="Cut the video into one file per scene (frame-exact, with sound) and write the "
                        "scene list next to them: output/scenecut/<video>/scenes_00001/. Only the "
                        "selected scenes when a SceneCut Viewer selected some. The viewers' Export "
                        "button does the same without this node.",
            inputs=[
                Scenes.Input("scenes"),
                io.String.Input("filename_prefix", default="scenecut/{video}/scenes",
                                tooltip="Folder under the output folder; {video} is the video's file "
                                        "name without extension. A number is added: scenes_00001."),
                io.Combo.Input("clips", options=list(CLIP_CHOICES), default=CLIP_DEFAULT,
                               tooltip="Codec of the clip files. H.264 plays everywhere; the NVIDIA "
                                       "ones encode on the GPU; ProRes and FFV1 are for editing "
                                       "(large files). 'none' writes only scenes.json."),
                io.Combo.Input("quality", options=list(QUALITY), default=DEFAULT_QUALITY,
                               tooltip="high: visually lossless. medium: smaller. small: previews. "
                                       "(ProRes and FFV1 ignore it.)"),
                io.Boolean.Input("audio", default=True, tooltip="Include the sound in each clip."),
                io.Boolean.Input("include_signals", default=True, advanced=True,
                                 tooltip="Also store the per-frame difference and brightness curves "
                                         "in scenes.json (about 11 bytes per frame), so snapping and "
                                         "the viewer work after loading without decoding again."),
            ],
            outputs=[
                io.String.Output(display_name="path", tooltip="scenes.json"),
                io.String.Output(display_name="folder", tooltip="The folder with the clips"),
            ],
        )

    @classmethod
    def execute(cls, scenes, filename_prefix, clips=CLIP_DEFAULT, quality=DEFAULT_QUALITY, audio=True,
                include_signals=True) -> io.NodeOutput:
        codec = CLIP_CHOICES.get(clips, clips if clips in CODECS or clips == JSON_ONLY else DEFAULT_CODEC)
        if codec != JSON_ONLY and not codec_works(codec):
            raise RuntimeError(f"{CODECS[codec].label} is not available on this machine"
                               + (" (it needs an NVIDIA GPU)." if CODECS[codec].gpu else ".")
                               + " Choose H.264.")
        sl = scenes
        indices = sl.selected_indices()
        folder, rel = new_export_folder(sl, filename_prefix)
        frames = sum(sl.segments[i].length for i in indices) if codec != JSON_ONLY else 0
        progress = Progress(max(1, frames))
        result = write_export(sl, indices, folder, codec, quality, audio, include_signals,
                              progress=lambda done, total, n: progress(done, total),
                              check_abort=progress.check_abort)
        n = len(result["clips"])
        text = f"saved {rel}/: scenes.json" + (f" and {n} clip{'s' if n != 1 else ''}" if n else "")
        return io.NodeOutput(result["json"], folder, ui=ui.PreviewText(text))


class SceneCutLoadScenes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutLoadScenes",
            display_name="SceneCut Load Scenes",
            category=CATEGORY,
            description="Load a scene list saved by SceneCut Save Scenes or a viewer's Export: the "
                        "folder or its scenes.json. Relative paths are looked up in the output "
                        "folder, then the input folder.",
            inputs=[
                io.String.Input("path", default="scenecut/video/scenes_00001"),
                io.String.Input("video_path", default="", optional=True,
                                tooltip="Use this video instead of the one recorded in the file "
                                        "(moved file, other machine). Must be the same video."),
                image_output_input(),
            ],
            outputs=[
                Scenes.Output(display_name="scenes"),
                io.Int.Output(display_name="shot_count"),
                images_output(),
            ],
        )

    @classmethod
    def execute(cls, path, video_path="", output_images="off") -> io.NodeOutput:
        with open(_resolve_json(path), encoding="utf-8") as f:
            sl = SceneList.from_dict(json.load(f))
        if video_path.strip():
            from dataclasses import replace

            sl.source = replace(sl.source, path=os.path.abspath(os.path.expanduser(video_path.strip())))
        return io.NodeOutput(sl, len(sl.shots), scene_images(sl, output_images))

    @classmethod
    def fingerprint_inputs(cls, path, video_path="", output_images="off"):
        full = _resolve_json(path)
        return (os.path.getmtime(full) if os.path.isfile(full) else float("nan"), video_path, output_images)

    @classmethod
    def validate_inputs(cls, path, video_path="", output_images="off"):
        if not os.path.isfile(_resolve_json(path)):
            return f"Scene list not found: {path}"
        return True
