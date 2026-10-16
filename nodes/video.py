from __future__ import annotations

import os

from comfy_api.latest import io

from ..scenecut.media import probe
from .common import CATEGORY, video_from_path


def _resolve(path: str) -> str:
    path = (path or "").strip().strip('"').strip("'")
    return os.path.abspath(os.path.expanduser(path)) if path else ""


class SceneCutLoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutLoadVideo",
            display_name="SceneCut Load Video (Path)",
            category=CATEGORY,
            search_aliases=["scene video", "load film", "open video path", "video from path"],
            description="Load a video from any path on disk, so long films need not be copied into "
                        "ComfyUI's input folder. Outputs a standard VIDEO; for files in input/, the "
                        "stock Load Video node works just as well.",
            inputs=[
                io.String.Input("path", default="", placeholder="/path/to/video.mp4",
                                tooltip="Absolute path to a video file."),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Int.Output(display_name="frame_count"),
                io.Float.Output(display_name="fps"),
                io.Float.Output(display_name="duration"),
            ],
        )

    @classmethod
    def execute(cls, path) -> io.NodeOutput:
        full = _resolve(path)
        info = probe(full)
        return io.NodeOutput(video_from_path(full), info.frame_count, info.fps_float, info.duration)

    @classmethod
    def fingerprint_inputs(cls, path):
        full = _resolve(path)
        return os.path.getmtime(full) if os.path.isfile(full) else float("nan")

    @classmethod
    def validate_inputs(cls, path):
        full = _resolve(path)
        if not full:
            return "Enter the path of a video file."
        if not os.path.isfile(full):
            return f"Video not found: {full}"
        return True
