"""Glue between the pure `scenecut` package and ComfyUI: types, model files, VRAM, progress."""
from __future__ import annotations

import logging
import os
import re
import uuid

import numpy as np
import torch

import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io

from ..scenecut import weights
from ..scenecut.frames import (IMAGE_MODE_HELP, IMAGE_MODES, check_memory, frames_for, images_key,
                               images_to_video, read_frames, single_scene)
from ..scenecut.omnishotcut import load_omnishotcut
from ..scenecut.presets import PRESETS

log = logging.getLogger("SceneCut")

MODEL_FOLDER = "scenecut"
CATEGORY = "SceneCut"
CATEGORY_ADVANCED = "SceneCut/advanced"

Scenes = io.Custom("SCENECUT_SCENES")
Detector = io.Custom("SCENECUT_DETECTOR")

PRESET_LABELS = {p.label: key for key, p in PRESETS.items()}


def device_choices() -> list[str]:
    """default (ComfyUI's device, managed with the other models), cpu, and every CUDA device."""
    out = ["default", "cpu"]
    try:
        out += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    return out


def device_tooltip() -> str:
    lines = ["default: ComfyUI's GPU, managed with the other models. cpu: slow but needs no GPU."]
    try:
        for i in range(torch.cuda.device_count()):
            lines.append(f"cuda:{i}: {torch.cuda.get_device_name(i)}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def register_model_folder() -> None:
    folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER),
                                       is_default=True)


def default_model_folder() -> str:
    return folder_paths.get_folder_paths(MODEL_FOLDER)[0]


def detector_choices() -> list[str]:
    files = [f for f in folder_paths.get_filename_list(MODEL_FOLDER)
             if f.lower().endswith((".pth", ".pt", ".safetensors")) and "omnishotcut" in f.lower()]
    if weights.OMNISHOTCUT.filename not in files:
        files.insert(0, weights.OMNISHOTCUT.filename)
    return files


class DetectorHandle:
    """A loaded OmniShotCut behind a ComfyUI ModelPatcher, so ComfyUI can move it on and off
    the GPU alongside every other model."""

    def __init__(self, path: str, device=None):
        self.path = path
        load_device = device or mm.get_torch_device()
        self.model = load_omnishotcut(path, "cpu")
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model, load_device=load_device, offload_device=mm.unet_offload_device())

    def ready(self):
        """Move to the inference device (evicting other models if needed); returns the model."""
        # 8 windows of 100 frames through ResNet-18 at 128x96 plus attention over 1200 tokens
        mm.load_models_gpu([self.patcher], memory_required=768 * 1024 * 1024)
        return self.patcher.model


_DETECTORS: dict[tuple, DetectorHandle] = {}


def get_detector(name: str = weights.OMNISHOTCUT.filename, device: str = "default") -> DetectorHandle:
    path = folder_paths.get_full_path(MODEL_FOLDER, name)
    if path is None:
        if name != weights.OMNISHOTCUT.filename:
            raise FileNotFoundError(f"{name} not found in models/{MODEL_FOLDER}")
        path = weights.ensure(weights.OMNISHOTCUT, default_model_folder())
    key = (os.path.realpath(path), device)
    if key not in _DETECTORS:
        dev = None if device == "default" else torch.device(device)
        _DETECTORS[key] = DetectorHandle(path, dev)
    return _DETECTORS[key]


def video_path(video) -> str:
    """A file path holding exactly the frames of `video`. A plain file is used in place; a trimmed,
    cropped or in-memory video is written to the temp folder first (re-encoded)."""
    src = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    crop = getattr(video, "_VideoFromFile__crop", None)
    start, dur = video.get_active_trim_window() if hasattr(video, "get_active_trim_window") else (0, 0)
    if isinstance(src, str) and os.path.isfile(src) and not crop and not start and not dur:
        return src
    out = os.path.join(folder_paths.get_temp_directory(), "scenecut", f"input_{uuid.uuid4().hex[:12]}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log.warning("SceneCut: the video is trimmed, cropped or in memory; writing a re-encoded copy to "
                "%s for detection. Load the file directly for frame-exact results.", out)
    video.save_to(out)
    return out


def video_from_path(path: str):
    return InputImpl.VideoFromFile(path)


class Progress:
    """ComfyUI progress bar plus the interrupt check, in the shape the core expects."""

    def __init__(self, total: int):
        self.bar = comfy.utils.ProgressBar(max(1, total))

    def __call__(self, done: int, total: int | None = None) -> None:
        self.bar.update_absolute(done, total)

    @staticmethod
    def check_abort() -> None:
        mm.throw_exception_if_processing_interrupted()


# ------------------------------------------------------------------------------------------------
# File names
_UNSAFE = re.compile(r"[^\w.\- ]+", re.UNICODE)


def video_stem(sl) -> str:
    """The source video's file name without extension, safe as a folder name."""
    stem = os.path.splitext(os.path.basename(sl.source.path))[0]
    if stem.startswith("scenecut_images_"):
        stem = "images"
    stem = _UNSAFE.sub("_", stem).strip(" .") or "video"
    return stem[:80]


def expand_prefix(prefix: str, sl) -> str:
    """Replace {video} in a filename prefix with the source video's name."""
    return (prefix or "scenecut/{video}/scenes").replace("{video}", video_stem(sl))


# ------------------------------------------------------------------------------------------------
# IMAGE batches out of and into scene lists
def image_output_input() -> io.Combo.Input:
    return io.Combo.Input("output_images", options=list(IMAGE_MODES), default="off",
                          tooltip=IMAGE_MODE_HELP)


def images_output() -> io.Image.Output:
    return io.Image.Output(display_name="images",
                           tooltip="Frames chosen by output_images (a single black frame when off).")


def scene_images(sl, mode: str) -> torch.Tensor:
    """IMAGE batch (N,H,W,3) float in 0..1 for an output_images choice."""
    key = IMAGE_MODES.get(mode, None)
    if key is None:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    frames = frames_for(sl, key)
    if not frames:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    w, h = sl.source.width, sl.source.height
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001
        available = None
    check_memory(len(frames), w, h, available)
    progress = Progress(max(frames) + 1)
    rgb = read_frames(sl.source.path, frames, w, h, progress=progress, check_abort=progress.check_abort)
    out = torch.empty((len(frames), h, w, 3), dtype=torch.float32)
    for i in range(0, len(frames), 64):                     # convert in chunks: no uint8+float copy of all
        out[i:i + 64] = torch.from_numpy(rgb[i:i + 64]).float().div_(255.0)
    return out


def images_input() -> io.Image.Input:
    return io.Image.Input("images", optional=True,
                          tooltip="Use these frames instead of a scene list, as one single scene "
                                  "(whatever is in them).")


def fps_input() -> io.Float.Input:
    return io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.001, optional=True,
                          force_input=True,
                          tooltip="Frame rate of `images` (not needed for a scene list). Default 24.")


def scenes_from_images(images: torch.Tensor, fps: float | None = None):
    """One-scene list over a lossless temp video of `images`."""
    fps = float(fps) if fps else 24.0
    arr = (images.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    folder = os.path.join(folder_paths.get_temp_directory(), "scenecut")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"scenecut_images_{images_key(arr, fps)}.mkv")
    return single_scene(images_to_video(arr, fps, path))


def resolve_scenes(scenes=None, images=None, fps=None):
    """The scene list a node works on: `images` (as one scene) when connected, else `scenes`."""
    if images is not None:
        if scenes is not None:
            log.info("SceneCut: both scenes and images are connected; using the images")
        return scenes_from_images(images, fps)
    if scenes is None:
        raise ValueError("Connect a scene list (scenes) or frames (images).")
    return scenes
