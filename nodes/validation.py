"""DINOv3 (+ SSCD) cut validation in ComfyUI: loading, the advanced nodes, shared helpers."""
from __future__ import annotations

import json
import logging
import os

import torch

import comfy.image_encoders.dino3 as dino3
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui

from ..scenecut import validate as V
from ..scenecut import weights
from ..scenecut.refine import merge_ultra_short
from .common import (CATEGORY_ADVANCED, MODEL_FOLDER, Progress, Scenes, device_choices, device_tooltip,
                     image_output_input, images_output, scene_images)

log = logging.getLogger("SceneCut")

Validator = io.Custom("SCENECUT_VALIDATOR")

PRECISIONS = {"fp32": torch.float32, "bf16": torch.bfloat16}
SSCD_OFF = "none"


def _dino_config() -> dict:
    path = os.path.join(os.path.dirname(dino3.__file__), "dino3_large.json")
    with open(path) as f:
        return json.load(f)


def dino_choices() -> list[str]:
    files = [f for f in folder_paths.get_filename_list("clip_vision")
             if "dinov3" in f.lower().replace("_", "").replace("-", "")]
    if weights.DINOV3.filename not in files:
        files.insert(0, weights.DINOV3.filename)
    return files


def sscd_choices() -> list[str]:
    files = [f for f in folder_paths.get_filename_list(MODEL_FOLDER) if "sscd" in f.lower()]
    if weights.SSCD.filename not in files:
        files.insert(0, weights.SSCD.filename)
    return [SSCD_OFF] + files


def _resolve(folder: str, name: str, model: weights.ModelFile) -> str:
    path = folder_paths.get_full_path(folder, name)
    if path is not None:
        return path
    if name != model.filename:
        raise FileNotFoundError(f"{name} not found in models/{folder}")
    return weights.ensure(model, folder_paths.get_folder_paths(folder)[0])


class ValidatorHandle:
    """DINOv3 ViT-L/16 behind a ComfyUI ModelPatcher, plus an optional SSCD TorchScript model.

    DINOv3 runs in fp32 (or bf16 on request): its activations overflow fp16, which is what ComfyUI
    would otherwise pick for an image encoder on many GPUs, and the result is NaN."""

    def __init__(self, dino_name: str, sscd_name: str = SSCD_OFF, precision: str = "fp32",
                 device: str = "default"):
        self.load_device = mm.get_torch_device() if device == "default" else torch.device(device)
        self.dtype = PRECISIONS[precision]
        if self.load_device.type == "cpu" and self.dtype == torch.bfloat16:
            self.dtype = torch.float32
        path = _resolve("clip_vision", dino_name, weights.DINOV3)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        if "layer.23.attention.o_proj.bias" not in sd or "layer.24.norm1.weight" in sd \
                or "layer.0.mlp.gate_proj.weight" in sd:
            raise ValueError(f"{os.path.basename(path)} is not DINOv3 ViT-L/16, the model SceneCut's "
                             "validation thresholds are calibrated for")
        offload = mm.unet_offload_device()
        model = dino3.DINOv3ViTModel(_dino_config(), self.dtype, offload, comfy.ops.manual_cast)
        missing, _ = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "mask_token" not in k]
        if missing:
            raise ValueError(f"DINOv3 weights incomplete: {missing[:4]}")
        self.model = model.eval().requires_grad_(False)
        self.patcher = comfy.model_patcher.CoreModelPatcher(self.model, load_device=self.load_device,
                                                            offload_device=offload)
        self.name = f"{os.path.basename(path)}:{precision}"
        self.sscd_path = None if sscd_name == SSCD_OFF else _resolve(MODEL_FOLDER, sscd_name, weights.SSCD)
        self._sscd = None

    def dino(self) -> V.DinoEmbedder:
        # batch of 48 frames at 224px through ViT-L: activations and attention maps
        mm.load_models_gpu([self.patcher], memory_required=2 * 1024 ** 3)
        model = self.patcher.model
        return V.DinoEmbedder(lambda x: model(x)[0], model.num_prefix_tokens, self.load_device,
                              self.dtype, name=self.name)

    def sscd(self):
        if self.sscd_path is None:
            return None
        if self._sscd is None:
            self._sscd = V.load_sscd(self.sscd_path, self.load_device)
            self._sscd.name = os.path.basename(self.sscd_path)
        return self._sscd


_HANDLES: dict[tuple, ValidatorHandle] = {}
_FEATURE_CACHE: dict = {}       # validate.run() cache: threshold changes reuse extracted features
_FEATURE_CACHE_VIDEOS = 4


def get_validator(dino_name: str = weights.DINOV3.filename, sscd_name: str = weights.SSCD.filename,
                  precision: str = "fp32", device: str = "default") -> ValidatorHandle:
    key = (dino_name, sscd_name, precision, device)
    if key not in _HANDLES:
        _HANDLES[key] = ValidatorHandle(dino_name, sscd_name, precision, device)
    return _HANDLES[key]


def run_validation(scenes, handle: ValidatorHandle, cfg: V.ValidatorConfig):
    while len(_FEATURE_CACHE) > _FEATURE_CACHE_VIDEOS:
        _FEATURE_CACHE.pop(next(iter(_FEATURE_CACHE)))
    progress = Progress(scenes.frame_count)
    sscd = handle.sscd() if cfg.use_sscd else None
    return V.run(scenes, handle.dino(), cfg, sscd, cache=_FEATURE_CACHE,
                 progress=lambda done, total: progress(done, total), check_abort=progress.check_abort)


class SceneCutValidatorLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutValidatorLoader",
            display_name="SceneCut Validator Loader",
            category=CATEGORY_ADVANCED,
            description="DINOv3 ViT-L/16 (models/clip_vision) and optionally SSCD (models/scenecut) "
                        "for cut validation. Both download on first use; DINOv3 from Meta's gated "
                        "repo if you are logged in to Hugging Face, else from a byte-identical mirror.",
            inputs=[
                io.Combo.Input("dinov3", options=dino_choices()),
                io.Combo.Input("sscd", options=sscd_choices(), default=weights.SSCD.filename,
                               tooltip="Copy-detection descriptor used to break close calls and "
                                       "judge flashes. 'none' validates with DINOv3 alone."),
                io.Combo.Input("precision", options=list(PRECISIONS), default="fp32",
                               tooltip="fp32 matches the calibration. bf16 is faster with tiny "
                                       "differences in the scores. (fp16 is not offered: DINOv3 "
                                       "overflows in it.)"),
                io.Combo.Input("device", options=device_choices(), default="default",
                               tooltip=device_tooltip()),
            ],
            outputs=[Validator.Output(display_name="validator")],
        )

    @classmethod
    def execute(cls, dinov3, sscd, precision, device) -> io.NodeOutput:
        return io.NodeOutput(get_validator(dinov3, sscd, precision, device))


class SceneCutValidate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutValidate",
            display_name="SceneCut Validate (Advanced)",
            category=CATEGORY_ADVANCED,
            search_aliases=["dino validation", "false cut filter", "flash filter"],
            description="Check every cut with DINOv3 and merge the false ones (flashes, fast motion, "
                        "a continuous shot split in two). Threshold and guard changes rerun "
                        "instantly; window and flash sensitivity re-read the video.",
            inputs=[
                Scenes.Input("scenes"),
                Validator.Input("validator", optional=True,
                                tooltip="From SceneCut Validator Loader. Default: DINOv3 fp32 + SSCD."),
                io.Int.Input("window", default=5, min=2, max=12,
                             tooltip="Frames compared on each side of a cut (plus a 3x wider window)."),
                io.Float.Input("flash_sensitivity", default=15.0, min=5.0, max=80.0, step=0.5,
                               tooltip="Brightness jump (0-255) that counts as a flash. Smaller "
                                       "treats more light changes as flashes rather than cuts."),
                io.Boolean.Input("adaptive_threshold", default=True,
                                 tooltip="Pick the same-shot threshold from this video's own "
                                         "scores. Off: use `threshold`."),
                io.Float.Input("threshold", default=0.80, min=0.5, max=0.99, step=0.01,
                               tooltip="Same-shot similarity above which a cut is merged (when "
                                       "adaptive_threshold is off). Higher keeps more cuts."),
                io.Float.Input("dense_weight", default=0.45, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="Weight of the spatial (patch) similarity against the "
                                       "global one."),
                io.Float.Input("short_guard_seconds", default=0.30, min=0.0, max=2.0, step=0.05,
                               advanced=True,
                               tooltip="Cuts beside shots this short are only merged on flash "
                                       "evidence."),
                io.Boolean.Input("flash_cleanup", default=True, advanced=True,
                                 tooltip="Merge slivers of a few frames between flash boundaries."),
                io.Float.Input("ultra_short_seconds", default=0.05, min=0.0, max=1.0, step=0.01,
                               advanced=True, tooltip="Final clean-up of fragments. 0 disables."),
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
    def execute(cls, scenes, window, flash_sensitivity, adaptive_threshold, threshold, dense_weight=0.45,
                short_guard_seconds=0.30, flash_cleanup=True, ultra_short_seconds=0.05,
                output_images="off", validator=None) -> io.NodeOutput:
        handle = validator or get_validator()
        cfg = V.ValidatorConfig(window=window, flash_sensitivity=flash_sensitivity,
                                use_sscd=handle.sscd_path is not None,
                                threshold=None if adaptive_threshold else threshold,
                                dense_weight=dense_weight, short_guard_seconds=short_guard_seconds,
                                tiny_flash_pass=flash_cleanup)
        sl = run_validation(scenes, handle, cfg)
        if ultra_short_seconds:
            sl = merge_ultra_short(sl, ultra_short_seconds)
        summary = sl.summary()
        return io.NodeOutput(sl, len(sl.shots), summary, scene_images(sl, output_images),
                             ui=ui.PreviewText(summary))
