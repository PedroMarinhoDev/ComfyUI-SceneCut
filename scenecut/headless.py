"""SceneCut without ComfyUI: load the models from plain files and run a preset end to end.

The nodes load DINOv3 through ComfyUI's own ViT module. Scripts and other pipelines have no
ComfyUI to import, so this loads the same weights file into Hugging Face's implementation
(`transformers`, an optional dependency used only here). Both run the same network on the same
weights, and the decision logic in `validate` is shared.

    from scenecut.headless import Models, run

    models = Models.load("/path/to/ComfyUI/models", device="cuda:0")
    scenes = run("film.mp4", "film", models)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import validate as V
from . import weights
from .omnishotcut import OmniShotCut, load_omnishotcut
from .pipeline import run_preset
from .presets import Preset, get_preset
from .scenes import SceneList

# DINOv3 ViT-L/16, the model the validation thresholds are calibrated for.
DINOV3_VITL16 = {
    "hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24,
    "num_attention_heads": 16, "num_register_tokens": 4, "patch_size": 16, "num_channels": 3,
    "image_size": 224, "layer_norm_eps": 1e-5, "rope_theta": 100.0, "pos_embed_rescale": 2.0,
    "query_bias": True, "key_bias": False, "value_bias": True, "proj_bias": True,
    "mlp_bias": True, "use_gated_mlp": False,
}
PRECISIONS = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load_dino(path: str, device="cuda", precision: str = "fp32") -> V.DinoEmbedder:
    """DINOv3 ViT-L/16 from its `.safetensors` file, as the embedder `validate` expects."""
    from safetensors.torch import load_file
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    device = torch.device(device)
    dtype = PRECISIONS[precision]
    if device.type == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32
    sd = load_file(path)
    if "layer.23.attention.o_proj.bias" not in sd or "layer.24.norm1.weight" in sd \
            or "layer.0.mlp.gate_proj.weight" in sd:
        raise ValueError(f"{os.path.basename(path)} is not DINOv3 ViT-L/16, the model SceneCut's "
                         "validation thresholds are calibrated for")
    model = DINOv3ViTModel(DINOv3ViTConfig(**DINOV3_VITL16))
    if not any(k.startswith("layer.") for k in model.state_dict()):
        # transformers 5 nests the blocks one level down; the file keeps the original names
        sd = {("model." + k if k.startswith("layer.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "mask_token" not in k]
    if missing or unexpected:
        raise ValueError(f"DINOv3 weights do not match: missing {missing[:4]}, unexpected {unexpected[:4]}")
    model = model.eval().requires_grad_(False).to(device, dtype)
    prefix = 1 + DINOV3_VITL16["num_register_tokens"]
    return V.DinoEmbedder(lambda x: model(pixel_values=x).last_hidden_state, prefix, device, dtype,
                          name=f"{os.path.basename(path)}:{precision}")


@dataclass
class Models:
    detector: OmniShotCut
    dino: Optional[V.DinoEmbedder] = None
    sscd: Optional[V.SscdEmbedder] = None

    @classmethod
    def load(cls, models_dir: str, device="cuda", validate: bool = True, use_sscd: bool = True,
             precision: str = "fp32", allow_download: bool = True) -> "Models":
        """Models from a ComfyUI-style models folder (`scenecut/` and `clip_vision/` inside it),
        downloading what is missing."""
        scenecut_dir = os.path.join(models_dir, weights.OMNISHOTCUT.folder)
        detector = load_omnishotcut(weights.ensure(weights.OMNISHOTCUT, scenecut_dir, allow_download),
                                    device)
        dino = sscd = None
        if validate:
            dino_dir = os.path.join(models_dir, weights.DINOV3.folder)
            dino = load_dino(weights.ensure(weights.DINOV3, dino_dir, allow_download), device, precision)
            if use_sscd:
                path = weights.ensure(weights.SSCD, scenecut_dir, allow_download)
                sscd = V.load_sscd(path, device)
                sscd.name = os.path.basename(path)
        return cls(detector, dino, sscd)


def run(path: str, preset: "str | Preset", models: Models, cache: Optional[dict] = None,
        progress: Optional[Callable[[int, int], None]] = None,
        check_abort: Optional[Callable[[], None]] = None) -> SceneList:
    """Detect, refine and (when the models include DINOv3) validate: what SceneCut Detect does."""
    p = get_preset(preset) if isinstance(preset, str) else preset
    return run_preset(models.detector, path, p, dino=models.dino, sscd=models.sscd, cache=cache,
                      progress=progress, check_abort=check_abort)
