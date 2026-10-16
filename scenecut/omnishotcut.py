"""OmniShotCut shot-query transformer, written for this pack.

The architecture follows the OmniShotCut paper and checkpoint (UVA Computer Vision Lab, MIT):
a ResNet-18 trunk turns every frame of a 100-frame window into a 3x4 grid of features, a 6+6
layer DETR-style encoder/decoder attends over all 1200 tokens with 3-D sine position codes, and
24 shot queries each predict one shot of the window:

    range head   where the shot ends: a distribution over frame positions 0..window
    intra head   what the shot is: a plain shot or a transition (dissolve, wipe, fade, ...)
    inter head   how it relates to the previous shot: hard cut, sudden jump, transition, ...

Only inference is implemented. The checkpoint's parameter names are the contract; everything
else (padding masks, auxiliary decoder outputs, training augmentation) is gone.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

# Labels as trained. The heads carry one extra trailing "no object" column, unused at inference.
INTRA_LABELS = ("shot", "dissolve", "wipe", "push", "slide", "zoom", "fade", "doorway", "padding")
INTER_LABELS = ("new_start", "hard_cut", "transition_source", "transition", "sudden_jump", "padding")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class OmniShotCutConfig:
    hidden_dim: int = 384
    nheads: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    num_queries: int = 24
    window: int = 100             # frames per inference window
    process_width: int = 128
    process_height: int = 96
    num_intra: int = 10
    num_inter: int = 7
    backbone: str = "resnet18"

    @classmethod
    def from_checkpoint_args(cls, args) -> "OmniShotCutConfig":
        a = vars(args) if not isinstance(args, dict) else args
        if a.get("backbone", "resnet18") != "resnet18" or a.get("dilation", False):
            raise ValueError(f"unsupported OmniShotCut backbone: {a.get('backbone')}")
        if a.get("pre_norm", False) or a.get("position_embedding", "sine") not in ("sine", "v2"):
            raise ValueError("unsupported OmniShotCut transformer variant")
        return cls(
            hidden_dim=int(a["hidden_dim"]), nheads=int(a["nheads"]),
            enc_layers=int(a["enc_layers"]), dec_layers=int(a["dec_layers"]),
            dim_feedforward=int(a["dim_feedforward"]), num_queries=int(a["num_queries"]),
            window=int(a["max_process_window_length"]),
            process_width=int(a["process_width"]), process_height=int(a["process_height"]),
            num_intra=int(a["num_intra_relation_classes"]),
            num_inter=int(a["num_inter_relation_classes"]),
        )

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class _ResNetTrunk(nn.Module):
    """ResNet-18 up to layer4 (stride 32) with frozen batch norm, as the checkpoint stores it."""

    def __init__(self):
        super().__init__()
        import torchvision
        from torchvision.ops.misc import FrozenBatchNorm2d

        r = torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.out_channels = 512

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class _EncoderLayer(nn.Module):
    def __init__(self, d: int, nhead: int, ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead)
        self.linear1 = nn.Linear(d, ff)
        self.linear2 = nn.Linear(ff, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, pos):
        q = x + pos
        x = self.norm1(x + self.self_attn(q, q, x, need_weights=False)[0])
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


class _DecoderLayer(nn.Module):
    def __init__(self, d: int, nhead: int, ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead)
        self.multihead_attn = nn.MultiheadAttention(d, nhead)
        self.linear1 = nn.Linear(d, ff)
        self.linear2 = nn.Linear(ff, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, tgt, memory, pos, query_pos):
        q = tgt + query_pos
        tgt = self.norm1(tgt + self.self_attn(q, q, tgt, need_weights=False)[0])
        tgt = self.norm2(tgt + self.multihead_attn(tgt + query_pos, memory + pos, memory,
                                                   need_weights=False)[0])
        return self.norm3(tgt + self.linear2(F.relu(self.linear1(tgt))))


class _Encoder(nn.Module):
    def __init__(self, cfg: OmniShotCutConfig):
        super().__init__()
        self.layers = nn.ModuleList(_EncoderLayer(cfg.hidden_dim, cfg.nheads, cfg.dim_feedforward)
                                    for _ in range(cfg.enc_layers))


class _Decoder(nn.Module):
    def __init__(self, cfg: OmniShotCutConfig):
        super().__init__()
        self.layers = nn.ModuleList(_DecoderLayer(cfg.hidden_dim, cfg.nheads, cfg.dim_feedforward)
                                    for _ in range(cfg.dec_layers))
        self.norm = nn.LayerNorm(cfg.hidden_dim)


class _Transformer(nn.Module):
    def __init__(self, cfg: OmniShotCutConfig):
        super().__init__()
        self.encoder = _Encoder(cfg)
        self.decoder = _Decoder(cfg)


def sine_position_3d(frames: int, h: int, w: int, dim: int, device, temperature: float = 10000.0):
    """3-D sine/cosine position code over (time, row, column), `dim // 3` channels per axis, each
    axis normalised to (0, 2*pi]. Returns (frames * h * w, dim) in time-major token order."""
    n = dim // 3
    scale = 2 * math.pi
    eps = 1e-6
    z = torch.arange(1, frames + 1, dtype=torch.float32, device=device) / (frames + eps) * scale
    y = torch.arange(1, h + 1, dtype=torch.float32, device=device) / (h + eps) * scale
    x = torch.arange(1, w + 1, dtype=torch.float32, device=device) / (w + eps) * scale
    dim_t = torch.arange(n, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / n)

    def code(v):  # (L,) -> (L, n), sin on even channels and cos on odd, interleaved
        p = v[:, None] / dim_t
        return torch.stack((p[:, 0::2].sin(), p[:, 1::2].cos()), dim=2).flatten(1)

    cz, cy, cx = code(z), code(y), code(x)
    pos = torch.cat((
        cz[:, None, None, :].expand(frames, h, w, n),
        cy[None, :, None, :].expand(frames, h, w, n),
        cx[None, None, :, :].expand(frames, h, w, n),
    ), dim=-1)
    return pos.reshape(frames * h * w, 3 * n)


class OmniShotCut(nn.Module):
    def __init__(self, cfg: OmniShotCutConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.backbone = _ResNetTrunk()
        self.input_proj = nn.Conv2d(self.backbone.out_channels, d, kernel_size=1)
        self.transformer = _Transformer(cfg)
        self.query_embed = nn.Embedding(cfg.num_queries, d)
        self.intra_relation_class_embed = nn.Linear(d, cfg.num_intra)
        self.inter_relation_class_embed = nn.Linear(d, cfg.num_inter)
        self.range_class_embed = nn.Linear(d, cfg.window + 2)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self._pos_cache: dict = {}

    def _pos(self, h: int, w: int, device) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._pos_cache:
            self._pos_cache[key] = sine_position_3d(self.cfg.window, h, w, self.cfg.hidden_dim, device)
        return self._pos_cache[key]

    def forward(self, frames: torch.Tensor) -> dict:
        """frames: uint8 (B, window, H, W, 3) RGB at the process resolution.
        Returns logits: intra (B, Q, num_intra), inter (B, Q, num_inter), range (B, Q, window + 2)."""
        b, t, h, w, _ = frames.shape
        if t != self.cfg.window:
            raise ValueError(f"expected {self.cfg.window} frames per window, got {t}")
        x = frames.reshape(b * t, h, w, 3).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        feat = self.input_proj(self.backbone(x))                     # (B*T, d, h', w')
        _, d, fh, fw = feat.shape
        # tokens in (time, row, column) order, sequence-first for nn.MultiheadAttention
        src = feat.reshape(b, t, d, fh * fw).permute(1, 3, 0, 2).reshape(t * fh * fw, b, d)
        pos = self._pos(fh, fw, feat.device)[:, None, :].to(src.dtype)

        memory = src
        for layer in self.transformer.encoder.layers:
            memory = layer(memory, pos)

        query_pos = self.query_embed.weight[:, None, :].expand(-1, b, -1)
        tgt = torch.zeros_like(query_pos)
        for layer in self.transformer.decoder.layers:
            tgt = layer(tgt, memory, pos, query_pos)
        hs = self.transformer.decoder.norm(tgt).transpose(0, 1)      # (B, Q, d)

        return {
            "intra": self.intra_relation_class_embed(hs),
            "inter": self.inter_relation_class_embed(hs),
            "range": self.range_class_embed(hs),
        }


@contextlib.contextmanager
def strict_fp32():
    """Full-precision convolutions and matmuls for the duration of the block.

    cuDNN runs fp32 convolutions in TF32 by default. On this model that moves logits by up to
    ~0.6 and the result depends on which GPU and kernel ran, so a borderline boundary can come
    and go between machines. The model is small; exact fp32 costs next to nothing."""
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    saved = (cudnn.allow_tf32, matmul.allow_tf32)
    cudnn.allow_tf32 = False
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        cudnn.allow_tf32, matmul.allow_tf32 = saved


def _remap_checkpoint_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("backbone.0.body."):
            k = "backbone." + k[len("backbone.0.body."):]
        if k.endswith("num_batches_tracked"):
            continue
        out[k] = v
    return out


def read_checkpoint(path: str) -> tuple[dict, OmniShotCutConfig]:
    """State dict + config from the official `.pth` (loaded without executing pickle code) or
    from a `.safetensors` file whose metadata carries the config."""
    if path.endswith(".safetensors"):
        import json

        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        if "omnishotcut_config" not in meta:
            raise ValueError(f"{os.path.basename(path)} has no omnishotcut_config metadata")
        cfg = OmniShotCutConfig(**json.loads(meta["omnishotcut_config"]))
        return _remap_checkpoint_keys(load_file(path)), cfg

    # The official checkpoint pickles its training argparse.Namespace next to the weights.
    # weights_only=True refuses anything but tensors and allow-listed types.
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model" not in ckpt or "args" not in ckpt:
        raise ValueError(f"{os.path.basename(path)} is not an OmniShotCut checkpoint")
    return _remap_checkpoint_keys(ckpt["model"]), OmniShotCutConfig.from_checkpoint_args(ckpt["args"])


def load_omnishotcut(path: str, device="cpu") -> OmniShotCut:
    sd, cfg = read_checkpoint(path)
    model = OmniShotCut(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k not in ("mean", "std")]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.eval().requires_grad_(False).to(device)
