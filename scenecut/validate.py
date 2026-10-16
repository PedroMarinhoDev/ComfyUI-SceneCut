"""Cut validation: decide which detected boundaries are real cuts.

A detector proposes boundaries; some are not cuts at all (a camera flash, lightning, a strobe, a
whip pan, a continuous shot the detector split). Each boundary is scored by comparing what comes
before it with what comes after it:

  DINOv3     global (CLS) and spatial (7x7 pooled patch) embeddings of the frames on each side,
             their similarity across the boundary, and how abruptly they change at it (novelty)
  flash      a brightness spike that returns to its baseline marks a flash, not a cut; similarities
             are then recomputed without the flashed frames
  SSCD       an optional copy-detection descriptor of a median composite of steady frames on each
             side, a tie-breaker that ignores lighting
  histogram  hue/saturation overlap a few frames either side

A per-video threshold (Otsu on the observed scores, clamped) separates "same shot" from "cut", and a
fixed cascade of rules decides each boundary. The design follows SceneCutGUI's validator (enoky,
MIT); this is a separate implementation that decodes frames exactly and streams, so memory stays
flat however long the video is.

Feature extraction reads the video once: frames near each cut are decoded at 288x288 (SSCD's input
size) and downscaled on the GPU to 224x224 for DINOv3. A cut's features are computed as soon as the
decoder has passed it, and its frames are dropped.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .boundary import best_split
from .media import iter_frames_at, prefetch
from .omnishotcut import strict_fp32
from .scenes import SceneList


# --------------------------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidatorConfig:
    # --- exposed: features (changing these re-reads the video) ---
    window: int = 5                    # frames compared on each side of a cut
    flash_sensitivity: float = 15.0    # luma jump (0-255) that counts as a flash; smaller = more flashes
    use_sscd: bool = True
    # --- exposed: decisions (instant to change) ---
    threshold: Optional[float] = None  # same-shot similarity threshold; None = per video (adaptive)
    dense_weight: float = 0.45         # spatial vs global similarity in the blended score
    short_guard_seconds: float = 0.30  # cuts next to shots this short need flash evidence to merge
    tiny_flash_pass: bool = True       # merge sub-0.12 s slivers between flash boundaries

    # --- internal constants (calibrated together; change with care) ---
    max_large_window: int = 24
    dense_grid: int = 7
    novelty_cls_weight: float = 0.6
    sim_small_weight: float = 0.65     # small vs large window similarity
    # adaptive threshold
    default_threshold: float = 0.88
    thr_min_samples: int = 10
    thr_flat_std: float = 0.01
    thr_flat_bump: float = 0.01
    thr_flat_lo: float = 0.86
    thr_flat_hi: float = 0.93
    thr_lo: float = 0.80
    thr_hi: float = 0.94
    thr_pct_lo: int = 75
    thr_pct_hi: int = 90
    # hard keep / merge
    adj_strong_cut: float = 0.68
    hard_keep_novelty: float = 0.32
    hard_keep_adj_dense: float = 0.72
    merge_pixel_min: float = 0.14
    merge_adj_min: float = 0.78
    # flash detection
    flash_scan: int = 12
    flash_center: int = 4
    flash_base_gap: int = 5
    flash_min_dur: int = 1
    flash_max_dur: int = 8
    flash_amp_floor: float = 10.0
    flash_amp_scale: float = 0.6
    flash_spike_z: float = 2.8
    flash_drift_floor: float = 12.0
    flash_drift_amp_ratio: float = 0.60
    flash_w_amp: float = 0.52
    flash_w_z: float = 0.15
    flash_w_dur: float = 0.18
    flash_w_return: float = 0.15
    flash_amp_span: float = 1.8
    flash_z_offset: float = 3.0
    flash_z_span: float = 3.5
    flash_off_boundary_penalty: float = 0.5
    flash_is_flash_conf: float = 0.55
    flash_afterglow: float = 0.40
    flash_stable_tol_floor: float = 12.0
    flash_stable_tol_frac: float = 0.35
    flash_stable_tol_relax: float = 1.6
    flash_move_margin: float = 0.10    # similarity evidence needed to move a cut off the detector's frame
    # flash decisions
    flash_conf_min: float = 0.55
    flash_conf_min_sscd: float = 0.45
    flash_same_shot_off: float = 0.04
    flash_dense_same: float = 0.78
    flash_sscd_same: float = 0.72
    flash_strong_novelty: float = 0.28
    flash_strong_adj: float = 0.70
    flash_strong_adj_dense: float = 0.74
    # SSCD
    sscd_k: int = 5
    sscd_min_k: int = 3
    sscd_size: int = 288
    sscd_req_at_max_sens: float = 0.78
    sscd_req_sens_slope: float = 0.0015
    sscd_req_motion_w: float = 0.05
    sscd_req_floor: float = 0.62
    sscd_req_ceil: float = 0.84
    sscd_sens_lo: float = 15.0
    sscd_sens_hi: float = 80.0
    ambiguous_band: float = 0.10
    sscd_cut_margin: float = 0.10
    # short-shot guard and tiny-flash pass
    guard_flash_conf_min: float = 0.40
    guard_flash_conf_min_sscd: float = 0.35
    guard_flash_like_conf: float = 0.35
    guard_same_shot_off: float = 0.05
    tiny_guard_seconds: float = 0.12
    tiny_ultra_short_seconds: float = 0.04
    tiny_flash_conf: float = 0.35
    tiny_score_flash_w: float = 1.2
    # a cut the detector says ends a transition: also compare with frames from before it
    lookback_seconds: float = 1.2
    lookback_frames: int = 8
    lookback_cut: float = 0.60         # below this the scene before the transition is a different one
    # throughput
    batch_size: int = 48

    def feature_key(self) -> tuple:
        """Everything that changes extracted features (as opposed to decisions)."""
        return (self.window, float(self.flash_sensitivity), bool(self.use_sscd), float(self.lookback_seconds),
                int(self.lookback_frames))

    def window_geometry(self) -> tuple[int, int, int]:
        """(gap, small window, large window) in frames."""
        gap = max(2, max(3, self.window // 2))
        small = max(2, int(self.window))
        large = int(min(max(small * 3, small + 2), self.max_large_window))
        return gap, small, large

    def reach(self) -> int:
        """How far from a cut its features read: large window plus the gap."""
        gap, _, large = self.window_geometry()
        return max(gap + large, self.flash_scan + 1)


# --------------------------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEmbedder:
    """Wraps a DINOv3 forward pass: (B,3,224,224) normalised -> token sequence (B, 1+R+P, D).

    Returns per frame an L2-normalised CLS vector and the patch tokens mean-pooled to a 7x7 grid
    (each cell L2-normalised); the grid is kept in fp16 to bound memory."""

    def __init__(self, forward: Callable, num_prefix_tokens: int, device, dtype=torch.float32,
                 size: int = 224, name: str = "dinov3"):
        self.forward = forward
        self.num_prefix = int(num_prefix_tokens)
        self.device = torch.device(device)
        self.dtype = dtype
        self.size = size
        self.name = name
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, frames224: torch.Tensor, grid: int = 7):
        """frames224: float (B,3,224,224) in 0..255 on self.device."""
        x = ((frames224 / 255.0) - self.mean) / self.std
        with strict_fp32():
            seq = self.forward(x.to(self.dtype)).float()
        cls = F.normalize(seq[:, 0], dim=-1)
        patches = seq[:, self.num_prefix:]
        b, p, d = patches.shape
        side = int(round(math.sqrt(p)))
        spatial = patches.transpose(1, 2).reshape(b, d, side, side)
        pooled = F.adaptive_avg_pool2d(spatial, (grid, grid)).flatten(2).transpose(1, 2)
        dense = F.normalize(pooled, dim=-1)
        return cls.cpu(), dense.to(torch.float16).cpu()


class SscdEmbedder:
    """SSCD copy-detection descriptor (TorchScript) of one RGB image."""

    def __init__(self, module, device, name: str = "sscd"):
        self.module = module.eval()
        self.device = torch.device(device)
        self.name = name
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(rgb).to(self.device).float().div_(255.0).permute(2, 0, 1)[None]
        x = (x - self.mean) / self.std
        with strict_fp32():
            out = self.module(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out.flatten().float(), dim=0).cpu()


def load_sscd(path: str, device) -> SscdEmbedder:
    return SscdEmbedder(torch.jit.load(path, map_location=device), device)


# --------------------------------------------------------------------------------------------- #
# Per-frame signals
# --------------------------------------------------------------------------------------------- #
@torch.inference_mode()
def frame_signals(frames224: torch.Tensor):
    """luma (Rec.709 mean), 99.5th percentile of HSV value, and a normalised 16x16 hue/saturation
    histogram (OpenCV's 8-bit HSV convention: H in [0,180), S and V in [0,255]) per frame."""
    r, g, b = frames224[:, 0], frames224[:, 1], frames224[:, 2]
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).mean(dim=(1, 2))
    v = frames224.max(dim=1).values
    mn = frames224.min(dim=1).values
    vhi = torch.quantile(v.flatten(1), 0.995, dim=1)
    delta = v - mn
    s = torch.where(v > 0, 255.0 * delta / v.clamp(min=1e-6), torch.zeros_like(v))
    safe = delta.clamp(min=1e-6)
    h = torch.where(v == r, 60.0 * (g - b) / safe,
                    torch.where(v == g, 120.0 + 60.0 * (b - r) / safe, 240.0 + 60.0 * (r - g) / safe))
    h = torch.where(delta > 0, h, torch.zeros_like(h))
    h = torch.remainder(h, 360.0) / 2.0
    hb = (h * 16.0 / 180.0).long().clamp(0, 15)
    sb = (s * 16.0 / 256.0).long().clamp(0, 15)
    idx = (hb * 16 + sb).flatten(1)
    hist = torch.zeros(idx.shape[0], 256, device=idx.device)
    hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=hist.dtype))
    hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return luma.cpu().numpy(), vhi.cpu().numpy(), hist.cpu().numpy()


class _Frames:
    """Per-frame features for the frames currently in reach of a cut."""

    def __init__(self):
        self.cls: dict[int, torch.Tensor] = {}
        self.dense: dict[int, torch.Tensor] = {}
        self.luma: dict[int, float] = {}
        self.vhi: dict[int, float] = {}
        self.hist: dict[int, np.ndarray] = {}
        self.rgb: dict[int, np.ndarray] = {}      # SSCD-size frames near cuts

    def flash_signal(self, i: int) -> Optional[float]:
        l, v = self.luma.get(i), self.vhi.get(i)
        if l is None and v is None:
            return None
        return float(max(x for x in (l, v) if x is not None))

    def drop_before(self, frame: int) -> None:
        for d in (self.cls, self.dense, self.luma, self.vhi, self.hist, self.rgb):
            for k in [k for k in d if k < frame]:
                del d[k]


# --------------------------------------------------------------------------------------------- #
# Per-cut features
# --------------------------------------------------------------------------------------------- #
def _lower_median(x: torch.Tensor) -> float:
    return float(x.flatten().median().item())


def _cls_median(fr: _Frames, a: list[int], b: list[int]) -> Optional[float]:
    va = [fr.cls[i] for i in a if i in fr.cls]
    vb = [fr.cls[i] for i in b if i in fr.cls]
    if not va or not vb:
        return None
    return _lower_median(torch.stack(va) @ torch.stack(vb).T)


def _mean_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bidirectional mean of best-match cosine between patch sets; a: (..., P, D), b: (..., P, D)."""
    sim = a @ b.transpose(-1, -2)
    return 0.5 * (sim.max(dim=-1).values.mean(dim=-1) + sim.max(dim=-2).values.mean(dim=-1))


def _dense_median(fr: _Frames, a: list[int], b: list[int], device) -> Optional[float]:
    da = [fr.dense[i] for i in a if i in fr.dense]
    db = [fr.dense[i] for i in b if i in fr.dense]
    if not da or not db:
        return None
    A = torch.stack(da).to(device).float()
    B = torch.stack(db).to(device).float()
    with strict_fp32():
        s = _mean_max(A[:, None], B[None, :]).flatten().cpu().numpy()
    s = s[~np.isnan(s)]
    return float(np.median(s)) if s.size else None


def _windows(cut: int, w: int, gap: int, total: int) -> tuple[list[int], list[int]]:
    pre_end = max(0, cut - gap)
    post_start = min(total, cut + gap)
    return (list(range(max(0, pre_end - w), pre_end)),
            list(range(post_start, min(total, post_start + w))))


def _longest_run(idxs: list[int]) -> list[int]:
    if not idxs:
        return []
    idxs = sorted(set(idxs))
    best = cur = [idxs[0]]
    for x in idxs[1:]:
        cur = cur + [x] if x == cur[-1] + 1 else [x]
        if len(cur) > len(best):
            best = cur
    return best


def flash_event(fr: _Frames, cut: int, total: int, cfg: ValidatorConfig) -> dict:
    """Is there a brightness spike at the cut that returns to its baseline? Confidence 0..1."""
    clamp = lambda i: int(max(0, min(total - 1, i)))  # noqa: E731
    sig = fr.flash_signal
    pre = [v for v in (sig(clamp(cut + k)) for k in range(-cfg.flash_scan, -cfg.flash_base_gap)) if v is not None]
    post = [v for v in (sig(clamp(cut + k)) for k in range(cfg.flash_base_gap, cfg.flash_scan + 1)) if v is not None]
    none = {"is_flash": False, "conf": 0.0, "amp": 0.0, "dur": 0, "baseline": None,
            "spike_frames": [], "boundary_hit": False}
    if len(pre) + len(post) < 6:
        return none
    b_pre = float(np.median(pre)) if pre else float(np.median(post))
    b_post = float(np.median(post)) if post else float(np.median(pre))
    baseline = float(np.median([b_pre, b_post]))
    drift = abs(b_pre - b_post)
    scale = max(1.0, 1.4826 * float(np.median(np.abs(np.asarray(pre + post, np.float32) - baseline))))

    center = [(clamp(cut + k), sig(clamp(cut + k))) for k in range(-cfg.flash_center, cfg.flash_center + 1)]
    center = [(i, v) for i, v in center if v is not None]
    if not center:
        return {**none, "baseline": baseline}
    bright = [(i, v - baseline) for i, v in center]
    dark = [(i, baseline - v) for i, v in center]
    kind = "bright" if max(d for _, d in bright) >= max(d for _, d in dark) else "dark"
    deltas = bright if kind == "bright" else dark
    amp = float(max(d for _, d in deltas))
    amp_thr = max(cfg.flash_amp_floor, cfg.flash_amp_scale * cfg.flash_sensitivity)
    z = amp / scale

    run = _longest_run([i for i, d in deltas if d >= amp_thr and d / scale >= cfg.flash_spike_z])
    dur = len(run)
    hit = any(abs(i - cut) <= 1 for i in run)

    amp_score = float(np.clip((amp - amp_thr) / max(1e-6, cfg.flash_amp_span * amp_thr), 0, 1))
    z_score = float(np.clip((z - cfg.flash_z_offset) / cfg.flash_z_span, 0, 1))
    dur_score = (0.0 if dur <= 0 else 1.0 if dur <= 3 else 0.85 if dur <= 6
                 else 0.70 if dur <= cfg.flash_max_dur else 0.0)
    if amp <= 1e-3:
        ret_score = 0.0
    else:
        ratio = drift / amp
        ret_score = 1.0 if ratio <= 0.30 else 0.7 if ratio <= 0.60 else 0.25 if ratio <= 1.00 else 0.0
    conf = float(np.clip(cfg.flash_w_amp * amp_score + cfg.flash_w_z * z_score
                         + cfg.flash_w_dur * dur_score + cfg.flash_w_return * ret_score, 0, 1))
    if not hit:
        conf *= cfg.flash_off_boundary_penalty
    is_flash = conf >= cfg.flash_is_flash_conf and hit and cfg.flash_min_dur <= dur <= cfg.flash_max_dur
    return {"is_flash": bool(is_flash), "conf": conf, "kind": kind if is_flash else None,
            "amp": amp, "dur": dur, "drift": drift, "baseline": baseline,
            "run_frames": run,
            "spike_frames": sorted({clamp(i + k) for i in run for k in (-1, 0, 1)}),
            "boundary_hit": hit}


def _without_flash(fr: _Frames, idxs: list[int], fe: dict, cfg: ValidatorConfig) -> list[int]:
    if not fe.get("is_flash"):
        return list(idxs)
    spikes = set(fe["spike_frames"])
    base, amp, kind = fe["baseline"], fe["amp"], fe["kind"]
    out = []
    for i in idxs:
        if i in spikes:
            continue
        s = fr.flash_signal(i)
        if s is not None and base is not None and amp > 1e-6:
            if kind == "bright" and s > base + cfg.flash_afterglow * amp:
                continue
            if kind == "dark" and s < base - cfg.flash_afterglow * amp:
                continue
        out.append(i)
    return out


def _steady_near(fr: _Frames, cut: int, total: int, side: str, fe: dict, cfg: ValidatorConfig) -> list[int]:
    """Up to sscd_k frames next to the cut whose brightness sits at the flash baseline."""
    cands = [cut - k for k in range(1, 13)] if side == "pre" else [cut + k for k in range(0, 12)]
    cands = [int(max(0, min(total - 1, i))) for i in cands]
    spikes = set(fe.get("spike_frames") or [])
    base = fe.get("baseline")
    if base is None:
        return [i for i in cands if i not in spikes][: cfg.sscd_k]

    def gather(tol):
        out = []
        for i in cands:
            if i in spikes:
                continue
            v = fr.flash_signal(i)
            if v is not None and abs(v - base) <= tol:
                out.append(i)
                if len(out) >= cfg.sscd_k:
                    break
        return out

    tol = max(cfg.flash_stable_tol_floor, cfg.flash_stable_tol_frac * abs(fe.get("amp", 0.0)))
    chosen = gather(tol)
    return chosen if len(chosen) >= cfg.sscd_min_k else gather(tol * cfg.flash_stable_tol_relax)


def _sscd_similarity(fr: _Frames, cut: int, total: int, fe: dict, sscd, cfg) -> Optional[float]:
    pre = [fr.rgb[i] for i in _steady_near(fr, cut, total, "pre", fe, cfg) if i in fr.rgb]
    post = [fr.rgb[i] for i in _steady_near(fr, cut, total, "post", fe, cfg) if i in fr.rgb]
    if len(pre) < cfg.sscd_min_k or len(post) < cfg.sscd_min_k:
        return None
    comp = lambda fs: np.clip(np.median(np.stack(fs).astype(np.float32), axis=0), 0, 255).astype(np.uint8)  # noqa: E731
    s = float((sscd(comp(pre)) * sscd(comp(post))).sum())
    return None if math.isnan(s) else s


def _flash_side_cut(fr: _Frames, cut: int, total: int, fe: dict, cfg) -> Optional[int]:
    """Where a kept cut beside a flash belongs. Each flash frame, and the frames either side of the
    cut, goes with the shot it looks like (DINOv3 similarity to steady frames before and after),
    and the cut goes between the two groups. The detector's frame stays unless the evidence
    clearly says otherwise: a flash often starts or ends exactly on a real cut."""
    run = fe.get("run_frames") or []
    if not fe.get("is_flash") or not run:
        return None
    lo = max(0, min(min(run), cut - 1) - 1)
    hi = min(total - 1, max(max(run), cut) + 1)
    span = list(range(lo, hi + 1))
    pre = [fr.cls[i] for i in _steady_near(fr, cut, total, "pre", fe, cfg) if i in fr.cls and i < lo]
    post = [fr.cls[i] for i in _steady_near(fr, cut, total, "post", fe, cfg) if i in fr.cls and i > hi]
    if not pre or not post or any(i not in fr.cls for i in span):
        return None
    a = F.normalize(torch.stack(pre).mean(0), dim=0)
    b = F.normalize(torch.stack(post).mean(0), dim=0)
    to_a = [float(fr.cls[i] @ a) for i in span]
    to_b = [float(fr.cls[i] @ b) for i in span]
    new, _ = best_split(lo, to_a, to_b, prefer=cut, margin=cfg.flash_move_margin)
    return new


def cut_features(fr: _Frames, cut: int, total: int, cfg: ValidatorConfig, device,
                 sscd=None, lookback: Optional[tuple[int, int]] = None) -> Optional[dict]:
    """Everything the decision needs about one boundary, or None without enough frames. `lookback`
    (first, end) adds `s_lookback`: how alike the frames there and the frames after the cut are."""
    gap, ws, wl = cfg.window_geometry()
    pre_s, post_s = _windows(cut, ws, gap, total)
    pre_l, post_l = _windows(cut, wl, gap, total)
    if sum(i in fr.cls for i in pre_s) < 2 or sum(i in fr.cls for i in post_s) < 2:
        return None
    k = cfg.sim_small_weight

    def blend2(small, large):
        if small is None:
            return large
        return small if large is None else k * small + (1 - k) * large

    def sims(ps, qs, pl, ql):
        s_cls = blend2(_cls_median(fr, ps, qs),
                       _cls_median(fr, pl, ql) if sum(i in fr.cls for i in pl) >= 2
                       and sum(i in fr.cls for i in ql) >= 2 else None)
        d_small = _dense_median(fr, ps, qs, device)
        d_large = _dense_median(fr, pl, ql, device) if len(pl) >= 2 and len(ql) >= 2 else None
        return s_cls, blend2(d_small, d_large)

    s_cls, s_dense = sims(pre_s, post_s, pre_l, post_l)

    s_adj = s_adj_dense = None
    if cut - 1 in fr.cls and cut in fr.cls:
        s_adj = float((fr.cls[cut - 1] * fr.cls[cut]).sum())
    if cut - 1 in fr.dense and cut in fr.dense:
        with strict_fp32():
            v = float(_mean_max(fr.dense[cut - 1].to(device).float(), fr.dense[cut].to(device).float()))
        s_adj_dense = None if math.isnan(v) else v
    nov_cls = 1.0 - s_adj if s_adj is not None else 0.0
    if s_adj_dense is None:
        novelty = nov_cls
    else:
        novelty = cfg.novelty_cls_weight * nov_cls + (1 - cfg.novelty_cls_weight) * (1.0 - s_adj_dense)

    def continuity(idxs):
        v = [fr.cls[i] for i in idxs if i in fr.cls]
        if len(v) < 2:
            return 1.0
        m = torch.stack(v)
        return float((m[:-1] * m[1:]).sum(dim=1).mean())

    pre_cont, post_cont = continuity(pre_s), continuity(post_s)
    motion = max(0.0, 1.0 - min(pre_cont, post_cont))

    fe = flash_event(fr, cut, total, cfg)
    s_cls_st, s_dense_st = s_cls, s_dense
    if fe["is_flash"]:
        ps, qs = _without_flash(fr, pre_s, fe, cfg), _without_flash(fr, post_s, fe, cfg)
        if sum(i in fr.cls for i in ps) >= 2 and sum(i in fr.cls for i in qs) >= 2:
            pl, ql = _without_flash(fr, pre_l, fe, cfg), _without_flash(fr, post_l, fe, cfg)
            small = _cls_median(fr, ps, qs)
            large = (_cls_median(fr, pl, ql) if sum(i in fr.cls for i in pl) >= 2
                     and sum(i in fr.cls for i in ql) >= 2 else None)
            s_cls_st = blend2(small, large)
            s_dense_st = _dense_median(fr, ps, qs, device)

    def hist_near(offsets):
        for o in offsets:
            if 0 <= cut + o < total and cut + o in fr.hist:
                return fr.hist[cut + o]
        return None

    h1, h2 = hist_near((-4, -5, -3, -6)), hist_near((4, 5, 3, 6))
    pixel_sim = float(np.minimum(h1, h2).sum()) if h1 is not None and h2 is not None else None

    return {
        "cut": int(cut),
        "s_cls": float(s_cls), "s_dense": None if s_dense is None else float(s_dense),
        "s_cls_stable": float(s_cls_st), "s_dense_stable": None if s_dense_st is None else float(s_dense_st),
        "s_adj": s_adj, "s_adj_dense": s_adj_dense, "novelty": float(novelty),
        "pre_cont": pre_cont, "post_cont": post_cont, "motion": float(motion),
        "pixel_sim": pixel_sim, "flash": fe,
        "sscd": _sscd_similarity(fr, cut, total, fe, sscd, cfg) if sscd is not None else None,
        "flash_cut": _flash_side_cut(fr, cut, total, fe, cfg),
        "s_lookback": _lookback_similarity(fr, lookback, post_s, cfg, device) if lookback else None,
    }


def _lookback_similarity(fr: _Frames, lookback: tuple[int, int], post: list[int], cfg: ValidatorConfig,
                         device) -> Optional[float]:
    idxs = [i for i in range(*lookback) if i in fr.cls]
    if len(idxs) < 2:
        return None
    s_cls, s_dense = _cls_median(fr, idxs, post), _dense_median(fr, idxs, post, device)
    if s_cls is None:
        return None
    return float(s_cls if s_dense is None else (1 - cfg.dense_weight) * s_cls + cfg.dense_weight * s_dense)


# --------------------------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------------------------- #
def blended(f: dict, cfg: ValidatorConfig) -> float:
    """Same-shot score: global and spatial similarity, flash frames excluded when there is a flash."""
    s_cls = f["s_cls_stable"] if f["flash"]["is_flash"] else f["s_cls"]
    s_dense = f["s_dense_stable"] if f["flash"]["is_flash"] else f["s_dense"]
    if s_dense is None:
        return float(s_cls)
    return float((1 - cfg.dense_weight) * s_cls + cfg.dense_weight * s_dense)


def _otsu(scores: np.ndarray, bins: int = 128, fallback: float = 0.88) -> float:
    hist, edges = np.histogram(np.clip(scores, 0, 1), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    w1, w2 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    centers = (edges[:-1] + edges[1:]) / 2.0
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-6)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1e-6))[::-1]
    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centers[int(np.nanargmax(between))]) if between.size else fallback


def adaptive_threshold(scores: list[float], cfg: ValidatorConfig) -> float:
    """Same-shot threshold from this video's own score distribution."""
    if len(scores) < cfg.thr_min_samples:
        return cfg.default_threshold
    s = np.asarray(scores, np.float32)
    if float(np.std(s)) < cfg.thr_flat_std:
        return float(np.clip(float(np.mean(s)) + cfg.thr_flat_bump, cfg.thr_flat_lo, cfg.thr_flat_hi))
    t = _otsu(s, fallback=cfg.default_threshold)
    t = max(float(np.percentile(s, cfg.thr_pct_lo)), min(t, float(np.percentile(s, cfg.thr_pct_hi))))
    return float(np.clip(t, cfg.thr_lo, cfg.thr_hi))


def _sscd_required(motion: float, cfg: ValidatorConfig) -> float:
    sens = max(cfg.sscd_sens_lo, min(cfg.sscd_sens_hi, cfg.flash_sensitivity))
    base = cfg.sscd_req_at_max_sens - cfg.sscd_req_sens_slope * (cfg.sscd_sens_hi - sens)
    base -= cfg.sscd_req_motion_w * max(0.0, min(1.0, motion))
    return float(max(cfg.sscd_req_floor, min(cfg.sscd_req_ceil, base)))


def sscd_used(f: dict, thr: float, cfg: ValidatorConfig) -> Optional[float]:
    """SSCD only speaks for flashes and for scores near the threshold."""
    if f.get("sscd") is None:
        return None
    if f["flash"]["is_flash"] or abs(blended(f, cfg) - thr) <= cfg.ambiguous_band:
        return f["sscd"]
    return None


def decide(f: dict, thr: float, cfg: ValidatorConfig) -> tuple[bool, str]:
    """(keep, reason). Order: flash rules, clear discontinuity, same shot, SSCD tie-break, keep."""
    s_blend = blended(f, cfg)
    s_dense, s_adj, s_adj_dense = f["s_dense"], f["s_adj"], f["s_adj_dense"]
    s_sscd = sscd_used(f, thr, cfg)
    nov = f["novelty"]
    fe = f["flash"]

    if fe["is_flash"]:
        if fe["conf"] < (cfg.flash_conf_min_sscd if s_sscd is not None else cfg.flash_conf_min):
            return True, "keep_flash_low_conf"
        same = (s_blend >= thr - cfg.flash_same_shot_off
                or (s_dense is not None and s_dense >= cfg.flash_dense_same)
                or (s_sscd is not None and s_sscd >= cfg.flash_sscd_same))
        strong = nov >= cfg.flash_strong_novelty and (
            (s_adj is not None and s_adj < cfg.flash_strong_adj)
            or (s_adj_dense is not None and s_adj_dense < cfg.flash_strong_adj_dense))
        if strong and not same:
            return True, "keep_flash_strong_cut"
        return False, "merge_flash"

    if nov >= cfg.hard_keep_novelty and s_adj_dense is not None and s_adj_dense <= cfg.hard_keep_adj_dense:
        return True, "keep_hard_discontinuity"
    if s_adj_dense is None and s_adj is not None and s_adj < cfg.adj_strong_cut:
        return True, "keep_adj_strong_cut"

    pix_ok = f["pixel_sim"] is None or f["pixel_sim"] >= cfg.merge_pixel_min
    adj_ok = s_adj is None or s_adj >= cfg.merge_adj_min
    if s_blend >= thr and pix_ok and adj_ok:
        return False, "merge_same_shot"

    if s_sscd is not None and abs(s_blend - thr) <= cfg.ambiguous_band:
        req = _sscd_required(f["motion"], cfg)
        if s_sscd >= req:
            return False, "merge_same_shot_sscd"
        if s_sscd <= req - cfg.sscd_cut_margin:
            return True, "keep_cut_sscd"
    return True, "default_keep"


def record(f: dict, thr: float, cfg: ValidatorConfig, **extra) -> dict:
    """The features worth keeping with a decision (for the viewer and for tuning)."""
    fe = f["flash"]
    r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()
         if k not in ("flash", "flash_cut", "sscd")}
    r.update({"s_blend": round(blended(f, cfg), 4), "s_sscd": sscd_used(f, thr, cfg),
              "flash": bool(fe["is_flash"]), "flash_conf": round(float(fe["conf"]), 4),
              "flash_amp": round(float(fe["amp"]), 2), "flash_dur": int(fe["dur"]),
              "threshold": round(thr, 4)})
    r.update(extra)
    return r


# --------------------------------------------------------------------------------------------- #
# Feature extraction over a video (one decode pass, bounded memory)
# --------------------------------------------------------------------------------------------- #
def extract(path: str, cuts: list[int], total: int, dino: DinoEmbedder, cfg: ValidatorConfig,
            sscd: Optional[SscdEmbedder] = None,
            progress: Optional[Callable[[int, int], None]] = None,
            check_abort: Optional[Callable[[], None]] = None,
            lookbacks: Optional[dict[int, tuple[int, int]]] = None) -> dict[int, dict]:
    """Features for each cut (and for the flash-corrected position of flash cuts). `lookbacks`
    {cut: (first, end)}: frames before a transition to compare with the frames after the cut."""
    lookbacks = lookbacks or {}
    cuts = sorted(set(int(c) for c in cuts if 0 < c < total))
    out: dict[int, dict] = {}
    if not cuts:
        return out
    gap, ws, wl = cfg.window_geometry()
    reach = cfg.reach()                         # frames read on each side of a cut
    shift = cfg.flash_center + 2                # a flash correction moves a cut this far at most
    need: set[int] = set()
    for c in cuts:
        need.update(range(max(0, c - reach), min(total, c + reach)))
        need.update(range(max(0, c - cfg.flash_scan), min(total, c + cfg.flash_scan + 1)))
        if c in lookbacks:
            need.update(range(*lookbacks[c]))
    order = sorted(need)
    # lowest frame each remaining cut still needs (frames below it can be dropped)
    low = [min(c - reach - shift, lookbacks[c][0] if c in lookbacks else c) for c in cuts]
    for k in range(len(low) - 2, -1, -1):
        low[k] = min(low[k], low[k + 1])
    rgb_near = set()
    if sscd is not None:
        for c in cuts:
            rgb_near.update(range(max(0, c - 12), min(total, c + 12)))

    fr = _Frames()
    device = dino.device
    pending_idx: list[int] = []
    pending_rgb: list[np.ndarray] = []
    next_cut = 0                                 # index into cuts of the next cut to finish
    size = dino.size

    def flush():
        if not pending_idx:
            return
        x = torch.from_numpy(np.stack(pending_rgb)).to(device).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True,
                          align_corners=False).round_().clamp_(0, 255)
        luma, vhi, hist = frame_signals(x)
        cls, dense = dino(x, cfg.dense_grid)
        for j, i in enumerate(pending_idx):
            fr.cls[i], fr.dense[i] = cls[j], dense[j]
            fr.luma[i], fr.vhi[i], fr.hist[i] = float(luma[j]), float(vhi[j]), hist[j]
        pending_idx.clear()
        pending_rgb.clear()

    def finish_ready(upto: int):
        nonlocal next_cut
        while next_cut < len(cuts) and (cuts[next_cut] + reach + shift <= upto or upto >= total - 1):
            c = cuts[next_cut]
            f = cut_features(fr, c, total, cfg, device, sscd, lookbacks.get(c))
            if f is not None:
                out[c] = f
                fc = f.get("flash_cut")
                if fc is not None and fc != c and 0 < fc < total and fc not in out:
                    g = cut_features(fr, fc, total, cfg, device, None)
                    if g is not None:
                        out[fc] = {**g, "derived_from": c}
            next_cut += 1
            if next_cut < len(cuts):
                fr.drop_before(low[next_cut])

    frames = prefetch(iter_frames_at(path, order, cfg.sscd_size, cfg.sscd_size))
    last = -1
    try:
        for i, rgb in frames:
            pending_idx.append(i)
            pending_rgb.append(rgb)
            if i in rgb_near:
                fr.rgb[i] = rgb
            last = i
            if len(pending_idx) >= cfg.batch_size or (
                    next_cut < len(cuts) and i >= cuts[next_cut] + reach + shift):
                flush()
                finish_ready(i)
                if progress:
                    progress(i, order[-1])
                if check_abort:
                    check_abort()
    finally:
        frames.close()
    flush()
    finish_ready(total)
    if progress:
        progress(order[-1], order[-1])
    return out


# --------------------------------------------------------------------------------------------- #
# Applying decisions to a scene list
# --------------------------------------------------------------------------------------------- #
def lookback_windows(sl: SceneList, cfg: ValidatorConfig) -> dict[int, tuple[int, int]]:
    """For each cut the detector says ends a transition (a dissolve or fade it did not give a range
    of its own): frames from before the transition, inside the previous segment. The frames just
    before such a cut are the end of the transition and already look like the next shot, so they
    alone would call it the same shot."""
    fps = float(sl.fps) if sl.fps else 25.0
    back = max(cfg.lookback_frames, int(round(cfg.lookback_seconds * fps)))
    out = {}
    segs = sl.segments
    for i in range(1, len(segs)):
        cut = segs[i].cut
        if cut is None or cut.relation != "transition" or segs[i - 1].kind != "shot" or segs[i].kind != "shot":
            continue
        first, end = segs[i].start - back - cfg.lookback_frames, segs[i].start - back
        if first >= segs[i - 1].start:
            out[segs[i].start] = (first, end)
    return out


def eligible_cuts(sl: SceneList) -> list[int]:
    """Boundaries between two shots; boundaries touching a transition are not judged."""
    segs = sl.segments
    return [segs[i].start for i in range(1, len(segs))
            if segs[i - 1].kind == "shot" and segs[i].kind == "shot"]


def apply(sl: SceneList, feats: dict[int, dict], cfg: ValidatorConfig) -> SceneList:
    """Decide every eligible boundary of `sl` from precomputed features; returns a new list."""
    out = sl.copy()
    before = len(out.segments)
    fps = float(out.fps) if out.fps else 0.0
    guard = max(2, int(round(fps * cfg.short_guard_seconds))) if fps > 0 else 8
    eligible = set(eligible_cuts(sl))
    scores = [blended(feats[c], cfg) for c in sorted(eligible) if c in feats]
    thr = float(cfg.threshold) if cfg.threshold is not None else adaptive_threshold(scores, cfg)

    segs = out.segments
    keep_mask, moves, notes = {}, {}, {}
    cur_start, cur_end = segs[0].start, segs[0].end
    for i in range(1, len(segs)):
        nxt = segs[i]
        c = nxt.start
        if c not in eligible or c not in feats:
            keep_mask[i] = True
            cur_start, cur_end = nxt.start, nxt.end
            continue
        f = feats[c]
        keep, reason = decide(f, thr, cfg)
        if not keep and f.get("s_lookback") is not None and f["s_lookback"] < min(thr, cfg.lookback_cut):
            keep, reason = True, "keep_after_transition"
        fe = f["flash"]
        s_sscd = sscd_used(f, thr, cfg)
        min_conf = cfg.guard_flash_conf_min_sscd if s_sscd is not None else cfg.guard_flash_conf_min
        guard_note = None
        if cur_end - cur_start <= guard or nxt.length <= guard:
            flash_like = bool(fe["spike_frames"]) and fe["boundary_hit"] and fe["conf"] >= cfg.guard_flash_like_conf
            same = blended(f, cfg) >= max(0.0, thr - cfg.guard_same_shot_off)
            if flash_like and same:
                keep, guard_note = False, "short_guard_merge_flash"
            elif not (fe["is_flash"] and fe["conf"] >= min_conf):
                keep, guard_note = True, "short_guard_keep"
        notes[i] = record(f, thr, cfg, decision="keep" if keep else "merge", reason=reason,
                          guard=guard_note)
        keep_mask[i] = keep
        if keep:
            new = c
            if fe["is_flash"] and fe["conf"] >= min_conf and f.get("flash_cut") is not None:
                lo, hi = cur_start + 1, nxt.end - 1
                if lo <= hi:
                    new = int(max(lo, min(hi, f["flash_cut"])))
            if new != c:
                moves[i] = new
            cur_start, cur_end = new, nxt.end
        else:
            cur_end = nxt.end

    for i, new in moves.items():
        segs[i - 1].end = new
        segs[i].start = new
        if segs[i].cut is not None:
            if segs[i].cut.detected_frame is None:
                segs[i].cut.detected_frame = segs[i].cut.frame
            segs[i].cut.frame = new
        notes[i]["moved_for_flash"] = new
    for i, note in notes.items():
        if keep_mask[i] and segs[i].cut is not None:
            segs[i].cut.validation = note
    for i in sorted((i for i, k in keep_mask.items() if not k), reverse=True):
        out.remove_boundary(i, "validate", notes[i]["reason"] if not notes[i]["guard"] else notes[i]["guard"],
                            validation=notes[i])

    merged_tiny = _tiny_flash_pass(out, feats, thr, cfg) if cfg.tiny_flash_pass else 0
    out.record("validate", {**{k: v for k, v in asdict(cfg).items()
                               if k in ("window", "flash_sensitivity", "use_sscd", "threshold",
                                        "dense_weight", "short_guard_seconds", "tiny_flash_pass")},
                            "thresholdUsed": round(thr, 4)},
               before, judged=len(notes), tinyFlashMerged=merged_tiny)
    out.check()
    return out


def _tiny_flash_pass(sl: SceneList, feats: dict[int, dict], thr: float, cfg: ValidatorConfig) -> int:
    """Slivers of a few frames between two cuts, one of them a flash, are the flash itself."""
    fps = float(sl.fps) if sl.fps else 0.0
    ultra = max(1, int(round(fps * cfg.tiny_ultra_short_seconds))) if fps > 0 else 1
    guard = max(2, int(round(fps * cfg.tiny_guard_seconds))) if fps > 0 else 2

    def info(cut: int):
        f = feats.get(cut)
        if f is None:
            return False, 0.0
        fe = f["flash"]
        like = bool(fe["spike_frames"]) and fe["boundary_hit"] and fe["conf"] >= cfg.tiny_flash_conf
        s = blended(f, cfg)
        same = s >= max(0.0, thr - cfg.guard_same_shot_off)
        return like, fe["conf"] * cfg.tiny_score_flash_w + ((s - thr) if same else 0.0)

    segs, merged, i = sl.segments, 0, 1
    while i < len(segs) - 1:
        seg = segs[i]
        if seg.kind == "shot" and seg.length <= guard and segs[i - 1].kind == "shot" and segs[i + 1].kind == "shot":
            like_a, score_a = info(seg.start)
            like_b, score_b = info(seg.end)
            if seg.length <= ultra or like_a or like_b:
                reason = f"flash sliver of {seg.length} frames"
                if score_a >= score_b:
                    sl.remove_boundary(i, "validate_flash", reason)       # sliver joins previous
                else:
                    sl.remove_boundary(i + 1, "validate_flash", reason)   # next joins the sliver
                merged += 1
                continue
        i += 1
    return merged


def run(sl: SceneList, dino: DinoEmbedder, cfg: ValidatorConfig, sscd: Optional[SscdEmbedder] = None,
        cache: Optional[dict] = None, progress=None, check_abort=None) -> SceneList:
    """Validate `sl`. `cache` (any dict kept between calls) lets threshold changes skip decoding."""
    cuts = eligible_cuts(sl)
    lookbacks = lookback_windows(sl, cfg)
    mtime = os.path.getmtime(sl.source.path) if os.path.exists(sl.source.path) else 0
    key = (sl.source.path, mtime, sl.frame_count, dino.name,
           getattr(sscd, "name", None) if cfg.use_sscd else None, cfg.feature_key())
    store = cache.setdefault(key, {}) if cache is not None else {}
    # A position first computed as another cut's flash correction has no SSCD; redo it as a cut.
    missing = [c for c in cuts if c not in store or (store[c] or {}).get("derived_from") is not None
               or (c in lookbacks and store[c] is not None and store[c].get("s_lookback") is None)]
    if missing:
        store.update(extract(sl.source.path, missing, sl.frame_count, dino, cfg,
                             sscd if cfg.use_sscd else None, progress, check_abort,
                             {c: w for c, w in lookbacks.items() if c in missing}))
        for c in missing:
            store.setdefault(c, None)
    feats = {c: f for c, f in store.items() if f is not None}
    return apply(sl, feats, cfg)


def config_with(cfg: Optional[ValidatorConfig] = None, **kw) -> ValidatorConfig:
    return replace(cfg or ValidatorConfig(), **kw)
