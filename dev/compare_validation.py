#!/usr/bin/env python3
"""Run our validator on the exact cut list the old pipeline judged, and compare cut by cut.

The old pipeline stored ~30 features and the decision for every boundary it validated
(dev/reference/<video>/pipeline_<preset>.json). We rebuild a scene list with those boundaries,
validate it, and report feature differences and decision agreement. Our features are saved to
dev/reference/<video>/ours_validation_<preset>.json for closer study.

    python dev/compare_validation.py short-film [--preset film] [--device cpu] [--threads 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", default=["short-film"])
    p.add_argument("--preset", default="film")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--precision", default="fp32")
    a = p.parse_args()

    import torch

    torch.set_num_threads(a.threads)
    from conftest import load_pack

    load_pack()
    from comfyui_scenecut.nodes.validation import get_validator
    from comfyui_scenecut.scenecut import validate as V
    from comfyui_scenecut.scenecut.media import probe
    from comfyui_scenecut.scenecut.scenes import Cut, SceneList, Segment

    handle = get_validator(precision=a.precision, device=a.device)
    for name in a.videos:
        ref = json.loads((ROOT / "dev/reference" / name / f"pipeline_{a.preset}.json").read_text())
        val = ref["validation"]
        decisions = val["decisions"]
        cuts = [int(d["cut_frame"]) for d in decisions]
        total = int(ref["totalFrames"])
        info = probe(str(ROOT / "dev/videos" / f"{name}.mp4"), count_frames=False)
        bounds = [0] + cuts + [total]
        segs = [Segment(s, e, cut=Cut(frame=s) if s else None) for s, e in zip(bounds, bounds[1:])]
        sl = SceneList(source=info, frame_count=total, segments=segs)
        sl.check()

        cfg = V.ValidatorConfig(window=int(val["window"]), flash_sensitivity=float(val["flashSensitivity"]),
                                use_sscd=handle.sscd_path is not None)
        cache: dict = {}
        t0 = time.time()
        out = V.run(sl, handle.dino(), cfg, handle.sscd(), cache=cache,
                    progress=lambda d, t: print(f"\r  decoding/embedding {d}/{t}", end="", file=sys.stderr))
        print(file=sys.stderr)
        elapsed = time.time() - t0
        feats = next(iter(cache.values()))
        thr = out.history[-1]["params"]["thresholdUsed"]

        # per-cut comparison
        keys = ["s_cls", "s_dense", "s_adj", "s_adj_dense", "novelty", "motion", "pixel_sim", "s_blend"]
        diffs = {k: [] for k in keys}
        agree, rows = 0, []
        kept_cuts = set(out.cuts)
        for d in decisions:
            c = int(d["cut_frame"])
            f = feats.get(c)
            ours_keep = c in kept_cuts or any(s.cut and s.cut.detected_frame == c for s in out.segments)
            ref_keep = bool(d["final_keep"])
            agree += ours_keep == ref_keep
            if f is not None:
                rec = V.record(f, thr, cfg)
                for k in keys:
                    if d.get(k) is not None and rec.get(k) is not None:
                        diffs[k].append(abs(float(rec[k]) - float(d[k])))
                ours_reason = V.decide(f, thr, cfg)[1]
            else:
                rec, ours_reason = {}, "no features"
            if ours_keep != ref_keep or ours_reason != d["reason"].replace("merge_flash_default", "merge_flash"):
                rows.append((c, ref_keep, d["reason"], ours_keep, ours_reason,
                             d.get("s_blend"), rec.get("s_blend"), d.get("novelty"), rec.get("novelty"),
                             d.get("s_adj_dense"), rec.get("s_adj_dense"),
                             d.get("flash_is_flash"), rec.get("flash")))
        print(f"== {name} / {a.preset}: {len(decisions)} boundaries, threshold ref {val['baseThreshold']:.3f} "
              f"ours {thr:.3f}, decisions agree {agree}/{len(decisions)}, "
              f"kept ref {val['boundariesOut']} ours {len(out.cuts)} | {elapsed:.0f}s on {a.device}")
        for k in keys:
            v = np.asarray(diffs[k])
            if v.size:
                print(f"   {k:12s} |diff| median {np.median(v):.4f}  p90 {np.percentile(v, 90):.4f}  max {v.max():.4f}")
        if rows:
            print("   differing reasons / decisions (cut, ref keep, ref reason, ours keep, ours reason, "
                  "s_blend ref/ours, novelty ref/ours, adj_dense ref/ours, flash ref/ours):")
            for r in rows:
                print("    ", " ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r))
        dump = {str(c): V.record(f, thr, cfg) for c, f in feats.items() if f is not None}
        (ROOT / "dev/reference" / name / f"ours_validation_{a.preset}.json").write_text(json.dumps(
            {"threshold": thr, "kept": out.cuts, "features": dump}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
