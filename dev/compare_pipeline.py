#!/usr/bin/env python3
"""Compare our detect -> transitions -> min length -> snap stages with the old pipeline.

The reference ran with validation on, so after the min-length stage we compare against its
final cut list: every final reference cut should be one of our snapped cuts, at the same frame.

    python dev/compare_pipeline.py [videos...] [--preset film|music-video]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scenecut.omnishotcut import load_omnishotcut  # noqa: E402
from scenecut.pipeline import detect  # noqa: E402
from scenecut.refine import handle_transitions, merge_short, snap_cuts  # noqa: E402

DEFAULT_CKPT = Path(os.environ.get("COMFYUI_PATH", Path.home() / "ComfyUI")) / "models/scenecut/OmniShotCut_ckpt.pth"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", default=["short-film", "music-video", "ai-fantasi"])
    p.add_argument("--preset", nargs="*", default=["film", "music-video"])
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    model = load_omnishotcut(a.ckpt, a.device)
    for name in a.videos:
        video = str(ROOT / "dev/videos" / f"{name}.mp4")
        base = detect(model, video)
        merged = handle_transitions(base, "merge_previous")
        for preset in a.preset:
            ref_path = ROOT / "dev/reference" / name / f"pipeline_{preset}.json"
            if not ref_path.exists():
                continue
            ref = json.loads(ref_path.read_text())
            counts = ref["counts"]
            short = merge_short(merged, ref["params"]["minSceneLen"])
            snapped = snap_cuts(short, 2)
            ours = set(snapped.cuts)
            unsnapped = set(short.cuts)
            ref_cuts = [s["endFrame"] for s in ref["scenes"][:-1]]
            exact = sum(c in ours for c in ref_cuts)
            offsets = Counter()
            for c in ref_cuts:
                if c not in ours:
                    d = min(ours, key=lambda x: abs(x - c)) - c if ours else None
                    offsets[d] += 1
            print(f"{name:12s} {preset:12s} raw {len(base.segments)}/{counts['rawRanges']} "
                  f"after-transitions {len(merged.segments)}/{counts['afterConversion']} "
                  f"after-min-length {len(short.segments)}/{counts['afterMinSceneLen']} "
                  f"| ref final cuts {len(ref_cuts)}: {exact} at our snapped frames"
                  f" ({sum(c in unsnapped for c in ref_cuts)} at unsnapped)"
                  + (f", others offset by {dict(offsets)}" if offsets else "")
                  + f" | snap moved {snapped.history[-1]['moved']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
