#!/usr/bin/env python3
"""Run our OmniShotCut detection on the test videos and diff it against the reference dumps.

    python dev/compare_raw.py [--backend ffmpeg|pyav] [--batch N] [--ref raw_notf32.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scenecut.detect import detect_video  # noqa: E402
from scenecut.media import probe  # noqa: E402
from scenecut.omnishotcut import load_omnishotcut  # noqa: E402

DEFAULT_CKPT = Path(os.environ.get("COMFYUI_PATH", Path.home() / "ComfyUI")) / "models/scenecut/OmniShotCut_ckpt.pth"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", default=["short-film", "music-video", "ai-fantasi"])
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--backend", default="auto")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ref", default="raw_notf32.json")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    model = load_omnishotcut(a.ckpt, a.device)
    failures = 0
    for name in a.videos:
        video = ROOT / "dev/videos" / f"{name}.mp4"
        ref = json.loads((ROOT / "dev/reference" / name / a.ref).read_text())
        t0 = time.time()
        info = probe(str(video))
        t1 = time.time()
        raw = detect_video(model, str(video), overlap=ref["overlap"], batch_windows=a.batch,
                           backend=a.backend)
        t2 = time.time()
        ours = [[s, e] for s, e, _ in raw.ranges()]
        ours_intra = [b.intra for _, _, b in raw.ranges()]
        ours_inter = [b.inter for _, _, b in raw.ranges()]
        ok_ranges = ours == ref["ranges"]
        ok_labels = ours_intra == ref["intra"] and ours_inter == ref["inter"]
        conf_diff = max((abs(b.confidence - c) for (_, _, b), c in zip(raw.ranges(), ref["confidence"])),
                        default=0.0) if ok_ranges else float("nan")
        print(f"{name:12s} frames ours={raw.frame_count} ref={ref['decodedFrames']} probe={info.frame_count} "
              f"| ranges {len(ours)} vs {len(ref['ranges'])} {'SAME' if ok_ranges else 'DIFF'} "
              f"| labels {'SAME' if ok_labels else 'DIFF'} | max conf diff {conf_diff:.1e} "
              f"| probe {t1 - t0:.1f}s detect {t2 - t1:.1f}s")
        if not ok_ranges:
            failures += 1
            a_ends = {e for _, e in ours}
            b_ends = {e for _, e in ref["ranges"]}
            print("   only ours:", sorted(a_ends - b_ends)[:20], " only ref:", sorted(b_ends - a_ends)[:20])
        elif not ok_labels:
            failures += 1
            diffs = [i for i in range(len(ours)) if (ours_intra[i], ours_inter[i]) != (ref["intra"][i], ref["inter"][i])]
            print("   label diffs at ranges:", diffs[:20])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
