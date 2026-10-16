#!/usr/bin/env python3
"""Reference outputs from the old headless SceneCutGUI pipeline, for parity tests.

This is a test oracle, not part of the pack. It runs in the *old* project's environment, against
a SceneCutGUI checkout, and writes plain JSON/tensors that tests/ compare against:

    /path/to/old-pipeline/.venv/bin/python dev/ref_oracle.py \
        --scenecutgui /path/to/old-pipeline/vendor/SceneCutGUI \
        --shotcut /path/to/old-pipeline/shotcut.py \
        --weights /path/to/old-pipeline/weights \
        dev/videos/short-film.mp4 dev/videos/music-video.mp4

Per video it writes dev/reference/<stem>/:
    raw.json         OmniShotCut boundaries before any post-processing (overlap 20, confidence 0)
    raw_notf32.json  the same with TF32 convolutions off (--no-tf32): deterministic fp32
    window0.pt       logits of the first 100-frame window, for a numeric comparison
    pipeline_*.json  the full shotcut.py result for each preset, with validator decisions
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_shotcut(shotcut_path: Path, scenecutgui: Path):
    os.environ["SCENECUTGUI_DIR"] = str(scenecutgui)
    spec = importlib.util.spec_from_file_location("old_shotcut", shotcut_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raw_boundaries(shotcut, video: str, weights: Path, out_dir: Path) -> None:
    import torch

    scg = shotcut.scg
    device = shotcut._resolve_device("cuda")
    model = scg.omnishotcut.load(str(weights / "OmniShotCut_ckpt.pth"), device=device)
    margs = model._model_args
    w, h = int(margs.process_width), int(margs.process_height)

    fd, raw_path = tempfile.mkstemp(suffix=".rgb24")
    os.close(fd)
    try:
        video_np, decoded = scg._omnishot_decode_to_memmap(video, w, h, raw_path)

        from omnishotcut.engine import split_videos, video_transform
        chunk = split_videos(video_np, int(margs.max_process_window_length), 20)[0][0]
        x = video_transform(chunk.copy()).unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model._model(x)
        suffix = "" if torch.backends.cudnn.allow_tf32 else "_notf32"
        torch.save({k: v.detach().cpu() for k, v in outputs.items() if torch.is_tensor(v)},
                   out_dir / f"window0{suffix}.pt")
        torch.save(torch.from_numpy(chunk.copy()), out_dir / "window0_frames.pt")

        ranges, intra, inter, confs = scg._omnishot_predict_windows(
            video_np, model._model, margs, 20, confidence=0.0)
        fps, total = scg.get_video_info(video)
        (out_dir / f"raw{suffix}.json").write_text(json.dumps({
            "video": Path(video).name,
            "decodedFrames": int(decoded),
            "probedFrames": int(total),
            "fps": fps,
            "overlap": 20,
            "ranges": ranges,
            "intra": intra,
            "inter": inter,
            "confidence": confs,
        }, indent=1))
        del video_np
    finally:
        with contextlib.suppress(OSError):
            os.remove(raw_path)
    del model
    shotcut.free_cuda()


def pipeline(shotcut, video: str, weights: Path, out_dir: Path, preset: str) -> None:
    args = shotcut.build_parser().parse_args([
        video, "--weights-dir", str(weights), "--preset", preset, "--ai-validate",
    ])
    result = shotcut.detect(video, args)
    (out_dir / f"pipeline_{preset}.json").write_text(json.dumps(result, indent=1))
    shotcut.free_cuda()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--scenecutgui", required=True, type=Path)
    p.add_argument("--shotcut", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", type=Path, default=HERE / "reference")
    p.add_argument("--skip-pipeline", action="store_true")
    p.add_argument("--no-tf32", action="store_true",
                   help="disable TF32 convolutions (cuDNN's default) and write raw_notf32.json; "
                        "TF32 moves OmniShotCut's logits by up to ~0.6 and differs per GPU")
    a = p.parse_args()
    if a.no_tf32:
        import torch
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    shotcut = load_shotcut(a.shotcut.resolve(), a.scenecutgui.resolve())
    for video in a.videos:
        video = str(Path(video).resolve())
        out_dir = a.out / Path(video).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {video}", file=sys.stderr)
        raw_boundaries(shotcut, video, a.weights.resolve(), out_dir)
        if not a.skip_pipeline:
            for preset in ("film", "music-video"):
                pipeline(shotcut, video, a.weights.resolve(), out_dir, preset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
