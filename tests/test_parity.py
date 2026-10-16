"""Our OmniShotCut against reference dumps of the original implementation (dev/ref_oracle.py).

Needs dev/videos/<name>.mp4 and dev/reference/<name>/ (not distributed); skips otherwise.
"""
import json
from pathlib import Path

import pytest
import torch

from scenecut.omnishotcut import load_omnishotcut, strict_fp32
from scenecut.pipeline import detect
from scenecut.refine import handle_transitions, merge_short

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ["short-film", "music-video", "ai-fantasi"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _need(name: str, *files: str) -> Path:
    ref = ROOT / "dev" / "reference" / name
    if not (ROOT / "dev" / "videos" / f"{name}.mp4").exists() or not all((ref / f).exists() for f in files):
        pytest.skip(f"reference data for {name} not present")
    return ref


@pytest.fixture(scope="module")
def model(omni_ckpt):
    return load_omnishotcut(omni_ckpt, DEVICE)


@pytest.mark.parametrize("name", VIDEOS)
def test_window_logits_match_fp32_reference(model, name):
    ref = _need(name, "window0_notf32.pt", "window0_frames.pt")
    frames = torch.load(ref / "window0_frames.pt")
    expected = torch.load(ref / "window0_notf32.pt")
    with torch.inference_mode(), strict_fp32():
        out = model(frames[None].to(DEVICE))
    for ours, theirs in (("intra", "intra_clip_logits"), ("inter", "inter_clip_logits"),
                         ("range", "pred_shot_logits")):
        assert (out[ours].float().cpu() - expected[theirs]).abs().max() < 2e-3


@pytest.mark.parametrize("name", VIDEOS)
def test_boundaries_and_stage_counts_match(model, name):
    ref = _need(name, "raw_notf32.json", "pipeline_film.json")
    raw = json.loads((ref / "raw_notf32.json").read_text())
    sl = detect(model, str(ROOT / "dev" / "videos" / f"{name}.mp4"))
    assert [[s.start, s.end] for s in sl.segments] == raw["ranges"]
    assert sl.frame_count == raw["decodedFrames"]

    counts = json.loads((ref / "pipeline_film.json").read_text())["counts"]
    merged = handle_transitions(sl, "merge_previous")
    assert len(merged.segments) == counts["afterConversion"]
    assert len(merge_short(merged, 12).segments) == counts["afterMinSceneLen"]
