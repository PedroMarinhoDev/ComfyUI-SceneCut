"""Test setup.

Core tests import `scenecut` directly. Node tests load the pack by path as `comfyui_scenecut`
(its folder name has a dash) with ComfyUI on sys.path:

    COMFYUI_PATH=~/AI/ComfyUI ~/AI/ComfyUI/.venv/bin/python -m pytest tests

Parity tests need the reference dumps in dev/reference and the videos in dev/videos (not
distributed); they skip without them.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK))

COMFY = os.environ.get("COMFYUI_PATH", os.path.expanduser("~/AI/ComfyUI"))
OMNI_CKPT = os.environ.get(
    "SCENECUT_OMNISHOTCUT", os.path.join(COMFY, "models", "scenecut", "OmniShotCut_ckpt.pth"))


def load_pack():
    """Import the pack as ComfyUI would, or skip when ComfyUI is not available."""
    if "comfyui_scenecut" in sys.modules:
        return sys.modules["comfyui_scenecut"]
    if not os.path.isdir(os.path.join(COMFY, "comfy")):
        pytest.skip("ComfyUI not found (set COMFYUI_PATH)")
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    import comfy.cli_args

    comfy.cli_args.args.cpu = os.environ.get("SCENECUT_TEST_GPU", "") == ""
    spec = importlib.util.spec_from_file_location("comfyui_scenecut", PACK / "__init__.py",
                                                  submodule_search_locations=[str(PACK)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_scenecut"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def omni_ckpt():
    if not os.path.isfile(OMNI_CKPT):
        pytest.skip(f"OmniShotCut checkpoint not found at {OMNI_CKPT}")
    return OMNI_CKPT


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """Three visually unrelated 2-second shots at 25 fps: hard cuts at frames 50 and 100."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg binary needed to build the synthetic video")
    out = tmp_path_factory.mktemp("video") / "three_shots.mp4"
    srcs = ["testsrc2=size=320x240:rate=25:duration=2",
            "mandelbrot=size=320x240:rate=25",
            "smptehdbars=size=320x240:rate=25:duration=2"]
    cmd = [ffmpeg, "-v", "error", "-y"]
    for s in srcs:
        cmd += ["-f", "lavfi", "-t", "2", "-i", s]
    cmd += ["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map", "[v]", "-c:v", "libx264", "-g", "250", "-crf", "18", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)
