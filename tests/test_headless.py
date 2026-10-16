"""SceneCut without ComfyUI: `scenecut.headless` and `pipeline.run_preset`.

The model tests need the OmniShotCut checkpoint, and for validation DINOv3, SSCD and
`transformers`; the parity test also needs dev/videos and dev/reference. Each skips without them.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from scenecut import validate as V
from scenecut.omnishotcut import load_omnishotcut
from scenecut.pipeline import run_preset
from scenecut.presets import get_preset

ROOT = Path(__file__).resolve().parent.parent
MODELS = os.environ.get("SCENECUT_MODELS", os.path.join(
    os.environ.get("COMFYUI_PATH", os.path.expanduser("~/AI/ComfyUI")), "models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_run_preset_without_validator(omni_ckpt, synthetic_video):
    sl = run_preset(load_omnishotcut(omni_ckpt, DEVICE), synthetic_video, get_preset("film"))
    assert sl.cuts == [50, 100]
    assert [h["stage"] for h in sl.history][-1] == "ultra_short"       # no validation without DINOv3


def test_run_preset_validates_when_given_an_embedder(omni_ckpt, synthetic_video):
    """Validation runs through run_preset (it used to raise NotImplementedError). The embedder
    here is a stand-in that makes every frame look different, so both cuts are kept."""
    def forward(x):
        b = x.shape[0]
        seq = torch.zeros(b, 5 + 196, 8, device=x.device)
        seq[..., 0] = x.mean(dim=(1, 2, 3))[:, None]
        seq[..., 1] = x[:, 0].std(dim=(1, 2))[:, None]
        seq[..., 2] = x[:, 2].mean(dim=(1, 2))[:, None]
        return seq

    dino = V.DinoEmbedder(forward, 5, DEVICE, name="stand-in")
    sl = run_preset(load_omnishotcut(omni_ckpt, DEVICE), synthetic_video, get_preset("film"), dino=dino)
    stages = [h["stage"] for h in sl.history]
    assert "validate" in stages and stages[-1] == "ultra_short"
    assert sl.segments[0].start == 0 and sl.segments[-1].end == sl.frame_count


def _models():
    pytest.importorskip("transformers")
    from scenecut import weights
    from scenecut.headless import Models

    need = [os.path.join(MODELS, weights.OMNISHOTCUT.folder, weights.OMNISHOTCUT.filename),
            os.path.join(MODELS, weights.DINOV3.folder, weights.DINOV3.filename),
            os.path.join(MODELS, weights.SSCD.folder, weights.SSCD.filename)]
    if not all(os.path.isfile(p) for p in need):
        pytest.skip(f"SceneCut weights not found under {MODELS}")
    return Models.load(MODELS, device=DEVICE, allow_download=False)


def test_headless_dino_matches_the_calibrated_model():
    """The weights load into the Hugging Face module completely (a key mismatch raises in
    load_dino), and it embeds: unit vectors, no NaN, the same image to the same vector and a
    different image to a different one. The cut-for-cut check is the parity test below."""
    models = _models()
    g = torch.Generator().manual_seed(0)
    a = torch.rand(1, 3, 224, 224, generator=g) * 255
    b = torch.zeros(1, 3, 224, 224)
    b[..., :112] = 255.0
    cls, dense = models.dino(torch.cat([a, a, b]).to(models.dino.device))
    assert cls.shape == (3, 1024) and dense.shape == (3, 49, 1024)
    assert not torch.isnan(cls).any() and not torch.isnan(dense.float()).any()
    assert float(cls[0] @ cls[1]) == pytest.approx(1.0, abs=1e-4)
    assert float(cls[0] @ cls[2]) < 0.99


@pytest.mark.parametrize("name,preset", [("short-film", "film")])
def test_headless_matches_the_node_pipeline(name, preset):
    """Same cuts as the ComfyUI run recorded in dev/reference (ours_validation_<preset>.json)."""
    video = ROOT / "dev" / "videos" / f"{name}.mp4"
    ref = ROOT / "dev" / "reference" / name / f"ours_validation_{preset}.json"
    if not video.exists() or not ref.exists():
        pytest.skip(f"reference data for {name} not present")
    from scenecut.headless import run

    sl = run(str(video), preset, _models())
    expected = json.loads(ref.read_text())
    assert sl.cuts == expected["kept"]
    used = [h for h in sl.history if h["stage"] == "validate"][0]["params"]["thresholdUsed"]
    assert used == pytest.approx(expected["threshold"], abs=1e-3)
