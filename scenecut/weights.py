"""Model files SceneCut uses, where they come from, and downloading them on first use."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("SceneCut")


@dataclass(frozen=True)
class ModelFile:
    filename: str
    folder: str               # ComfyUI model folder name
    size: int
    sha256: Optional[str]
    license: str
    purpose: str
    repos: tuple = ()         # Hugging Face (repo, filename) pairs, tried in order
    url: Optional[str] = None # plain HTTPS fallback
    extra: tuple = ()         # (repo, filename, saved-as) files to fetch alongside, e.g. a licence
    page: str = ""


OMNISHOTCUT = ModelFile(
    filename="OmniShotCut_ckpt.pth",
    folder="scenecut",
    size=164_149_963,
    sha256="5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa",
    license="MIT",
    purpose="shot boundary detection",
    repos=(("uva-cv-lab/OmniShotCut", "OmniShotCut_ckpt.pth"),),
    page="https://huggingface.co/uva-cv-lab/OmniShotCut",
)

# Meta's official repo is gated (accept the licence on its page and log in with `hf auth login`).
# The mirror holds the byte-identical file (same SHA-256), which the DINOv3 licence permits when
# the licence travels with it; the checksum is verified either way.
DINOV3 = ModelFile(
    filename="dinov3_vitl16_pretrain_lvd1689m.safetensors",
    folder="clip_vision",
    size=1_212_559_808,
    sha256="dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    license="DINOv3 License (Meta)",
    purpose="cut validation (DINOv3 ViT-L/16)",
    repos=(("facebook/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors"),
           ("camenduru/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors")),
    extra=(("camenduru/dinov3-vitl16-pretrain-lvd1689m", "LICENSE.md",
            "dinov3_vitl16_pretrain_lvd1689m.LICENSE.md"),),
    page="https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
)

SSCD = ModelFile(
    filename="sscd_disc_large.torchscript.pt",
    folder="scenecut",
    size=178_141_104,
    sha256="16a196359f91e5fc4e774d5679244c8a661c182309ad4ede0c7e83f6a9861fa0",
    license="MIT",
    purpose="cut validation tie-breaks (SSCD copy detection)",
    url="https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt",
    page="https://github.com/facebookresearch/sscd-copy-detection",
)


def ensure(model: ModelFile, folder: str, allow_download: bool = True) -> str:
    """Path of `model` inside `folder`, downloading it if it is missing."""
    path = os.path.join(folder, model.filename)
    if os.path.isfile(path):
        return path
    if not allow_download:
        raise FileNotFoundError(f"{model.filename} not found in {folder}; get it from {model.page}")
    os.makedirs(folder, exist_ok=True)
    tmp_dir = os.path.join(folder, ".scenecut-download")
    errors = []
    try:
        for repo, name in model.repos:
            try:
                log.info("SceneCut: downloading %s (%.0f MB) from huggingface.co/%s",
                         model.filename, model.size / 1e6, repo)
                from huggingface_hub import hf_hub_download

                got = hf_hub_download(repo, name, local_dir=tmp_dir)
                _verify_and_place(got, path, model)
                break
            except Exception as e:  # noqa: BLE001 - gated, offline, missing: try the next source
                errors.append(f"{repo}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        else:
            if not model.url:
                raise RuntimeError("; ".join(errors))
            log.info("SceneCut: downloading %s (%.0f MB) from %s", model.filename,
                     model.size / 1e6, model.url)
            got = os.path.join(tmp_dir, model.filename)
            os.makedirs(tmp_dir, exist_ok=True)
            _http_get(model.url, got)
            _verify_and_place(got, path, model)
        for repo, name, saved in model.extra:
            try:
                from huggingface_hub import hf_hub_download

                shutil.copyfile(hf_hub_download(repo, name, local_dir=tmp_dir), os.path.join(folder, saved))
            except Exception:  # noqa: BLE001 - a companion file is nice to have
                pass
    except Exception as e:
        raise RuntimeError(
            f"Could not download {model.filename} ({model.purpose}). Put it in {folder} by hand: "
            f"{model.page}\n{e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return path


def _verify_and_place(got: str, path: str, model: ModelFile) -> None:
    if model.sha256 and _sha256(got) != model.sha256:
        os.remove(got)
        raise RuntimeError(f"{model.filename}: checksum mismatch after download")
    os.replace(got, path)


def _http_get(url: str, dest: str) -> None:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as r, open(dest + ".part", "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    os.replace(dest + ".part", dest)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()
