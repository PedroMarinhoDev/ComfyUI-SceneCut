"""The nodes as ComfyUI runs them, on a synthetic three-shot video."""
import json
import os

import numpy as np
import pytest
import torch

from conftest import load_pack


@pytest.fixture(scope="module")
def pack():
    return load_pack()


@pytest.fixture(scope="module")
def video(pack, synthetic_video):
    from comfy_api.latest import InputImpl

    return InputImpl.VideoFromFile(synthetic_video)


def test_every_schema_builds(pack):
    from comfyui_scenecut.nodes import NODES

    ids = set()
    for node in NODES:
        schema = node.define_schema()
        assert schema.node_id.startswith("SceneCut") and schema.node_id not in ids
        ids.add(schema.node_id)


def test_browse_policy(pack, tmp_path, monkeypatch):
    from comfy.cli_args import args
    from comfyui_scenecut.nodes import routes

    monkeypatch.delenv("SCENECUT_BROWSE_ROOTS", raising=False)
    monkeypatch.setattr(args, "listen", "127.0.0.1")
    assert routes.browse_roots()[0] is None                     # local: anywhere
    monkeypatch.setattr(args, "listen", "0.0.0.0,::")
    roots, places = routes.browse_roots()                        # on the network: input/output only
    assert roots is not None and {p["name"] for p in places} <= {"input", "output"}
    monkeypatch.setenv("SCENECUT_BROWSE_ROOTS", str(tmp_path))
    assert routes.browse_roots()[0] == [str(tmp_path)]           # explicit roots win


def test_load_video_from_path(pack, synthetic_video):
    from comfyui_scenecut.nodes.video import SceneCutLoadVideo

    assert SceneCutLoadVideo.validate_inputs("") is not True
    assert SceneCutLoadVideo.validate_inputs("/no/such/file.mp4") is not True
    video, frames, fps, duration = SceneCutLoadVideo.execute(f'"{synthetic_video}"').args
    assert frames == 150 and fps == 25.0 and abs(duration - 6.0) < 0.1
    assert video.get_stream_source() == synthetic_video


def test_simple_detect_finds_both_cuts(pack, video, omni_ckpt):
    from comfyui_scenecut.nodes.detect import SceneCutDetect

    out = SceneCutDetect.execute(video, "Film / TV", False, device="cpu")
    sl, shots, summary, images = out.args
    assert tuple(images.shape) == (1, 64, 64, 3)                              # output_images off
    assert shots == 3
    assert [abs(c - e) <= 1 for c, e in zip(sl.cuts, (50, 100))] == [True, True]
    assert "3 shots" in summary


def test_advanced_chain_and_files(pack, video, omni_ckpt, tmp_path, monkeypatch):
    from comfyui_scenecut.nodes.detect import SceneCutDetectAdvanced, SceneCutRefine
    from comfyui_scenecut.nodes.files import SceneCutLoadScenes, SceneCutSaveScenes
    import folder_paths

    raw, _, _ = SceneCutDetectAdvanced.execute(video, 20, 0.0, 4, "pyav").args
    refined, shots, _, _ = SceneCutRefine.execute(raw, "keep", 0.4, 2, 0.05).args
    assert shots == 3

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    path, folder = SceneCutSaveScenes.execute(refined, "scenecut/{video}/scenes", "none (scene list only)").args
    assert json.load(open(path))["format"] == "scenecut.scenes"
    assert folder == str(tmp_path / "scenecut" / "three_shots" / "scenes_00001")   # {video} in the prefix
    loaded, n, images = SceneCutLoadScenes.execute(folder, output_images="first frame per scene").args
    assert n == 3 and loaded.cuts == refined.cuts
    assert tuple(images.shape) == (3, 240, 320, 3)


def test_contact_sheet_pages(pack, video, omni_ckpt):
    from comfyui_scenecut.nodes.detect import SceneCutDetect
    from comfyui_scenecut.nodes.view import SceneCutContactSheet

    sl = SceneCutDetect.execute(video, "Film / TV", False).args[0]
    pages = SceneCutContactSheet.execute("3", 96, 2, 1, 1, 4, scenes=sl).args[0]
    assert isinstance(pages, torch.Tensor) and pages.ndim == 4 and pages.shape[0] == 2
    assert 0.0 <= float(pages.min()) and float(pages.max()) <= 1.0


@pytest.fixture(scope="module")
def dino_weights(pack):
    import folder_paths
    from comfyui_scenecut.scenecut import weights

    if folder_paths.get_full_path("clip_vision", weights.DINOV3.filename) is None:
        pytest.skip("DINOv3 weights not installed (not downloaded by tests)")


def test_validate_keeps_real_cuts(pack, video, omni_ckpt, dino_weights):
    from comfyui_scenecut.nodes.detect import SceneCutDetectAdvanced, SceneCutRefine
    from comfyui_scenecut.nodes.validation import SceneCutValidate, get_validator

    raw = SceneCutDetectAdvanced.execute(video, 20, 0.0, 4, "pyav").args[0]
    refined = SceneCutRefine.execute(raw, "merge_previous", 0.4, 2, 0.05).args[0]
    handle = get_validator(sscd_name="none", device="cpu")
    sl, shots, summary, _ = SceneCutValidate.execute(refined, 5, 15.0, True, 0.8, validator=handle).args
    assert shots == 3 and sl.cuts == refined.cuts                    # unrelated patterns: real cuts
    for seg in sl.segments[1:]:
        v = seg.cut.validation
        assert v["decision"] == "keep" and v["s_blend"] < 0.8
    assert sl.history[-2]["stage"] == "validate"                     # then the fragment clean-up

    # a fixed threshold reuses the extracted features (no decoding) and still keeps both cuts
    again = SceneCutValidate.execute(refined, 5, 15.0, False, 0.99, validator=handle).args[0]
    assert again.cuts == refined.cuts


def test_viewer_writes_proxy_sprites_and_report(pack, video, omni_ckpt, tmp_path, monkeypatch):
    import folder_paths
    from comfyui_scenecut.nodes.detect import SceneCutDetect
    from comfyui_scenecut.nodes.viewer import SceneCutViewer
    from comfyui_scenecut.scenecut.media import probe

    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "out"))
    sl = SceneCutDetect.execute(video, "Film / TV", False).args[0]
    out = SceneCutViewer.execute(sl, proxy=True, proxy_height="240", save_report=True)
    assert out.args[0] is sl
    info = out.ui["scenecut_viewer"][0]
    folder = tmp_path / "temp" / info["subfolder"]
    data = json.load(open(folder / info["data"]))
    assert data["source"]["frames"] == 150 and len(data["segments"]) == 3
    assert data["segments"][1]["relation"] == "hard_cut" and len(data["segments"][0]["tiles"]) == 3
    proxy = folder / data["proxy"]["file"]
    assert probe(str(proxy)).frame_count == 150 and data["proxy"]["height"] == 240
    assert all((folder / p).exists() for p in data["sprites"]["pages"])
    report = tmp_path / "out" / info["report"]
    html = report.read_text()
    assert "SceneCutViewer" in html and data["proxy"]["file"] in html
    assert (report.parent / data["proxy"]["file"]).exists()
    # a second run reuses the cached proxy
    mtime = proxy.stat().st_mtime
    SceneCutViewer.execute(sl, proxy=True, proxy_height="240")
    assert proxy.stat().st_mtime == mtime
    assert "report" in str(report) and "three_shots" in str(report)          # {video} in the prefix


def test_viewer_review_fix_and_save(pack, video, omni_ckpt, tmp_path, monkeypatch):
    import folder_paths
    from comfyui_scenecut.nodes import routes
    from comfyui_scenecut.nodes.detect import SceneCutDetect
    from comfyui_scenecut.nodes.viewer import SceneCutSelectedScenes, SceneCutViewer

    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "out"))
    sl = SceneCutDetect.execute(video, "Film / TV", False).args[0]
    info = SceneCutViewer.execute(sl, proxy=True, proxy_height="240").ui["scenecut_viewer"][0]
    data, _ = routes.review_scenes({"data": info["data"]})[1:]
    cut = sl.cuts[0]

    # the cut is right: auto fix leaves it; manual fix on its first frame moves it by one
    assert routes.fix_cut({"data": info["data"], "frame": cut, "mode": "auto"})["to"] == cut
    r = routes.fix_cut({"data": info["data"], "frame": cut, "mode": "manual"})
    assert (r["cut"], r["to"]) == (cut, cut + 1)
    # an edit made earlier is taken into account
    edits = [{"cut": cut, "to": cut + 1, "mode": "manual"}]
    assert routes.fix_cut({"data": info["data"], "frame": cut + 1, "mode": "auto", "edits": edits})["to"] == cut

    review = {"base": data["base"], "edits": edits, "selection": [0, 2]}
    out = SceneCutViewer.execute(sl, proxy=True, proxy_height="240", review=json.dumps(review),
                                 output_images="first frame per scene")
    reviewed, images = out.args
    assert reviewed.cuts[0] == cut + 1 and reviewed.selection == [0, 2]
    assert images.shape[0] == 2 and out.ui["scenecut_viewer"][0]["review"]["state"] == "applied"

    # Export from the viewer: the selected scenes as clips, with the fix applied
    job = routes.start_export({"data": info["data"], "review": review, "scope": "selected", "codec": "h264"})
    routes._EXPORTS[job["id"]].thread.join(timeout=120)
    st = routes._EXPORTS[job["id"]].status()
    assert st["state"] == "done" and st["files"] == ["three_shots_001.mp4", "three_shots_003.mp4"]
    stored = json.load(open(os.path.join(st["folder"], "scenes.json")))
    assert stored["selection"] == [0, 2] and stored["segments"][1]["start"] == cut + 1
    assert [c["frames"] for c in stored["clips"]] == [cut + 1, sl.frame_count - sl.cuts[1]]
    assert st["relative"].startswith("out/scenecut/three_shots/scenes_")
    with pytest.raises(ValueError):                                   # nothing selected
        routes.start_export({"data": info["data"], "review": {"base": data["base"]}, "scope": "selected"})
    # an explicit list (the Selected Scenes player), scene list only
    job = routes.start_export({"data": info["data"], "review": review, "scope": "list", "indices": [1],
                               "codec": "none"})
    routes._EXPORTS[job["id"]].thread.join(timeout=60)
    st = routes._EXPORTS[job["id"]].status()
    assert st["state"] == "done" and st["files"] == [] and os.listdir(st["folder"]) == ["scenes.json"]

    sel = SceneCutSelectedScenes.execute(reviewed, proxy=True, proxy_height="240").ui["scenecut_selected"][0]
    assert sel["selection"] == [0, 2]


def test_images_in_and_out(pack, tmp_path, monkeypatch):
    import folder_paths
    from comfyui_scenecut.nodes.common import resolve_scenes, scene_images
    from comfyui_scenecut.nodes.viewer import SceneCutViewer

    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    rng = np.random.default_rng(0)
    frames = torch.from_numpy(rng.integers(0, 256, (12, 33, 47, 3), dtype=np.uint8)).float() / 255.0
    sl = resolve_scenes(images=frames, fps=12.0)
    assert sl.frame_count == 12 and len(sl.segments) == 1 and float(sl.fps) == 12.0
    back = scene_images(sl, "all frames")
    assert torch.equal((back * 255).round(), (frames * 255).round())          # lossless round trip
    assert tuple(scene_images(sl, "off").shape) == (1, 64, 64, 3)
    out = SceneCutViewer.execute(images=frames, proxy=True, proxy_height="240")
    assert out.args[0].frame_count == 12
    with pytest.raises(ValueError):
        resolve_scenes()


def test_save_scenes_writes_clips(pack, video, omni_ckpt, tmp_path, monkeypatch):
    import av
    import folder_paths
    from comfyui_scenecut.nodes.detect import SceneCutDetect
    from comfyui_scenecut.nodes.files import SceneCutSaveScenes

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    sl = SceneCutDetect.execute(video, "Film / TV", False).args[0]
    sl.selection = [1]
    path, folder = SceneCutSaveScenes.execute(sl, "scenecut/{video}/scenes", "H.264 (mp4)", "medium", True).args
    assert sorted(os.listdir(folder)) == ["scenes.json", "three_shots_002.mp4"]
    with av.open(os.path.join(folder, "three_shots_002.mp4")) as c:
        assert sum(1 for _ in c.decode(video=0)) == sl.segments[1].length
    assert json.load(open(path))["clips"][0]["scene"] == 2
