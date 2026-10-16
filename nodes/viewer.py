from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil

import folder_paths
from comfy_api.latest import io

from ..scenecut.media import probe
from ..scenecut.proxy import cache_key, make_proxy, proxy_name, proxy_size
from ..scenecut.review import apply_review, fingerprint
from ..scenecut.viewer import build_data, make_sprites, report_html, write_data
from .common import (CATEGORY, Progress, Scenes, expand_prefix, fps_input, image_output_input, images_input,
                     images_output, resolve_scenes, scene_images)

log = logging.getLogger("SceneCut")

SUBFOLDER = "scenecut"
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
VIEWER_JS = os.path.join(WEB_DIR, "viewer_core.js")


def temp_dir() -> str:
    d = os.path.join(folder_paths.get_temp_directory(), SUBFOLDER)
    os.makedirs(d, exist_ok=True)
    return d


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def prepare_view(sl, proxy: bool = True, proxy_height: str = "360", thumbs_per_scene: str = "3") -> dict:
    """Proxy, thumbnail sprites, the scene list and the viewer JSON for `sl`, in temp/scenecut.
    Everything is cached: a rerun with the same scene list writes nothing. Returns the UI entry
    ({data, subfolder, type}) plus the data itself under "_data"."""
    src = sl.source.path
    if not os.path.isfile(src):
        raise FileNotFoundError(f"the scene list's video is missing: {src}")
    tmp = temp_dir()
    key = cache_key(src)
    total = sl.frame_count
    h = int(proxy_height) if proxy else 0
    seg_hash = _hash([(s.start, s.end) for s in sl.segments])
    full_hash = _hash(sl.to_dict(include_signals=False))
    data_name = f"view_{key}_{full_hash}_{thumbs_per_scene}_{h or 'np'}.json"
    data_path = os.path.join(tmp, data_name)
    scenes_name = f"scenes_{key}_{full_hash}.json"
    scenes_path = os.path.join(tmp, scenes_name)
    entry = {"data": data_name, "subfolder": SUBFOLDER, "type": "temp"}

    if os.path.isfile(data_path) and os.path.isfile(scenes_path):
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        files = list(data.get("sprites", {}).get("pages", [])) + ([data["proxy"]["file"]] if data.get("proxy") else [])
        if all(os.path.isfile(os.path.join(tmp, x)) for x in files):
            return {**entry, "_data": data}

    progress = Progress(total * (2 if proxy else 1))
    proxy_meta, thumb_source = None, src
    if proxy:
        name = proxy_name(src, h)
        path = make_proxy(src, os.path.join(tmp, name), h, frame_count=total,
                          progress=lambda d, t: progress(d), check_abort=progress.check_abort)
        w, ph = proxy_size(sl.source.width, sl.source.height, h)
        frames = probe(path).frame_count
        if frames != total:
            log.warning("SceneCut: proxy has %d frames, the video %d; thumbnails from the source", frames, total)
        else:
            thumb_source = path
        proxy_meta = {"file": name, "width": w, "height": ph, "frames": frames}

    stem = f"sprites_{key}_{seg_hash}_{thumbs_per_scene}"
    meta_path = os.path.join(tmp, f"{stem}.json")
    sprites = None
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            sprites = json.load(f)
        if not all(os.path.isfile(os.path.join(tmp, p)) for p in sprites["pages"]):
            sprites = None
    if sprites is None:
        offset = total if proxy else 0
        sprites = make_sprites(sl, thumb_source, tmp, stem, per_scene=int(thumbs_per_scene),
                               progress=lambda d, t: progress(offset + d), check_abort=progress.check_abort)
        write_data(sprites, meta_path)

    with open(scenes_path, "w", encoding="utf-8") as f:
        json.dump(sl.to_dict(include_signals=True), f, separators=(",", ":"))
    data = build_data(sl, sprites, proxy_meta)
    data["base"] = fingerprint(sl)
    data["scenesFile"] = scenes_name
    write_data(data, data_path)
    return {**entry, "_data": data}


def load_view(data_name: str) -> tuple[dict, str]:
    """(viewer data, temp folder) for a data file written by prepare_view; the name is checked."""
    if not data_name or os.path.basename(data_name) != data_name or not data_name.endswith(".json") \
            or not data_name.startswith("view_"):
        raise ValueError("bad viewer data name")
    tmp = temp_dir()
    with open(os.path.join(tmp, data_name), encoding="utf-8") as f:
        return json.load(f), tmp


def write_report(data: dict, tmp: str, prefix: str, sl) -> str:
    out_root, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        expand_prefix(prefix, sl), folder_paths.get_output_directory())
    folder = os.path.join(out_root, f"{name}_{counter:05}")
    os.makedirs(folder, exist_ok=True)
    files = list(data.get("sprites", {}).get("pages", []))
    if data.get("proxy"):
        files.append(data["proxy"]["file"])
    for f in files:
        shutil.copyfile(os.path.join(tmp, f), os.path.join(folder, f))
    with open(VIEWER_JS, encoding="utf-8") as f:
        js = f.read()
    title = os.path.basename(sl.source.path)
    with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as f:
        f.write(report_html(data, js, title))
    with open(os.path.join(folder, "scenes.json"), "w", encoding="utf-8") as f:
        json.dump(sl.to_dict(include_signals=False), f, indent=1)
    return os.path.join(subfolder, f"{name}_{counter:05}", "index.html")


def view_inputs() -> list:
    """Inputs shared by the viewer nodes: a scene list or frames, and the proxy settings."""
    return [
        Scenes.Input("scenes", optional=True),
        images_input(),
        fps_input(),
        io.Boolean.Input("proxy", default=True,
                         tooltip="Make a small H.264 copy for playback (cached). Off: thumbnails "
                                 "only, no playback."),
        io.Combo.Input("proxy_height", options=["240", "360", "480", "720"], default="360", advanced=True),
        io.Combo.Input("thumbs_per_scene", options=["1", "3"], default="3", advanced=True),
    ]


class SceneCutViewer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutViewer",
            display_name="SceneCut Viewer",
            category=CATEGORY,
            is_output_node=True,
            search_aliases=["scene timeline", "shot viewer", "cut review", "scene browser", "select scenes"],
            description="Review a scene list: play the video, step frame by frame, jump between cuts, "
                        "see every scene and every rejected cut with the reason and the validator's "
                        "scores. Fix cuts that are a frame or two off, select scenes, and save the "
                        "result with the buttons. Edits and the selection are kept in the workflow "
                        "and applied to the output.",
            inputs=view_inputs() + [
                io.Boolean.Input("save_report", default=False,
                                 tooltip="Also write a standalone HTML report (with the proxy and "
                                         "thumbnails) to the output folder."),
                io.String.Input("report_prefix", default="scenecut/{video}/report", optional=True,
                                advanced=True, tooltip="{video} is the video's file name."),
                image_output_input(),
                io.String.Input("review", default="", optional=True, socketless=True,
                                extra_dict={"hidden": True},
                                tooltip="Edits and selection made in the viewer (managed by the viewer)."),
            ],
            outputs=[
                Scenes.Output(display_name="scenes",
                              tooltip="The scene list with your cut fixes applied and your selection "
                                      "attached (nodes after this use the selected scenes)."),
                images_output(),
            ],
        )

    @classmethod
    def execute(cls, scenes=None, images=None, fps=None, proxy=True, proxy_height="360",
                thumbs_per_scene="3", save_report=False, report_prefix="scenecut/{video}/report",
                output_images="off", review="") -> io.NodeOutput:
        base = resolve_scenes(scenes, images, fps)
        entry = prepare_view(base, proxy, proxy_height, thumbs_per_scene)
        data = entry.pop("_data")
        reviewed, status = apply_review(base, review)
        entry["review"] = status
        if save_report:
            report_data = build_data(reviewed, _sprites_of(data), data.get("proxy"))
            report_data["selection"] = reviewed.selection
            entry["report"] = write_report(report_data, temp_dir(), report_prefix or "scenecut/{video}/report",
                                          reviewed)
        return io.NodeOutput(reviewed, scene_images(reviewed, output_images), ui={"scenecut_viewer": [entry]})


def _sprites_of(data: dict) -> dict | None:
    sp = data.get("sprites")
    if not sp:
        return None
    return {**sp, "sceneTiles": [s.get("tiles", []) for s in data["segments"]]}


class SceneCutSelectedScenes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SceneCutSelectedScenes",
            display_name="SceneCut Selected Scenes",
            category=CATEGORY,
            is_output_node=True,
            search_aliases=["selected shots", "scene player", "shot player", "scene bucket"],
            description="Play the selected scenes one at a time: arrows or the arrow keys move between "
                        "them, with each scene's details beside it. Connected to a SceneCut Viewer it "
                        "follows the selection as you make it. Can open in its own browser tab, which "
                        "stays in sync.",
            inputs=view_inputs() + [image_output_input()],
            outputs=[
                Scenes.Output(display_name="scenes"),
                images_output(),
            ],
        )

    @classmethod
    def execute(cls, scenes=None, images=None, fps=None, proxy=True, proxy_height="360",
                thumbs_per_scene="3", output_images="off") -> io.NodeOutput:
        sl = resolve_scenes(scenes, images, fps)
        entry = prepare_view(sl, proxy, proxy_height, thumbs_per_scene)
        entry.pop("_data")
        entry["selection"] = sl.selection
        return io.NodeOutput(sl, scene_images(sl, output_images), ui={"scenecut_selected": [entry]})
