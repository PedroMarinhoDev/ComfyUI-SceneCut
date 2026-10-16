"""HTTP routes for the frontend: the Load Video file picker and preview, the viewer's cut fixes
and saves, and the standalone Selected Scenes page.

Who may list (and preview) which folders:
  - SCENECUT_BROWSE_ROOTS set (folders separated by the OS path separator): only those folders.
  - ComfyUI listening on loopback only (the default): anywhere the ComfyUI user can read.
  - ComfyUI reachable over the network (--listen): only the input and output folders, unless
    SCENECUT_BROWSE_ROOTS widens it. Directory names are not shared with the network by default.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import shutil
import threading
import uuid

import folder_paths

from ..scenecut.browse import VIDEO_EXTENSIONS, list_dir, system_roots, within
from ..scenecut.export import (CODECS, DEFAULT_CODEC, DEFAULT_QUALITY, JSON_ONLY, QUALITY, ExportCancelled,
                               available_codecs, codec_works, write_export)
from ..scenecut.media import probe
from ..scenecut.proxy import make_proxy, proxy_name
from ..scenecut.review import apply_edits, apply_review, auto_fix, manual_fix, parse
from ..scenecut.scenes import SceneList

log = logging.getLogger("SceneCut")


def _loopback_only() -> bool:
    try:
        from comfy.cli_args import args
    except Exception:  # noqa: BLE001
        return True
    for host in str(getattr(args, "listen", "127.0.0.1") or "127.0.0.1").split(","):
        host = host.strip().strip("[]")
        if host in ("localhost", ""):
            continue
        try:
            if not ipaddress.ip_address(host).is_loopback:
                return False
        except ValueError:
            return False
    return True


def browse_roots() -> tuple[list[str] | None, list[dict]]:
    """(allowed roots or None for anywhere, shortcuts to show in the picker)."""
    env = os.environ.get("SCENECUT_BROWSE_ROOTS", "").strip()
    io_dirs = [("input", folder_paths.get_input_directory()),
               ("output", folder_paths.get_output_directory())]
    if env:
        roots = [os.path.abspath(os.path.expanduser(p)) for p in env.split(os.pathsep) if p.strip()]
        places = [(os.path.basename(p.rstrip(os.sep)) or p, p) for p in roots]
    elif _loopback_only():
        roots = None
        home = os.path.expanduser("~")
        places = [("home", home)]
        for name in ("Videos", "Movies", "Downloads", "Desktop"):
            p = os.path.join(home, name)
            if os.path.isdir(p):
                places.append((name, p))
        places += io_dirs + [(r, r) for r in system_roots()]
    else:
        roots = [p for _, p in io_dirs]
        places = io_dirs
    return roots, [{"name": n, "path": p} for n, p in places if os.path.isdir(p)]


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:  # noqa: BLE001 - not running inside a ComfyUI server (tests, scripts)
        return
    if getattr(PromptServer, "instance", None) is None:
        return
    routes = PromptServer.instance.routes

    @routes.get("/scenecut/browse")
    async def scenecut_browse(request):
        roots, places = browse_roots()
        q = request.rel_url.query
        path = q.get("path", "").strip()
        if not path or not os.path.isdir(os.path.expanduser(path)):
            # Fall back to the folder of a file path, then the first shortcut.
            cand = os.path.dirname(os.path.expanduser(path)) if path else ""
            path = cand if cand and os.path.isdir(cand) else (places[0]["path"] if places else "/")
        try:
            listing = list_dir(path, roots, show_all=q.get("all") == "1",
                               show_hidden=q.get("hidden") == "1")
        except PermissionError as e:
            return web.json_response({"error": str(e), "places": places}, status=403)
        except OSError as e:
            return web.json_response({"error": str(e), "places": places}, status=400)
        listing["places"] = places
        listing["restricted"] = roots is not None
        return web.json_response(listing)

    register_review_routes(routes, web)


# ------------------------------------------------------------------------------------------------
# Helpers shared by the routes (plain functions, so tests can call them)
STATIC_FILES = {"viewer_core.js", "selected_core.js", "export_dialog.js"}
_proxy_locks: dict[str, asyncio.Lock] = {}


def allowed_video(path: str) -> str:
    """The absolute path of a video the browse policy lets the frontend preview, or raise."""
    full = os.path.abspath(os.path.expanduser((path or "").strip().strip('"').strip("'")))
    if not os.path.isfile(full):
        raise FileNotFoundError(f"not found: {full}")
    if os.path.splitext(full)[1].lower() not in VIDEO_EXTENSIONS:
        raise PermissionError("not a video file")
    roots, _ = browse_roots()
    if not within(full, roots):
        raise PermissionError("outside the folders SceneCut may read (see SCENECUT_BROWSE_ROOTS)")
    return full


def review_scenes(body: dict):
    """(base scene list, viewer data, temp folder) for a request from the viewer."""
    from .viewer import load_view

    data, tmp = load_view(str(body.get("data", "")))
    with open(os.path.join(tmp, data["scenesFile"]), encoding="utf-8") as f:
        base = SceneList.from_dict(json.load(f))
    return base, data, tmp


def fix_cut(body: dict) -> dict:
    """Where the cut nearest body["frame"] should go (mode auto or manual), after body["edits"]."""
    base, data, tmp = review_scenes(body)
    sl, _ = apply_edits(base, body.get("edits") or [])
    frame = int(body["frame"])
    if body.get("mode") == "manual":
        return manual_fix(sl, frame)
    proxy = data.get("proxy")
    video = os.path.join(tmp, proxy["file"]) if proxy and os.path.isfile(os.path.join(tmp, proxy["file"])) \
        else base.source.path
    return auto_fix(sl, frame, video)


def reviewed_scenes(body: dict):
    """The scene list as the viewer shows it (its fixes applied, its selection attached)."""
    base, data, _ = review_scenes(body)
    review = parse(body.get("review"))
    if review.get("base") not in (None, data.get("base")):
        raise ValueError("this review was made on a different scene list")
    review["base"] = data.get("base")
    sl, status = apply_review(base, review)
    return sl, status


class ExportJob:
    """One export running on a background thread; the frontend polls its state."""

    def __init__(self, sl, indices, codec, quality, audio):
        from .files import new_export_folder

        self.id = uuid.uuid4().hex[:12]
        self.sl, self.indices, self.codec, self.quality, self.audio = sl, indices, codec, quality, audio
        self.folder, self.relative = new_export_folder(sl)
        self.state = "running"
        self.done = 0
        self.total = sum(sl.segments[i].length for i in indices) if codec != JSON_ONLY else 0
        self.clips_done = 0
        self.error = None
        self.result = None
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"scenecut-export-{self.id}")

    def _progress(self, done, total, clips):
        self.done, self.total, self.clips_done = done, total, clips

    def _check(self):
        if self.cancelled.is_set():
            raise ExportCancelled()

    def _run(self):
        try:
            self.result = write_export(self.sl, self.indices, self.folder, self.codec, self.quality,
                                       self.audio, include_signals=True, progress=self._progress,
                                       check_abort=self._check)
            self.clips_done = len(self.result["clips"])
            self.state = "done"
        except ExportCancelled:
            self.state = "cancelled"
            shutil.rmtree(self.folder, ignore_errors=True)
        except Exception as e:  # noqa: BLE001 - reported to the frontend
            log.exception("SceneCut export failed")
            self.state, self.error = "error", str(e)

    def status(self) -> dict:
        out_dir = folder_paths.get_output_directory()
        d = {"id": self.id, "state": self.state, "done": self.done, "total": self.total,
             "clips": self.clips_done, "clipsTotal": len(self.indices) if self.codec != JSON_ONLY else 0,
             "folder": self.folder, "relative": os.path.join(os.path.basename(out_dir), self.relative),
             "codec": self.codec, "error": self.error}
        if self.result:
            d["files"] = self.result["clips"]
        return d


_EXPORTS: dict[str, ExportJob] = {}


def start_export(body: dict) -> dict:
    sl, status = reviewed_scenes(body)
    scope = body.get("scope", "selected")
    if scope == "selected":
        if not sl.selection:
            raise ValueError("no scenes are selected")
        indices = sl.selected_indices()
    elif scope == "list":                      # an explicit list (the Selected Scenes player)
        indices = sorted({int(i) for i in body.get("indices") or [] if 0 <= int(i) < len(sl.segments)})
        if not indices:
            raise ValueError("no scenes to export")
    else:
        indices = list(range(len(sl.segments)))
    codec = body.get("codec", DEFAULT_CODEC)
    if codec != JSON_ONLY:
        if codec not in CODECS:
            raise ValueError(f"unknown codec {codec}")
        if not codec_works(codec):
            raise ValueError(f"{CODECS[codec].label} is not available on this machine")
    quality = body.get("quality", DEFAULT_QUALITY)
    if quality not in QUALITY:
        quality = DEFAULT_QUALITY
    if not os.path.isfile(sl.source.path):
        raise FileNotFoundError(f"the video is missing: {sl.source.path}")
    job = ExportJob(sl, indices, codec, quality, bool(body.get("audio", True)) and sl.source.has_audio)
    log.info("SceneCut: exporting %d scene%s of %s (%s) to %s", len(indices), "" if len(indices) == 1 else "s",
             os.path.basename(sl.source.path), codec, job.folder)
    for old in [k for k, j in _EXPORTS.items() if j.state != "running"][:-20]:
        _EXPORTS.pop(old, None)
    _EXPORTS[job.id] = job
    job.thread.start()
    return job.status()


def selected_page(query: dict) -> str:
    """The standalone Selected Scenes page; it loads its modules from /scenecut/static/."""
    title = re.sub(r"[<>&\"]", "", query.get("title", "Selected scenes"))[:120]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - SceneCut</title>
<style>html, body {{ margin: 0; height: 100%; background: #141417; color: #e8e8ec; }}
#app {{ position: fixed; inset: 0; }}</style></head>
<body><div id="app"></div>
<script type="module">
import {{ openSelectedPage }} from "./static/selected_core.js";
openSelectedPage(document.getElementById("app"), new URLSearchParams(location.search));
</script></body></html>
"""


def register_review_routes(routes, web) -> None:
    @routes.get("/scenecut/media")
    async def scenecut_media(request):
        try:
            full = allowed_video(request.rel_url.query.get("path", ""))
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)
        except PermissionError as e:
            return web.json_response({"error": str(e)}, status=403)
        return web.FileResponse(full)

    @routes.get("/scenecut/probe")
    async def scenecut_probe(request):
        try:
            full = allowed_video(request.rel_url.query.get("path", ""))
            info = await asyncio.to_thread(probe, full, False)
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)
        except PermissionError as e:
            return web.json_response({"error": str(e)}, status=403)
        except Exception as e:  # noqa: BLE001 - unreadable file
            return web.json_response({"error": f"cannot read this video: {e}"}, status=400)
        d = info.to_dict()
        d["fps_float"] = info.fps_float
        d["size"] = os.path.getsize(full)
        return web.json_response(d)

    @routes.get("/scenecut/preview")
    async def scenecut_preview(request):
        """A browser-playable proxy for a video the browser cannot play itself (made once, cached;
        the viewer reuses it)."""
        from .viewer import SUBFOLDER, temp_dir

        q = request.rel_url.query
        try:
            full = allowed_video(q.get("path", ""))
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)
        except PermissionError as e:
            return web.json_response({"error": str(e)}, status=403)
        height = 360
        name = proxy_name(full, height)
        dst = os.path.join(temp_dir(), name)
        lock = _proxy_locks.setdefault(name, asyncio.Lock())
        try:
            async with lock:
                await asyncio.to_thread(make_proxy, full, dst, height)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"filename": name, "subfolder": SUBFOLDER, "type": "temp"})

    @routes.post("/scenecut/review/fix")
    async def scenecut_fix(request):
        try:
            body = await request.json()
            result = await asyncio.to_thread(fix_cut, body)
        except FileNotFoundError:
            return web.json_response({"error": "The viewer files are gone (ComfyUI empties its temp folder "
                                               "on restart). Run the workflow again."}, status=410)
        except (KeyError, TypeError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(result, status=200 if "error" not in result else 422)

    @routes.get("/scenecut/export/codecs")
    async def scenecut_export_codecs(request):
        codecs = await asyncio.to_thread(available_codecs)
        return web.json_response({"codecs": codecs, "default": DEFAULT_CODEC, "quality": list(QUALITY),
                                  "defaultQuality": DEFAULT_QUALITY, "jsonOnly": JSON_ONLY})

    @routes.post("/scenecut/export")
    async def scenecut_export(request):
        try:
            body = await request.json()
            result = await asyncio.to_thread(start_export, body)
        except FileNotFoundError as e:
            msg = str(e) if "video is missing" in str(e) else ("The viewer files are gone (ComfyUI empties its "
                                                              "temp folder on restart). Run the workflow again.")
            return web.json_response({"error": msg}, status=410)
        except (KeyError, TypeError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(result)

    @routes.get("/scenecut/export/{job}")
    async def scenecut_export_status(request):
        job = _EXPORTS.get(request.match_info["job"])
        if job is None:
            return web.json_response({"error": "no such export"}, status=404)
        return web.json_response(job.status())

    @routes.post("/scenecut/export/{job}/cancel")
    async def scenecut_export_cancel(request):
        job = _EXPORTS.get(request.match_info["job"])
        if job is None:
            return web.json_response({"error": "no such export"}, status=404)
        job.cancelled.set()
        return web.json_response(job.status())

    @routes.get("/scenecut/selected")
    async def scenecut_selected_page(request):
        return web.Response(text=selected_page(dict(request.rel_url.query)), content_type="text/html")

    @routes.get("/scenecut/static/{name}")
    async def scenecut_static(request):
        name = request.match_info["name"]
        if name not in STATIC_FILES:
            return web.Response(status=404)
        from .viewer import WEB_DIR

        resp = web.FileResponse(os.path.join(WEB_DIR, name))
        resp.content_type = "text/javascript"
        return resp
