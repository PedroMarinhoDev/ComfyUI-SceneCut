"""The demo page's data and the README's images, from the demo video.

    cp trailer.mp4 dev/demo/trailer.mp4                 # the demo video (not in the repo)
    python dev/site_assets.py [--device cuda:0]      # site/demo/*, site/viewer_core.js, assets/*
    python dev/comfy_screenshot.py --server URL      # assets/workflow.png (then rerun site_assets.py)

The demo video is the trailer of William Castle's Let's Kill Uncle (1966), in the public domain.
Runs SceneCut on it as the Detect node does (Film / TV preset, with validation), writes the
viewer's proxy, thumbnails and data for the page, and renders the README's banner, buttons and
icons, and a screenshot of the viewer, with headless Chrome at 2x. Needs ComfyUI's Python, the
SceneCut models, transformers (for DINOv3 outside ComfyUI), ffmpeg and google-chrome.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenecut.media import iter_frames_at, probe  # noqa: E402
from scenecut.headless import Models, run  # noqa: E402
from scenecut.presets import get_preset  # noqa: E402
from scenecut.proxy import make_proxy, proxy_size  # noqa: E402
from scenecut.viewer import build_data, make_sprites  # noqa: E402

ASSETS, SITE = ROOT / "assets", ROOT / "site"
DEMO = ROOT / "dev" / "demo" / "trailer.mp4"
SELECTION = [2, 7, 14, 18]         # scenes selected on the page: the robe, the karate title, the kids, the title card
INSPECT, INSPECT_OFFSET = 2, 30    # the scene the README screenshot inspects, and the frame in it
BANNER = [2, 3, 6, 7, 9, 11, 14, 15, 18]   # scenes in the banner
CHROME = "google-chrome"
ACCENT = "#3ECF6E"
THEMES = {
    "light": {"bg": "#FFFFFF", "card": "#F6F8FA", "ink": "#1F2328", "muted": "#59636E", "line": "#D1D9E0", "accent": "#1A7F37"},
    "dark": {"bg": "#0D1117", "card": "#151B23", "ink": "#F0F6FC", "muted": "#9198A1", "line": "#3D444D", "accent": ACCENT},
}
CUT = {"hard_cut": "#4285F4", "sudden_jump": "#AB47BC", "transition": "#FB8C00"}
FONT = "'IBM Plex Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace"


def models_dir() -> Path:
    comfy = Path(os.environ.get("COMFYUI_PATH", Path.home() / "AI" / "ComfyUI"))
    return comfy / "models"


def detect(device: str):
    models = Models.load(str(models_dir()), device=device, allow_download=False)
    return run(str(DEMO), get_preset("film"), models)


def site_data(sl) -> dict:
    out = SITE / "demo"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    proxy = make_proxy(str(DEMO), str(out / "proxy.mp4"), 360, frame_count=sl.frame_count)
    w, h = proxy_size(sl.source.width, sl.source.height, 360)
    sprites = make_sprites(sl, proxy, str(out), "thumbs", per_scene=3)
    data = build_data(sl, sprites, {"file": "proxy.mp4", "width": w, "height": h, "frames": probe(proxy).frame_count})
    data["source"]["path"] = data["source"]["name"]           # no local paths on the page
    data["selection"] = SELECTION
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    shutil.copyfile(ROOT / "web" / "viewer_core.js", SITE / "viewer_core.js")
    return data


def shoot(page: str, width: int, height: int, out: Path, wait_ms: int = 0) -> None:
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "page.html"
        source.write_text(page, encoding="utf-8")
        args = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                "--allow-file-access-from-files", "--autoplay-policy=no-user-gesture-required",
                f"--window-size={width},{height}", f"--screenshot={out}"]
        if wait_ms:
            args.append(f"--virtual-time-budget={wait_ms}")
        subprocess.run(args + [source.as_uri()], check=True, capture_output=True)


def strip_frames(sl, count: int = 9):
    """A frame from the middle of each of `count` scenes, for the banner."""
    picks = [sl.segments[i] for i in BANNER if i < len(sl.segments)][:count]
    frames = [s.start + s.length // 2 for s in picks]
    got = dict(iter_frames_at(str(DEMO), frames, 192, 108))
    import base64
    import io

    from PIL import Image

    out = []
    for s, f in zip(picks, frames):
        buf = io.BytesIO()
        Image.fromarray(got[f]).save(buf, "JPEG", quality=88)
        rel = s.cut.relation if s.cut else "hard_cut"
        out.append(("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), CUT.get(rel, CUT["hard_cut"])))
    return out


def banner(theme: str, frames) -> str:
    t = THEMES[theme]
    tiles = "".join(f'<div class="tile"><i style="background:{c}"></i><img src="{src}"></div>' for src, c in frames)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; width: 1280px; height: 320px; background: {t['bg']}; font-family: {FONT}; color: {t['ink']}; overflow: hidden; position: relative; }}
.bar {{ position: absolute; left: 0; top: 0; width: 8px; height: 320px; background: {ACCENT}; }}
.text {{ position: absolute; left: 64px; top: 58px; width: 560px; }}
h1 {{ margin: 0; font-size: 52px; font-weight: 700; letter-spacing: -1px; }}
h1 span {{ color: {t['accent']}; }}
p {{ margin: 14px 0 0; font-size: 20px; line-height: 1.4; color: {t['muted']}; }}
.built {{ position: absolute; left: 64px; bottom: 30px; font-size: 15px; color: {t['muted']}; letter-spacing: 0.4px; }}
.strip {{ position: absolute; left: 676px; top: 40px; width: 560px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.tile {{ position: relative; border-radius: 6px; overflow: hidden; border: 1px solid {t['line']}; height: 72px; background: {t['card']}; }}
.tile img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.tile i {{ position: absolute; left: 0; top: 0; bottom: 0; width: 4px; }}
.timeline {{ position: absolute; left: 676px; top: 290px; width: 560px; height: 6px; border-radius: 3px; background: {t['line']}; }}
</style></head><body><div class="bar"></div>
<div class="text"><h1>ComfyUI-<span>SceneCut</span></h1>
<p>Find every cut in a video, fix the ones a frame off, pick the scenes you want and export them as clips, natively in ComfyUI.</p></div>
<div class="built">OmniShotCut detection · DINOv3 validation · frame-exact</div>
<div class="strip">{tiles}</div></body></html>"""


def button(label: str, accent: bool, width: int) -> str:
    fill, ink = (ACCENT, "#0D1117") if accent else ("#57606A", "#FFFFFF")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="40" viewBox="0 0 {width} 40" role="img" aria-label="{html.escape(label)}">'
            f'<rect width="{width}" height="40" rx="8" fill="{fill}"/>'
            f'<text x="{width / 2}" y="25.5" text-anchor="middle" font-family="{FONT.replace(chr(39), "")}" font-size="15" font-weight="600" fill="{ink}">'
            f'{html.escape(label)}</text></svg>\n')


ICONS = {
    "detect": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M9 5v14M15 5v14"/>',
    "validate": '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
    "review": '<path d="M3 12h18"/><path d="M8 7v10M16 7v10"/><path d="M11 12l-3-3M11 12l-3 3"/>',
    "select": '<rect x="3" y="4" width="8" height="7" rx="1.5"/><rect x="13" y="4" width="8" height="7" rx="1.5"/><rect x="3" y="13" width="8" height="7" rx="1.5"/><path d="M14.5 16.5l2 2 3.5-3.5"/>',
    "export": '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 17v3h16v-3"/>',
    "frames": '<rect x="3" y="6" width="14" height="12" rx="1.5"/><path d="M7 6V4h14v12h-2"/>',
}


def icon(paths: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>\n')


def viewer_shot(out: Path, data: dict, scene: int = INSPECT, offset: int = INSPECT_OFFSET) -> None:
    """The viewer as the page shows it, large layout, one scene inspected, four selected. Headless
    Chrome does not paint video frames, so the frame under the playhead is shown as an image."""
    import base64
    import io

    from PIL import Image

    seg = data["segments"][scene]
    frame = seg["start"] + offset
    rgb = dict(iter_frames_at(str(DEMO), [frame], 1280, 720))[frame]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=90)
    still = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html, body {{ margin: 0; height: 100%; background: #16161a; }} #app {{ position: fixed; inset: 0; }}
.scv-video img.still {{ max-width: 100%; max-height: 100%; height: 100%; width: auto; display: block; }}</style></head>
<body><div id="app"></div><script type="module">
import {{ SceneCutViewer }} from "{(SITE / 'viewer_core.js').as_uri()}";
const data = await (await fetch("{(SITE / 'demo' / 'data.json').as_uri()}")).json();
const v = new SceneCutViewer(document.getElementById("app"), {{
  standalone: true, resolve: (n) => "{(SITE / 'demo').as_uri()}/" + n, onExport: () => {{}}, onFix: async () => ({{}}) }});
v.load(data);
v.setReview({{ base: data.base, edits: [], selection: data.selection }});
v.focusScene({scene});
v._setFrame({frame}, false);
v.video.style.display = "none";
const img = document.createElement("img");
img.className = "still"; img.src = "{still}";
v.videoBox.prepend(img);
</script></body></html>"""
    shoot(page, 1280, 760, out, wait_ms=4000)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not DEMO.exists():
        sys.exit("put the demo video at dev/demo/trailer.mp4 first")
    sl = detect(a.device)
    data = site_data(sl)
    (ASSETS / "icons").mkdir(parents=True, exist_ok=True)
    for name, (label, accent, width) in {"demo": ("▶  Try the live demo", True, 196), "start": ("Quick start", False, 128),
                                         "workflows": ("Workflows", False, 118)}.items():
        (ASSETS / f"button-{name}.svg").write_text(button(label, accent, width))
    for name, paths in ICONS.items():
        (ASSETS / "icons" / f"{name}.svg").write_text(icon(paths))
    shutil.copytree(ASSETS / "icons", SITE / "assets" / "icons", dirs_exist_ok=True)
    if (ASSETS / "workflow.png").exists():                 # from dev/comfy_screenshot.py
        shutil.copyfile(ASSETS / "workflow.png", SITE / "assets" / "workflow.png")
    frames = strip_frames(sl)
    for theme in THEMES:
        shoot(banner(theme, frames), 1280, 320, ASSETS / f"banner-{theme}.png")
    viewer_shot(ASSETS / "viewer.png", data)
    print(f"{len(data['segments'])} scenes, {len(sl.rejected)} merged boundaries; site/demo and assets written")


if __name__ == "__main__":
    main()
